# src/ivan/web/jobs.py
"""
Single-worker job runner for the web UI.

Every pipeline step is one of the existing CLI scripts run as a subprocess.
Jobs execute strictly one at a time (a threading worker drains a queue), which
is also what keeps SQLite happy: the scripts fork their own worker processes
and the DB has one writer at a time.

Logs go to <log_dir>/<job id>.log so they survive a restart; job metadata is
kept in memory and rebuilt from the log directory on start.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Optional

from dotenv import dotenv_values

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def clean_log_text(raw: str) -> str:
    """Strip ANSI escapes and collapse carriage-return progress updates to their final state."""
    out = []
    for line in _ANSI.sub("", raw).split("\n"):
        out.append(line.rsplit("\r", 1)[-1] if "\r" in line else line)
    return "\n".join(out)


@dataclass
class Job:
    id: str
    name: str
    cmd: list[str]
    status: str = "queued"          # queued | running | done | failed | cancelled
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    returncode: Optional[int] = None
    pid: Optional[int] = None
    meta: dict = field(default_factory=dict)

    @property
    def log_name(self) -> str:
        return f"{self.id}.log"

    @property
    def is_active(self) -> bool:
        return self.status in ("queued", "running")

    @property
    def duration(self) -> Optional[float]:
        if self.started is None:
            return None
        return (self.finished or time.time()) - self.started

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration
        return d


class JobRunner:
    def __init__(self, log_dir: Path, cwd: Path, env: Optional[dict] = None, nice: int = 10):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cwd = Path(cwd)
        self.nice = nice
        self.env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            # Make rich/tqdm behave in a pipe: no colour, no cursor games, fixed width.
            "TERM": "dumb", "NO_COLOR": "1", "COLUMNS": "120",
            **(env or {}),
        }
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._queue: Queue[str] = Queue()
        self._lock = threading.Lock()
        self._procs: dict[str, subprocess.Popen] = {}
        self._load_history()
        threading.Thread(target=self._worker, name="ivan-job-worker", daemon=True).start()

    # ---------------------------------------------------------------- public
    def submit(self, name: str, cmd: list[str], meta: Optional[dict] = None) -> Job:
        job = Job(id=datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
                  name=name, cmd=[str(c) for c in cmd], meta=meta or {})
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
        self._save_meta(job)
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def all(self, newest_first: bool = True) -> list[Job]:
        jobs = [self._jobs[i] for i in self._order]
        return jobs[::-1] if newest_first else jobs

    def active(self) -> list[Job]:
        return [j for j in self.all(newest_first=False) if j.is_active]

    def current(self) -> Optional[Job]:
        running = [j for j in self._jobs.values() if j.status == "running"]
        return running[0] if running else None

    def log_path(self, job: Job) -> Path:
        return self.log_dir / job.log_name

    def read_log(self, job: Job, tail_lines: Optional[int] = None) -> str:
        p = self.log_path(job)
        if not p.exists():
            return ""
        with p.open(errors="replace", newline="") as f:     # keep \r so progress lines collapse
            text = clean_log_text(f.read())
        if tail_lines:
            text = "\n".join(text.split("\n")[-tail_lines:])
        return text

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        proc = self._procs.get(job_id)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            return True
        if job.status == "queued":
            job.status = "cancelled"
            job.finished = time.time()
            self._save_meta(job)
            return True
        return False

    # --------------------------------------------------------------- private
    def _worker(self):
        while True:
            job_id = self._queue.get()
            job = self._jobs[job_id]
            if job.status != "queued":     # cancelled while waiting
                continue
            self._run(job)

    def _run(self, job: Job):
        job.status, job.started = "running", time.time()
        self._save_meta(job)
        cmd = job.cmd
        if self.nice and shutil.which("nice"):
            cmd = ["nice", "-n", str(self.nice), *cmd]
        # Re-read .env for every job so an edited key applies without restarting the server.
        env = {**self.env, **{k: v for k, v in dotenv_values(self.cwd / ".env").items() if v is not None}}
        with open(self.log_path(job), "w", buffering=1) as log:
            log.write(f"$ {' '.join(job.cmd)}\n\n")
            try:
                proc = subprocess.Popen(cmd, cwd=self.cwd, env=env,
                                        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                self._procs[job.id] = proc
                job.pid = proc.pid
                job.returncode = proc.wait()
            except Exception as e:           # bad command, missing interpreter, ...
                log.write(f"\n[runner] failed to start: {e}\n")
                job.returncode = -1
            finally:
                self._procs.pop(job.id, None)
        job.finished = time.time()
        job.status = "done" if job.returncode == 0 else ("cancelled" if job.returncode in (-15, 143) else "failed")
        self._save_meta(job)

    def _meta_path(self, job: Job) -> Path:
        return self.log_dir / f"{job.id}.json"

    def _save_meta(self, job: Job):
        self._meta_path(job).write_text(json.dumps(job.to_dict(), indent=1))

    def _load_history(self):
        for p in sorted(self.log_dir.glob("*.json")):
            try:
                d = json.loads(p.read_text())
                d.pop("duration", None)
                job = Job(**d)
                if job.is_active:            # the process died with the previous server
                    job.status, job.finished = "failed", job.finished or time.time()
                self._jobs[job.id] = job
                self._order.append(job.id)
            except Exception:
                continue

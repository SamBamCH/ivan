# ivan/core/_progress.py
from pathlib import Path
import sys
try:
    from rich.console import Console, Group
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
    from rich.panel import Panel
    from rich.live import Live
except Exception:
    Console = Progress = Group = Panel = Live = None

SCRIPT_NAME = Path(sys.argv[0]).stem

def prefix(script: str, proc_no, sub_no):
    return f"[{script}][{proc_no}][{sub_no}]"

class Reporter:
    def __init__(self, show_footer: bool = True):
        self.console = Console() if Console else None
        self.progress = None
        self.tasks = {}       # dict[key] -> task_id (or shadow meta when no-rich)
        self.footer_renderable = None
        self.live = None
        self.show_footer = bool(show_footer)

    def __enter__(self):
        if Progress:
            self.progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("{task.percentage:>5.1f}%"),
                TimeRemainingColumn(),
                console=self.console,
                transient=False,
                expand=True,
                refresh_per_second=20,
            )
            # group progress + footer (empty first)
            layout = Group(self.progress)
            self.live = Live(layout, console=self.console, transient=False, refresh_per_second=10)
            self.live.__enter__()
            self.progress.__enter__()
        return self

    def __exit__(self, *exc):
        if self.progress:
            self.progress.__exit__(*exc)
        if self.live:
            self.live.__exit__(*exc)

    # ---- task mgmt ----
    def add(self, key, desc, total=0):
        if not self.progress:
            self.tasks[key] = {"desc": desc, "total": total, "completed": 0}
            print(desc)
            return
        self.tasks[key] = self.progress.add_task(desc, total=total)

    def update_desc(self, key, desc):
        if not self.progress:
            return
        if key in self.tasks:
            self.progress.update(self.tasks[key], description=desc)

    def update_total(self, key, total):
        if not self.progress:
            if key in self.tasks: self.tasks[key]["total"] = total
            return
        if key in self.tasks:
            self.progress.update(self.tasks[key], total=total)

    def advance(self, key, step=1):
        if not self.progress:
            return
        if key in self.tasks:
            self.progress.advance(self.tasks[key], step)

    def complete(self, key, desc=None):
        if not self.progress:
            if desc: print(desc)
            return
        if key in self.tasks:
            tid = self.tasks[key]
            # snap to total if we have one
            t = self.progress.tasks[tid]
            final_completed = t.total if t.total else t.completed
            self.progress.update(tid, completed=final_completed)
            if desc:
                self.progress.update(tid, description=desc)

    # ---- phase stack helpers ----
    def init_phase_stack(self, group_key, proc_no):
        """
        Create (or reuse) fixed 6-phase rows keyed by (group_key,phase_no).
        """
        names = {
            1: "sampling field",
            2: "payouts",
            3: "build samples",
            4: "sims (ranking)",
            5: "persist results",
            6: "diagnostics",
        }
        for ph, label in names.items():
            k = (group_key, ph)
            if k not in self.tasks:
                self.add(k, f"{prefix('simulate_contest', proc_no, ph)} {label}", total=0)
            else:
                # ensure description is current
                self.update_desc(k, f"{prefix('simulate_contest', proc_no, ph)} {label}")
                # keep totals as-is; caller can update_total next
        return list(names.keys())

    def reset_phase_stack(self, group_key, proc_no):
        """
        Zero all 6 phases for a new contest and reset descriptions.
        """
        for ph in (1,2,3,4,5,6):
            k = (group_key, ph)
            if not self.progress:
                if k in self.tasks:
                    self.tasks[k]["completed"] = 0
                continue
            if k in self.tasks:
                tid = self.tasks[k]
                self.progress.update(tid, completed=0)  # totals left as-is
                self.update_desc(k, f"{prefix('simulate_contest', proc_no, ph)} " + [
                    "", ""][0])  # desc will be set by caller anyway

    # ---- footer (rolling diagnostics) ----
    def set_footer_text(self, text: str):
        if not self.live or not self.show_footer:
            return
        from rich.text import Text
        self.footer_renderable = Panel(Text(text), title="Last contest recap", border_style="cyan")
        # rebuild layout group
        self.live.update(Group(self.progress, self.footer_renderable))

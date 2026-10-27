# Working in Ivan

Rules for every Claude Code session in this repository. They exist because the docs drifted from the
code more than once; keep them short and keep them true.

## Docs are the current state, git is the past

- `docs/` describes the system **as it runs now**. Any change to how the process works (a timer, a
  button, a script, a flow, a deploy step, a config key, a file on disk) updates the page that
  describes it **in the same commit**. `docs/README.md` lists the pages; each sport has the same
  page set under `docs/<sport>/`, the web apps under `docs/web/`, machines and timers in
  `docs/deployment.md`.
- Dated findings, plans and measurements go under `docs/<sport>/notes/` with the date in the
  filename or the first line. They are not updated later; the page that quotes them is.
- Git history records how things got here. Do not write history into the docs ("until 2026-10-23
  this was...") beyond one line where the reason still matters for the reader.
- Before finishing a task, re-read the pages the change touches and fix what is no longer true.

## Every button is a script

- Everything the web UI does must be runnable from a terminal as a file in `scripts/<sport>/`
  (the button and the script call the same function). A new UI action comes with its script, and
  the page in `docs/web/` names both. This is how a failure is traced when the UI is opaque.

## The checkout is live

- The main checkout at `~/ivan` must stay on `master`: user timers and `serve.py` run from the
  working tree. Work in a worktree, commit there, and let the user bring `master` forward.
- The server (`ivan-server`, see `docs/deployment.md`) also runs a public Jellyfin and Tailscale
  to other machines. Touch only the Ivan app (`/opt/ivan/app`), its `ivan-web` unit and the
  `ivan` user's timers. Never DNS, cloudflared, the firewall or Tailscale.
- `deploy/push.sh` rsyncs the working tree to the server and restarts `ivan-web`, which drops any
  running job and the desk's page state: push between slates, not during one.

## Conventions

- Commands run from the repository root with `venv/bin/python`; every path resolves against the
  repository root.
- No live sim or model change without the backtest's field checks passing (see
  `docs/nfl/validation-and-backtests.md`).

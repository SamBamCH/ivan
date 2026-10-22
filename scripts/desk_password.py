#!/usr/bin/env python3
"""
scripts/desk_password.py

Set the desk password as a hash so the plain password is never on disk.

    scripts/desk_password.py                prompt for a password, print the IVAN_DESK_PASSWORD_HASH line for .env
    scripts/desk_password.py --convert-env  replace a plain IVAN_DESK_PASSWORD= line in .env with its hash (in place)
    scripts/desk_password.py --set          prompt and write/replace the hash line in .env

The hash is pbkdf2-sha256, 600k rounds, random salt (ivan.web.auth).
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
from pathlib import Path

from ivan.core.utils import PROJECT_ROOT
from ivan.web import auth

ENV_FILE = PROJECT_ROOT / ".env"


def write_env(hash_line: str) -> None:
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    lines = [l for l in lines if not re.match(rf"^\s*({auth.ENV}|{auth.ENV_HASH})\s*=", l)]
    lines.append(hash_line)
    tmp = ENV_FILE.with_suffix(".env.tmp")
    tmp.touch(mode=0o600, exist_ok=True); tmp.write_text("\n".join(lines) + "\n"); os.chmod(tmp, 0o600)
    tmp.replace(ENV_FILE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--convert-env", action="store_true"); ap.add_argument("--set", action="store_true")
    a = ap.parse_args()
    if a.convert_env:
        m = next((re.match(rf"^\s*{auth.ENV}\s*=\s*(.*?)\s*$", l) for l in ENV_FILE.read_text().splitlines() if re.match(rf"^\s*{auth.ENV}\s*=", l)), None)
        if not m or not m.group(1):
            print("no plain IVAN_DESK_PASSWORD line in .env"); return 1
        write_env(f"{auth.ENV_HASH}={auth.make_hash(m.group(1))}")
        print(f".env: plain password replaced by {auth.ENV_HASH}"); return 0
    pw = getpass.getpass("desk password: ")
    if len(pw) < 8:
        print("use at least 8 characters"); return 1
    if getpass.getpass("again: ") != pw:
        print("they differ"); return 1
    line = f"{auth.ENV_HASH}={auth.make_hash(pw)}"
    if a.set:
        write_env(line); print(f".env updated with {auth.ENV_HASH}")
    else:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

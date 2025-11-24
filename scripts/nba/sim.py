#!/usr/bin/env python3
# scripts/nfl/sim_all.py
import os, json, argparse
import pandas as pd

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console

from ivan.database.session import init_db, get_session, Base
from ivan.sports.nba.simulate_contest import simulate_one_contest  

def parse_args():
    p = argparse.ArgumentParser(description="Run NBA simulations for each contest in entries_csv")
    p.add_argument("--config", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/nba.json")))
    p.add_argument("-n", type=int, default=1000, help="Number of simulation samples per contest (default: 1000)")
    p.add_argument("-c", type=int, default=10000, help="Field cap (default: 100)")
    p.add_argument("-d", type=bool, default=False, help="Allow duplicates in classic slates (default: False)")
    return p.parse_args()

def main():
    args = parse_args()
    cfg   = json.load(open(args.config, encoding="utf-8"))

    engine  = init_db("nba")
    session = get_session(engine)

    # *** NEW: Clean old diagnostics once at start of entire sim run ***
    from ivan.sports.nfl.simulate_contest import clean_old_diagnostics
    console = Console()

    df = pd.read_csv(
        cfg["paths"]["entries_csv"],
        engine="python",
        usecols=["Contest ID", "Contest Name"],
        dtype={"Contest ID": str, "Contest Name": str},
    )
    df["Contest ID"]   = df["Contest ID"].str.strip()
    df["Contest Name"] = df["Contest Name"].str.strip()
    df = df.dropna(subset=["Contest ID", "Contest Name"]).copy()
    df["Contest ID"] = df["Contest ID"].astype(int)
    df = df.sort_values("Contest ID")
    unique_df = df.drop_duplicates(subset=["Contest Name"], keep="last")

    total_contests = len(unique_df)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task(
            f"[cyan]Processing {total_contests} contests...",
            total=total_contests
        )

        for idx, row in enumerate(unique_df.iterrows(), 1):
            _, row_data = row
            cid  = int(row_data["Contest ID"])
            name = row_data["Contest Name"]

            progress.update(overall_task, description=f"[cyan]Contest {idx}/{total_contests}: {name[:40]}...")

            try:
                simulate_one_contest(
                    session      = session,
                    contest_id   = cid,
                    contest_name = name,
                    config_path  = args.config,
                    field_cap    = args.c,
                    num_sims     = args.n,
                )
                console.print(f"✓ Completed contest {cid} ('{name}')")
            except Exception as e:
                console.print(f"[red]⚠️  Skipped contest {cid} ('{name}') due to error: {e}[/red]")

            progress.advance(overall_task)

    console.print(f"\n[bold green]✓ All {total_contests} contests completed![/bold green]")

if __name__ == "__main__":
    main()

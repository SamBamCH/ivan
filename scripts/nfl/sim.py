#!/usr/bin/env python3
# scripts/nfl/sim_all.py
import os, json, argparse
import pandas as pd

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console

from ivan.core.config import config_path, load_config
from ivan.database.session import init_db, get_session, Base
from ivan.sports.nfl.simulate_contest import simulate_one_contest  

def parse_args():
    p = argparse.ArgumentParser(description="Run NFL simulations for each contest in entries_csv")
    p.add_argument("--config", default=str(config_path("nfl")), help="path to nfl json config file")
    p.add_argument("--entries", default=None, help="entries CSV (default: paths.entries_csv from config, or $IVAN_ENTRIES_CSV)")
    p.add_argument("-n", type=int, default=1000, help="Number of simulation samples per contest (default: 1000)")
    p.add_argument("-c", type=int, default=10000, help="Field cap (default: 100)")
    p.add_argument("-d", action="store_true", help="Allow duplicates in classic slates (default: False)")
    return p.parse_args()

def main():
    args = parse_args()
    cfg   = load_config("nfl", args.config)
    entries_csv = args.entries or cfg["paths"]["entries_csv"]

    engine  = init_db("nfl", cfg.get("database_url"))
    Base.metadata.create_all(engine)
    session = get_session(engine)

    # *** NEW: Clean old diagnostics once at start of entire sim run ***
    from ivan.sports.nfl.simulate_contest import clean_old_diagnostics
    console = Console()
    console.print("🧹 Cleaning old diagnostic folders...")
    clean_old_diagnostics(days_to_keep=1)
    console.print()

    df = pd.read_csv(
        entries_csv,
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
                    use_classic_dupes = args.d,
                )
                console.print(f"✓ Completed contest {cid} ('{name}')")
            except Exception as e:
                console.print(f"[red]⚠️  Skipped contest {cid} ('{name}') due to error: {e}[/red]")

            progress.advance(overall_task)

    console.print(f"\n[bold green]✓ All {total_contests} contests completed![/bold green]")

if __name__ == "__main__":
    main()

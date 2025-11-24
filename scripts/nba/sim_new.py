from ivan.sports.nba.refactor.pre_lock.lineup_generation import generate_lineups, deduplicate_lineups
from ivan.sports.nba.refactor.pre_lock.contest_simulation import simulate_one_contest
import argparse
import logging
import pandas as pd

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console


def main():
    p = argparse.ArgumentParser(description="sim flow refactored")
    p.add_argument("--num_lineups", default=25000, type=int, help="number of lineups to generate before simming")
    p.add_argument("--num_jobs", default=8, type=int, help="number of jobs that can run in parallel on this machine.")
    p.add_argument("--fresh", action="store_true", help="whether to clear all lineups in DB prior to run or not. Typically for if there are huge news updates.")

    args = p.parse_args()
    #generate_lineups
    generate_lineups(args.num_lineups, args.num_jobs)
    deduplicate_lineups()
    # #sim
    # console = Console()

    # df = pd.read_csv(
    #     '/home/sam-pratt/ivan/data/nba/input/entries.csv',
    #     engine="python",
    #     usecols=["Contest ID", "Contest Name"],
    #     dtype={"Contest ID": str, "Contest Name": str}
    # )
    # df["Contest ID"]   = df["Contest ID"].str.strip()
    # df["Contest Name"] = df["Contest Name"].str.strip()
    # df = df.dropna(subset=["Contest ID", "Contest Name"]).copy()
    # df["Contest ID"] = df["Contest ID"].astype(int)
    # df = df.sort_values("Contest ID")
    # unique_df = df.drop_duplicates(subset=["Contest Name"], keep="last")
    # df["Contest ID"]   = df["Contest ID"].str.strip()
    # df["Contest Name"] = df["Contest Name"].str.strip()
    # df = df.dropna(subset=["Contest ID", "Contest Name"]).copy()
    # df["Contest ID"] = df["Contest ID"].astype(int)
    # df = df.sort_values("Contest ID")
    # unique_df = df.drop_duplicates(subset=["Contest Name"], keep="last")

    # total_contests = len(unique_df)

    # with Progress(
    #     SpinnerColumn(),
    #     TextColumn("[bold blue]{task.description}"),
    #     BarColumn(),
    #     TaskProgressColumn(),
    #     TimeRemainingColumn(),
    #     console=console,
    # ) as progress:
    #     overall_task = progress.add_task(
    #         f"[cyan]Processing {total_contests} contests...",
    #         total=total_contests
    #     )

    #     for idx, row in enumerate(unique_df.iterrows(), 1):
    #         _, row_data = row
    #         cid  = int(row_data["Contest ID"])
    #         name = row_data["Contest Name"]

    #         progress.update(overall_task, description=f"[cyan]Contest {idx}/{total_contests}: {name[:40]}...")

    #         try:
    #             simulate_one_contest(
    #                 contest_id   = cid,
    #                 contest_name = name,
    #                 field_cap    = args.c,
    #                 num_sims     = args.n,
    #             )
    #             console.print(f"✓ Completed contest {cid} ('{name}')")
    #         except Exception as e:
    #             console.print(f"[red]⚠️  Skipped contest {cid} ('{name}') due to error: {e}[/red]")

    #         progress.advance(overall_task)

    # console.print(f"\n[bold green]✓ All {total_contests} contests completed![/bold green]")

            
#choose

main()
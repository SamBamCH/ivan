import logging
from datetime import datetime, timedelta

from ivan.sports.nfl.paths import CHROME_PROFILE_DIR
from ivan.sports.common.draftkings import build_requests_session_from_chrome, download_contest_dataframe
from ivan.database.session import init_db, get_session
from ivan.sports.nfl.models import (
    ShowdownContestMetadata, 
    HistoricalContest, 
    ClassicContestMetadata,
)
from ivan.sports.nfl.historical_contest import process_showdown_contest, process_classic_contest

log = logging.getLogger(__name__)


def is_showdown_contest(contest):
    """Determine if a contest is a showdown contest."""
    if hasattr(contest, 'contest_type') and contest.contest_type:
        return contest.contest_type == 'Showdown Captain Mode'
    return False


def select_draft_groups_interactively(session, to_process):
    """
    Interactively select draft groups to process for classic contests.
    
    Shows the user the top 10 classic contests by total entries and allows them to
    select draft groups one by one until they type 'exit'.
    Note: Showdown contests are processed automatically and don't require selection.
    
    Returns:
        list: Selected draft_group_ids
    """
    selected_draft_groups = []
    
    # Get unique draft groups with their contest information
    draft_group_info = {}
    for hc in to_process:
        draft_group_id = hc.draft_group_id
        if draft_group_id not in draft_group_info:
            draft_group_info[draft_group_id] = {
                'draft_group_id': draft_group_id,
                'contests': [],
                'max_entries': 0,
                'total_entries': 0
            }
        
        draft_group_info[draft_group_id]['contests'].append(hc)
        # Track the highest entry contest for this draft group
        if hasattr(hc, 'total_entries') and hc.total_entries:
            draft_group_info[draft_group_id]['max_entries'] = max(
                draft_group_info[draft_group_id]['max_entries'],
                hc.total_entries
            )
            draft_group_info[draft_group_id]['total_entries'] += hc.total_entries
    
    print("\n" + "="*80)
    print("CLASSIC DRAFT GROUP SELECTION")
    print("(Showdown contests will be processed automatically)")
    print("="*80)
    
    while True:
        # Filter out already selected draft groups
        available_groups = {
            k: v for k, v in draft_group_info.items() 
            if k not in selected_draft_groups
        }
        
        if not available_groups:
            print("\nAll draft groups have been selected!")
            break
        
        # Sort by max entries (highest first) and take top 10
        sorted_groups = sorted(
            available_groups.items(),
            key=lambda x: x[1]['max_entries'],
            reverse=True
        )[:10]
        
        # Display options
        print(f"\nCurrently selected: {len(selected_draft_groups)} draft group(s)")
        if selected_draft_groups:
            print(f"Selected draft groups: {selected_draft_groups}")
        
        print("\nTop contests by total entries (showing up to 10):")
        print("-" * 80)
        print(f"{'#':<4} {'Draft Group ID':<20} {'Contests':<12} {'Max Entries':<15} {'Total Entries':<15}")
        print("-" * 80)
        
        for idx, (draft_group_id, info) in enumerate(sorted_groups, 1):
            print(f"{idx:<4} {draft_group_id:<20} {len(info['contests']):<12} "
                  f"{info['max_entries']:<15,} {info['total_entries']:<15,}")
        
        print("-" * 80)
        print("\nEnter the number (1-10) to select a draft group, or type 'exit' to finish:")
        
        user_input = input("> ").strip().lower()
        
        if user_input == 'exit':
            print("\nExiting selection.")
            break
        
        try:
            choice_idx = int(user_input)
            if 1 <= choice_idx <= len(sorted_groups):
                selected_group_id = sorted_groups[choice_idx - 1][0]
                selected_draft_groups.append(selected_group_id)
                print(f"\n✓ Added draft group: {selected_group_id}")
            else:
                print(f"\n✗ Invalid choice. Please enter a number between 1 and {len(sorted_groups)}")
        except ValueError:
            print("\n✗ Invalid input. Please enter a number or 'exit'")
    
    print("\n" + "="*80)
    print(f"Selection complete: {len(selected_draft_groups)} classic draft group(s) selected")
    print("="*80 + "\n")
    
    return selected_draft_groups


def run_pipeline(
    user_data_dir: str = None,
    profile_directory: str = None,
    headless: bool = False,
    days_back: int = 7,
):
    """
    Main pipeline to process showdown contests.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    engine = init_db("nfl")
    session = get_session(engine)
    
    # Build DraftKings session
    sess = build_requests_session_from_chrome(
        user_data_dir=user_data_dir,
        profile_directory=profile_directory,
        headless=headless,
    )
    
    # Find unprocessed contests
    now = datetime.utcnow()
    cutoff = now - timedelta(days=days_back)

    # Query both processed showdown and classic contests to exclude them
    processed_showdown = session.query(ShowdownContestMetadata.contest_id).subquery()
    processed_classic = session.query(ClassicContestMetadata.contest_id).subquery()

    to_process = (
        session.query(HistoricalContest)
        .filter(
            HistoricalContest.date >= cutoff,
            HistoricalContest.date <= now,
            ~HistoricalContest.contest_id.in_(processed_showdown),
            ~HistoricalContest.contest_id.in_(processed_classic),
            HistoricalContest.total_entries >= 5000,  # Filter by minimum entries
        )
        .order_by(HistoricalContest.date.desc())
        .all()
    )
    
    log.info(f"Found {len(to_process)} contests to process")

    # Separate showdown and classic contests
    # Note: Adjust the logic below if your contest identification differs
    showdown_contests = [
        hc for hc in to_process 
        if is_showdown_contest(hc)
    ]
    classic_contests = [
        hc for hc in to_process 
        if not is_showdown_contest(hc)
    ]
    
    log.info(f"Found {len(showdown_contests)} showdown contests (will process all)")
    log.info(f"Found {len(classic_contests)} classic contests (user selection required)")
    
    # Interactive draft group selection for classic contests only
    if classic_contests:
        selected_draft_groups = select_draft_groups_interactively(session, classic_contests)
        
        # Filter classic contests to only include selected draft_groups
        filtered_classic_contests = [
            hc for hc in classic_contests 
            if hc.draft_group_id in selected_draft_groups
        ]
    else:
        selected_draft_groups = []
        filtered_classic_contests = []
    
    # Combine showdown (all) and classic (filtered) contests
    to_process = showdown_contests + filtered_classic_contests

    to_process = to_process #added just for testing
    
    log.info(f"Will process {len(showdown_contests)} showdown contests + "
             f"{len(filtered_classic_contests)} classic contests = "
             f"{len(to_process)} total contests")
    
    success_count = 0
    for hc in to_process:
        log.info(f"Processing contest {hc.contest_id} from {hc.date}")
        
        try:
            df = download_contest_dataframe(sess, hc.contest_id)
            
            if df is None or df.empty:
                log.warning(f"Skipping contest {hc.contest_id}: no data")
                continue
            
            # Route to appropriate processor
            if is_showdown_contest(hc):
                success = process_showdown_contest(session, hc, df)
            else:
                success = process_classic_contest(session, hc, df)
            
            if success:
                success_count += 1
            
                
        except Exception as e:
            log.exception(f"Error processing contest {hc.contest_id}")
            continue
    
        
    log.info(f"Pipeline complete: {success_count}/{len(to_process)} contests processed")
    session.close()

def main() -> None:
    run_pipeline(
        user_data_dir=str(CHROME_PROFILE_DIR),
        profile_directory="Default",  # or "Profile 1", etc.
        headless=False,
    )

if __name__ == "__main__":
    main()

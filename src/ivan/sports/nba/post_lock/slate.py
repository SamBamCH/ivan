from typing import Dict, List, Tuple
from sqlalchemy.orm import Session
from ivan.sports.nba.post_lock.entry_parsing import OriginalEntry
from ivan.sports.nba.post_lock.contest import lateswap_one_contest
from ivan.sports.common.draftkings import build_requests_session_from_chrome
from ivan.sports.nba.post_lock.utils import _build_name_map
from ivan.sports.nba.paths import CHROME_PROFILE_DIR


def lateswap_slate(
    session: Session,
    entries: Dict[int, List['OriginalEntry']],
    num_candidates_per_entry: int = 10,
    num_field_samples: int = 5000,
    num_sims: int = 1000
) -> Tuple[List[Tuple['OriginalEntry', List[int], float, Dict]], List[Tuple[int, int, int, List[int], float, Dict]]]:
    
    http_sess = build_requests_session_from_chrome(
        user_data_dir=str(CHROME_PROFILE_DIR),
        profile_directory="Default",
        headless=False
    )
    all_results: List[Tuple['OriginalEntry', List[int], float, Dict]] = []
    all_candidate_rows: List[Tuple[int, int, int, List[int], float, Dict]] = []
    name_map = _build_name_map(session)
    for contest_id, group in entries.items():
        result, cand_rows = lateswap_one_contest(
            session=session,
            contest_id=contest_id,
            entries=group,
            http_sess=http_sess,
            name_map=name_map,
            num_candidates_per_entry=num_candidates_per_entry,
            num_field_samples=num_field_samples,
            num_sims=num_sims,
        )
        if result:
            all_results.extend(result)
        if cand_rows:
            all_candidate_rows.extend(cand_rows)

    return all_results, all_candidate_rows

    
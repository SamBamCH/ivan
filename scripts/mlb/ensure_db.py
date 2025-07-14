from ivan.database.management import clear_sims
from get_contest_ids import main           # ↪ sibling in the same folder
from get_payouts import fetch_and_store_all_payouts  # ↪ likewise
# 1) clear lineup_simulation_results table
clear_sims()
# 2) get_contest_ids
main()
# 3) get_payouts
fetch_and_store_all_payouts()
# 4) check_lineups?
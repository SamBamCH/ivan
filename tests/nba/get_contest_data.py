from ivan.sports.common.draftkings import build_requests_session_from_chrome, download_contest_dataframe

contest_id = int(184473377)
session = build_requests_session_from_chrome(
    user_data_dir="/home/sam-pratt/ivan/chrome-selenium/dk-profile",
    profile_directory="Default",
    headless=False,
)

df = download_contest_dataframe(sess=session, contest_id=contest_id)

print(df)
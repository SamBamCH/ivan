from ivan.sports.nfl.historical_contest import run_pipeline

def main() -> None:
    run_pipeline(
        user_data_dir=r"C:\Users\ivan-\ivan\chrome-selenium\dk-profile",
        profile_directory="Default",  # or "Profile 1", etc.
        headless=False,
    )

if __name__ == "__main__":
    main()

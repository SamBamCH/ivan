from sqlalchemy import create_engine, text

# 1) Open your DB
engine = create_engine(r"sqlite:///I:/data/mlb/mlb_history.sqlite", echo=False)

# 2) Run one UPDATE to overwrite every contest_date
PLACEHOLDER = "2000-01-01 00:00:00"
with engine.begin() as conn:
    conn.execute(
        text("UPDATE slate_kde_model SET contest_date = :ph"),
        {"ph": PLACEHOLDER}
    )

print("All contest_date values set to placeholder.")
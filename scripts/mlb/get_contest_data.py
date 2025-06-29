# src/ivan/sports/mlb/historical/get_contest_data.py

import sys
import time
import requests
from ivan.database.session import init_db, get_session, Base

from ivan.sports.mlb.historical.api import (
    upsert_contest_metadata,
    fetch_all_lineup_names
)
from ivan.sports.mlb.historical.processing import (
    process_entry_for_metrics
)
from ivan.sports.mlb.historical.storage import (
    write_payout_breakpoint,
    write_generalized_metrics
)

def _parse_cookie_header(raw_cookie_header: str) -> dict:
    """
    Parse a raw 'Cookie' header string into a dict of {cookie_name: cookie_value}.
    """
    cookies = {}
    for part in raw_cookie_header.split(';'):
        if '=' in part:
            name, value = part.strip().split('=', 1)
            cookies[name] = value
    return cookies

def _make_api_session(raw_cookie_header: str) -> requests.Session:
    """
    Create a requests.Session with default DK headers and all cookies extracted
    from the provided raw Cookie header string.
    """
    session = requests.Session()
    session.headers.update({
        "Accept":          "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin":          "https://www.draftkings.com",
        "Referer":         "https://www.draftkings.com/",
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    })
    cookies = _parse_cookie_header(raw_cookie_header)
    session.cookies.update(cookies)
    return session

def rebuild_contest_data(draft_group_id: int,contest_id: int, raw_cookie_header: str):
    engine = init_db("mlb")
    Base.metadata.create_all(engine)
    db = get_session(engine)

    api_session = _make_api_session(raw_cookie_header)

    # draft_group_id, contest_date = upsert_contest_metadata(
    #     db, contest_id, api_session
    # )
    # 2️⃣ Fetch *every* lineup by CSV export (uncapped)
    roster_lists = fetch_all_lineup_names(
        draft_group_id,
        contest_id,
        api_session
    )

    print(roster_lists)

    # previous_payout = None
    # P_list, O_list, S_list, E_list = [], [], [], []

    # all_entries.sort(key=lambda e: e["rank"])
    # for entry in all_entries:
    #     print(entry)
    #     entry_key           = entry["entryKey"]
    #     rank                = entry["rank"]
    #     lineup_id           = entry["lineupId"]
    #     username            = entry["userName"]
    #     fantasy_points      = float(entry["fantasyPoints"])
    #     payout              = float(entry.get("winningValue", 0.0))

    #     if previous_payout is None or payout != previous_payout:
    #         write_payout_breakpoint(
    #             db=db,
    #             contest_id=contest_id,
    #             entry_id=entry_key,
    #             lineup_id= lineup_id,
    #             username= username,
    #             rank=rank,
    #             fantasy_points=fantasy_points,
    #             payout=payout
    #         )
    #         previous_payout = payout

    #     proj_sum, own_sum, stack_type, total_own = process_entry_for_metrics(
    #         db=db,
    #         draft_group_id=draft_group_id,
    #         entry_key=entry_key,
    #         contest_date=contest_date,
    #         api_session=api_session
    #     )

    #     P_list.append(proj_sum)
    #     O_list.append(own_sum)
    #     S_list.append(stack_type)
    #     E_list.append(total_own)

    #     time.sleep(0.05)

    # generalized = {"P": P_list, "O": O_list, "S": S_list, "E": E_list}
    # write_generalized_metrics(
    #     db=db,
    #     contest_id=contest_id,
    #     metrics_dict=generalized
    # )

    # db.close()
    # print(f"Finished contest {contest_id}: {len(all_entries)} entries processed.")

if __name__ == "__main__":
    # need at least a draft_group_id and one contest_id
    if len(sys.argv) < 3:
        print("Usage: python get_contest_data.py <draft_group_id> <contest_id1> [contest_id2 …]")
        sys.exit(1)

    raw_cookie = 'LID=1; VIDN=4322402106; SINFN=PID=&AOID=&PUID=11085612&SSEG=&GLI=0&LID=1&site=US-DK; knownUser=true; _tt_enable_cookie=1; _svsid=867b7248a41218b6297d5da9dd60ae07; _hjSessionUser_2150570=eyJpZCI6ImJhZTA1ZjM3LTNkODMtNTAzZS1iNjI2LTM3NDhiMzZmYmQ2ZCIsImNyZWF0ZWQiOjE3MDkzOTg1ODMxMjMsImV4aXN0aW5nIjp0cnVlfQ==; _tgpc=1ae77fa8-6808-5939-810d-d5c62f01d2b1; _ga_VFMLT16C8W=GS1.2.1715790344.1.0.1715790344.0.0.0; _ga_2M5GYNY1YS=GS1.1.1715790343.1.0.1715790351.52.0.0; cto_bundle=DrlBE19oUGg5V25mT0tiTzc5QXRKVXUwVlk0WE9HaFRFTFlyczlsdjIyc29pSWk5RzViYlExQ0clMkJ0ZjNJaWNJM3lPMFNHODFyZmpYRnVpQU52dU1TWkIlMkJOMiUyRkxTTHBlR0tBbzhybk5vNzhSMHhpVnlFejd6MGVPdUJoSFowUTZLc2ttZW90RGFhWHVGODBwcU1TT3A4a3RsWVJidnR0NyUyQldNZkE1NGs4MHNVcTdwZE9YRGhuUXU2MXFYNTFuUmNTMFhWNyUyRmRpU2JtTjNtVlZnSndGU0QzJTJCTHQ4NVJUbnBNRFF5bnZIOUFDeHVkY1FseWVUY0k4QkVjbzBUSFNSdWlhRHZ3QWplcnpGV0UzRHRRVTcwUEYwZHZLZyUzRCUzRA; _hjSessionUser_2578739=eyJpZCI6ImIwYWM3YTIzLThmNTUtNTg0NS1iYWIzLTJmMDNiYjRhY2E5NyIsImNyZWF0ZWQiOjE3MDg3MTE0OTE5MzYsImV4aXN0aW5nIjp0cnVlfQ==; _ga_P2HEC98DVL=GS1.1.1731095784.4.0.1731095784.0.0.0; _ga_2MSNZ8X82V=GS1.1.1731095784.2.0.1731095784.0.0.0; _ga_5ZPQC83E2T=GS1.1.1731095784.2.0.1731095784.0.0.0; _ttp=WbPDhbHj5ff_2XkgvreiD42cJOi.tt.1; site=US-DK; _axwrt=213517a0-8a0b-47d9-b2fe-ec02219efa62; _ga_QG8WHJSQMJ=GS1.1.1741242686.1327.1.1741242690.56.0.0; _scid=FkxT-DzpjaCXmmcl5qi6ig3RH3H0IGHzoZDXPQ; __ssid=8a04c9546e88cf0187dc8e607de90e0; GC-HTML5-UUID=64052C79-ABDA-4197-9B84-1BE6A06EB2AE; _gcl_au=1.1.1343523600.1746877558; _fbp=fb.1.1748669538908.549758242880601285; _tglksd=eyJzIjoiMTc2OTIxYTUtMjU1NS01NzhlLWFmYWEtZDBlMGIyZTVkZDM4Iiwic3QiOjE3NDg5OTIxMDAxNjQsInNvZCI6Ind3dy5kcmFmdGtpbmdzLmNvbSIsInNvZHQiOjE3NDg5ODk5MDg0MDUsInNvZHMiOiJjIiwic29kc3QiOjE3NDg5OTIxMTIwMzl9; SN=1223548523; uk=1; jws_sb=eyJhbGciOiJodHRwOi8vd3d3LnczLm9yZy8yMDAxLzA0L3htbGRzaWctbW9yZSNobWFjLXNoYTI1NiIsImtpZCI6ImdZWkxtVTVLS1FtSGpsUWpURE95NkE0Y1cvNEdlVy9Xb3B6NGJpOEtvMFU9IiwidHlwIjoiSldUIiwidmVyc2lvbiI6MX0.eyJzZSI6IlVTLUlMLVNCIiwicHNDbGllbnRLZXkiOiIwN2Y1MDExZC03ZDYwLTRlN2ItOWY2Zi1jZTlmMjg0NzBkNWIiLCJuYmYiOjE3NDk5NDExODYsImV4cCI6MTc0OTk0MTU0NiwiaXNzIjoidXJuOmRrL2tyb25vcyIsImF1ZCI6InVybjpkayJ9.FPjG7mRhJxCKQQyZygUX0tUUl6ATX_vlQPe8hZgPbzo; jws_gs=eyJhbGciOiJodHRwOi8vd3d3LnczLm9yZy8yMDAxLzA0L3htbGRzaWctbW9yZSNobWFjLXNoYTI1NiIsImtpZCI6ImdZWkxtVTVLS1FtSGpsUWpURE95NkE0Y1cvNEdlVy9Xb3B6NGJpOEtvMFU9IiwidHlwIjoiSldUIiwidmVyc2lvbiI6MX0.eyJ2aWQiOiI0MzIyNDAyMTA2Iiwic2UiOiJVUy1JTC1TQiIsImdzU3RhdGUiOiJBY3RpdmUiLCJnc0NsaWVudEtleSI6IjA3ZjUwMTFkLTdkNjAtNGU3Yi05ZjZmLWNlOWYyODQ3MGQ1YiIsImdzRXhwaXJhdGlvbiI6IjIwMjUtMDYtMTQgMjM6MTY6MjZaIiwibmJmIjoxNzQ5OTQxMTg2LCJleHAiOjE3NDk5NDI5ODYsImlzcyI6InVybjpkay9rcm9ub3MiLCJhdWQiOiJ1cm46ZGsifQ.11zrW2ZEUM916PJ-47FIWSZTiPFSaRRdESSV2SEEouk; _dpm_id.16f4=f6f3e449-cca2-4997-9e80-4ef89a1d6dba.1719692680.105.1749941188.1749316733.5db028f5-2c3b-43e2-b3f1-a6b41881b706; _ScCbts=%5B%22299%3Bchrome.2%3A2%3A5%22%5D; _gid=GA1.2.1178891250.1750110855; _sctr=1%7C1750222800000; PRV=3P=1&V=4322402106&E=1750373192; _csrf=33a93500-f2c8-4a41-9019-5b87d472b3e1; notice_behavior=implied,us; mlc=true; STIDN=eyJDIjoxMjIzNTQ4NTIzLCJTIjoxMDExMjI5NDU1OTEsIlNTIjoxMDUxNDE0MzU3MTgsIlYiOjQzMjI0MDIxMDYsIkwiOjEsIkUiOiIyMDI1LTA2LTE5VDEyOjE1OjU3LjYxMjAzODNaIiwiU0UiOiJVUy1ESyIsIlVBIjoidEtJZTBISk9XUk9NRThNUUlKT1NKL3JLcnpBQjMwMmw1U3FTaHNEYnYydz0iLCJESyI6IjJiMWNmYzUxLWZhZTktNGQ1Mi1iNTNjLTZmNTEwOGI3NWZlZSIsIkRJIjoiZTI1NmM2MjMtNTJkMC00OGYwLWFkN2YtNDI3MmJjMGIzOWM3IiwiREQiOjY3MTY2NjU3ODE1fQ==; STH=a74dda0f5015ce83df2ef0ff80d55c70547de08ae15a0eff6de895abdc8de9a2; ak_bmsc=EDE66C4D46F8F2BE05F2533878B92ADA~000000000000000000000000000000~YAAQGXsyFwvUv3mXAQAAgYsCiBxRU3nV4w+ZAZycPXoCUjJIbSI3BRe+Fh+HhfwCKlmQ5kt+YBktjtVAz3QdUZU5WwHHO12eBeTEGdxa9fgxqUXZoBRuebEEH3aEU9BVIqe9WCR5TVkBNbHQdo7hodWFlZrEF2auWehO9yhw8RRId+bNuQZjngP3Oh5QL3lB8l15SwCtvaQmkcV6rf/geiDOPJBTgLmefVHPA5oSFUZGNOBXoFrwHm3okCxOtCJ11TGhrrSwsaGHAiT2+LLf+IqWrCCt0Rz/d3ujfUdXg7E0fue+B9A+wYM7K4HObkphH5F19LpUi+Upa2XrieQiWRX/Poy+b4fL3s0D0KH2WyOCxRR5gZ5bxHHxUa4T+84A7+Px8ZAnAZAnm4I6TUc=; _abck=F1E6A7F4BFD1E2C449F696DC2C7C0207~0~YAAQGXsyF+zUv3mXAQAAPsICiA5PjxR9oOAomqD9qdDCPyn4zQvjx0uQCcGWGFcsyiGigFbT3dKbn6dXS6UboNjT09h3pTm+8BwVGI/09ZLRBzHdeTCcMxGOhCxRrBLQE1m8RpArHEFGQfOpEX49fVGnYVVXOIzIAwElVgsY93+eZ72lyQa/xLxMPDRoavQclM4JHZUR6+j69iV/6AeQcC0hbdq9gVkWezqMX3kDTIxS3efd7XY5aCCiyFaje0jfOrlepVypNO1dpa+26kPTjBX/yJi/aVb6UTIXI3hEEBm4hBSQYAjNCgnrOgCatzjRDcZSg+xLuKyJ2x4c0APbjB3n930TO8O8pHkMrBqcOSaKbScmKJdj7TNMAFElT6AitfwQK2oWp0TJ7H6n/Tej8ROogFGsCVMxyYL6HDUgwJqKFQNpN8dNJ2o0bwziMLhRh8qd6aLq+H3zEiOh5ltBsYEDJr4VpcVMVrZzUWGoRbpHrIVaGhdi8vjaPIPd7gL+zLUW5NFlPFqjjHQd1QxhDxpqWOS0uCPRlp8oHTYe+L8E3mGvVKehSXi09gPQlWDgENk993gmd60CFndG2KFEinOW5DvWoJcFWeivJAydA0JVlb3UqGs05BNIiQfN8fpxZsvfty1ASv2Bn7cX+En5ocFwj9Tl88SxFw05gPn/0MhEOnkA41xHcbtCEioPWH9yX4+A0deeiEMHCZqwKckwWBZufPlwqzctpKGECYTRUyg4SknQUNH/a6QCoUdBFuFZBIIU11SkDkzl2Kn9GZvc1ONOw6KChI4+SEqO/vl6~-1~-1~-1; SIDN=101122945591; SSIDN=105141435718; EXC=105141435718:73; TAsessionID=8484bb5b-9d76-46b7-9764-8750e4b6c0d1|NEW; gch=eyJJZCI6MTcyMDA3OTM4MDYsIlJlc3RyaWN0ZWQiOmZhbHNlLCJTb3VyY2UiOjEsIlN0YXR1cyI6MSwiQ29va2llTmFtZSI6ImdjaCIsIkV4cGlyZXMiOiIyMDI1LTA2LTE5VDEzOjQ3OjU2LjMzNDUzNzZaIiwiR2VvbG9jYXRlSW5TZWNvbmRzIjozNjAwLCJMb2NhdGlvbiI6IlVTLVdJIiwiSGFzaCI6IlNiMkF3SklVWG9URjlUWFRybEpUZHJrRy96ZFZxNnBYVGFlWDJReFR2MlU9IiwiU2l0ZUV4cGVyaWVuY2UiOiJVUy1ESyIsIkJ1ZmZlclRpbWVTZWNvbmRzIjozMCwiSXBBZGRyZXNzIjoiMzUuMTQ4LjY4LjE3NCIsIkNvbm5lY3Rpb25UeXBlIjoic3RhdGljIiwiUHJpbWFyeUNpdHkiOiJNaWRkbGV0b24iLCJUYWdnZWRCb3VuZGFyaWVzIjoiIiwiRW5jcnlwdGVkQ29vcmRpbmF0ZXMiOnsiRW5jcnlwdGVkQ29vcmRpbmF0ZXMiOiJKT0hxam5RUlBmZ2p4dy9LdXF2emdXeEhQV0laZWUveld3RjB3cHBpMVdzNUtWN1RBVlJNSWtsaGFYSGNqeUN6IiwiSXYiOiI3TWUxRUV3dkpIaEpDTE44aHRSN29BPT0ifX0=; jwe=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsInZlcnNpb24iOiIxIn0.eyJ1bmlxdWVfbmFtZSI6IlNhbUJhbUNIIiwib3BpZCI6IjEiLCJ1ayI6IjlmYjVhYWZkLWJkNTQtNGY3OS1iNzJhLWI0NDc4YWVhZDMzYSIsInJvbGUiOiJOb3JtYWwiLCJlbWFpbCI6IlNhbUJhbUNoQEdtYWlsLmNvbSIsInJhZmlkIjoiIiwidmlkIjoiNDMyMjQwMjEwNiIsImxpZCI6IjEiLCJqaWQiOiI0OTA5ODY5NTExMiIsImprZXkiOiJjdXJyZW50IiwiZnRkYmUiOiJGYWxzZSIsImdlbyI6IlVTLVdJIiwiZnZzIjoiMCIsImNkYXRlIjoiMjAyNS0wNi0xNCAyMjo0NjoyMFoiLCJjdXIiOiJVU0QiLCJ1c3MiOiIxIiwiZ2Vvc2Itc2UiOiIiLCJnZW9zYi1pZCI6IjAiLCJnZW8tc2UiOiJVUy1ESyIsImdlby1sb2MiOiJVUy1XSSIsIml2IjoiV1VMREZDSytkMWdJMnQ2M3hrSlZIdz09IiwiZXVpZCI6ImtUVVFDeTF4dWZJUm56SjlCUEs5d2c9PSIsInV2cyI6WyIwLTEiLCIwLTciLCIwLTgiLCIwLTEwIiwiMC0xNCJdLCJES1AtRGVueUJlZ2lubmVyQ29udGVzdHMiOiJ0cnVlIiwiREtQLURlbnlDYXN1YWxDb250ZXN0cyI6InRydWUiLCJES1AtU2hvd1Nwb3J0c2Jvb2siOiJ0cnVlIiwiREtQLVZpZXdPZGRzIjoidHJ1ZSIsIkRLUC1EZW1vQ2FzaW5vIjoidHJ1ZSIsIkRLUC1WaWV3RGZzIjoidHJ1ZSIsIkRLUC1BbGxvd01hcmtldHBsYWNlRGVwb3NpdCI6InRydWUiLCJES1AtVmlld01hcmtldHBsYWNlIjoidHJ1ZSIsIkRLUC1BbGxvd01hcmtldHBsYWNlVmVyaWZpZWRUcmFuc2FjdGlvbiI6InRydWUiLCJES1AtU2hvd1Nwb3J0c2Jvb2tDcm9zc1NlbGwiOiJ0cnVlIiwiREtQLVBsYXlGcmVlUG9vbHMiOiJ0cnVlIiwiREtQLUFsbG93Q3JlZGl0Q2FyZEZ1bmRzIjoidHJ1ZSIsIkRLUC1BbGxvd1JlaWdubWFrZXJzIjoidHJ1ZSIsIkRLUC1BbGxvd1dpdGhkcmF3YWxQcm9jZXNzaW5nIjoidHJ1ZSIsIkRLUC1Nb2RpZnlBY2NvdW50IjoidHJ1ZSIsIkRLUC1WaWV3UGljazYiOiJ0cnVlIiwiREtQLUFsbG93UGFpZFJlaWdubWFrZXJzIjoidHJ1ZSIsIkRLUC1WaWV3RGtIb3JzZSI6InRydWUiLCJES1AtQWxsb3dPSURDIjoidHJ1ZSIsIkRLUC1BbGxvd09BdXRoIjoidHJ1ZSIsIkRLUC1TaG93REZTQ3Jvc3NTZWxsIjoidHJ1ZSIsIkRLUC1TaG93UGlja1NpeENyb3NzU2VsbCI6InRydWUiLCJES1AtT2ZmbGluZURlcG9zaXQiOiJ0cnVlIiwiREtQLUFsbG93TUZBUmVzdHJpY3RlZEFjdGlvbnMiOiJ0cnVlIiwic3hwIjoiVVMtREsiLCJhdXRoIjoiYmIwNmU0YTgtZThhNi00NjVhLTk3OWMtODNlNWVmMDhkZTUwIiwidmlwIjoiMyIsImx0IjoiZHJhZnRraW5ncyIsImxtIjoiY3JlZGVudGlhbHMiLCJzYnQiOiIzMDQwODY4MSIsImxsIjoiMjAyNS0wNi0xNCAwMTozNDoxOVoiLCJicmFuZCI6IkRLIiwibG95YWx0eSI6IkJyb256ZSIsIm5iZiI6MTc1MDMzODEzMywiZXhwIjoxNzUwMzM4NDMzLCJpc3MiOiJ1cm46ZGsvY2VyYmVydXMiLCJhdWQiOiJ1cm46ZGsifQ.E4usJHJh4jiK0uwQ02aKJtlCaxbgo5rXj8BcqU6guCo; iv=gIpDbBtvSH1NX624m+DIddytVRL8DLy6uTp7vs7AijU=; hgg=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ2aWQiOiI0MzIyNDAyMTA2IiwiZGtoLTEyNiI6IjgzM1RkX1lKIiwiZGtlLTEyNiI6IjAiLCJka2UtMjA0IjoiNzEwIiwiZGtlLTI4OCI6IjExMjgiLCJka2UtMzE4IjoiMTI2MCIsImRrZS0zNDUiOiIxMzUzIiwiZGtlLTM0NiI6IjEzNTYiLCJka2UtNDI5IjoiMTcwNSIsImRrZS03MDAiOiIyOTkyIiwiZGtlLTczOSI6IjMxNDAiLCJka2UtNzU3IjoiMzIxMiIsImRrZS04MDYiOiIzNDI2IiwiZGtlLTgwNyI6IjM0MzciLCJka2UtODI0IjoiMzUxMSIsImRrZS04MjUiOiIzNTE0IiwiZGtlLTgzNiI6IjM1NzAiLCJka2gtODk1IjoiOGVTdlpEbzAiLCJka2UtODk1IjoiMCIsImRrZS05MDMiOiIzODQ4IiwiZGtlLTkxNyI6IjM5MTMiLCJka2UtOTQ3IjoiNDA0MiIsImRrZS05NzYiOiI0MTcxIiwiZGtoLTE2NDEiOiJSMGtfbG1rRyIsImRrZS0xNjQxIjoiMCIsImRrZS0xNjUzIjoiNzEzMSIsImRrZS0xNjg2IjoiNzI3MSIsImRrZS0xNjg5IjoiNzI4NyIsImRrZS0xNzU0IjoiNzYwNSIsImRrZS0xNzYwIjoiNzY0OSIsImRrZS0xNzc0IjoiNzcwOSIsImRrZS0xNzgwIjoiNzczMSIsImRrZS0xNzk0IjoiNzgwMSIsImRraC0xODA1IjoiT0drYmxrSHgiLCJka2UtMTgwNSI6IjAiLCJka2UtMTgyOCI6Ijc5NTYiLCJka2UtMTg2MSI6IjgxNTciLCJka2UtMTg2OCI6IjgxODgiLCJka2UtMTg5OCI6IjgzMTQiLCJka2gtMTk1MiI6ImFVZ2VEWGJRIiwiZGtlLTE5NTIiOiIwIiwiZGtzLTIwNzkiOiI5MTMyIiwiZGtlLTIwOTciOiI5MjA1IiwiZGtlLTIxMDAiOiI5MjIzIiwiZGtlLTIxMDMiOiI5MjQxIiwiZGtlLTIxMzUiOiI5MzkzIiwiZGtlLTIxNDEiOiI5NDM1IiwiZGtoLTIxNTAiOiJOa2JhU0Y4ZiIsImRrZS0yMTUwIjoiMCIsImRrZS0yMTYxIjoiOTUxNSIsImRrcy0yMDE5IjoiODg0NyIsImRrcy0yMDIwIjoiODg1MCIsImRrcy0yMTk0IjoiOTY2MyIsImRrZS0yMTk1IjoiOTY2NSIsImRrZS0yMjIwIjoiOTc2OSIsImRrZS0yMjIyIjoiOTc3NCIsImRrZS0yMjI0IjoiOTc4NCIsImRrZS0yMjI2IjoiOTc5MCIsImRrZS0yMjM3IjoiOTgzNCIsImRrZS0yMjM4IjoiOTgzNyIsImRrZS0yMjQwIjoiOTg1NyIsImRrZS0yMjQ2IjoiOTg4NyIsImRrZS0yMjg4IjoiMTAwOTIiLCJka2UtMjI4OSI6IjEwMDk3IiwiZGtlLTIyOTEiOiIxMDEwMiIsImRrZS0yMzA3IjoiMTAyMjUiLCJka2UtMjMxOCI6IjEwMjgwIiwiZGtlLTIzMjQiOiIxMDMyMyIsImRrZS0yMzI4IjoiMTAzMzgiLCJka2UtMjMzMyI6IjEwMzcxIiwiZGtlLTIzMzkiOiIxMDM5NiIsImRrZS0yMzQwIjoiMTA0MDQiLCJka2UtMjM0MSI6IjEwNDA2IiwiZGtlLTIzNDIiOiIxMDQxMSIsImRrZS0yMzQ1IjoiMTA0MjkiLCJka2UtMjM0NiI6IjEwNDMyIiwiZGtlLTIzNDkiOiIxMDQ1OCIsImRrZS0yMzUyIjoiMTA0ODMiLCJka2UtMjM1NSI6IjEwNTAwIiwiZGtlLTIzNTgiOiIxMDUxNSIsImRrZS0yMzU5IjoiMTA1MTgiLCJka2UtMjM2NCI6IjEwNTQyIiwiZGtlLTIzNjUiOiIxMDU0NSIsImRrZS0yMzY2IjoiMTA1NDkiLCJka2gtMjM3NSI6IjRZY0xwb3gzIiwiZGtlLTIzNzUiOiIwIiwiZGtoLTIzNzYiOiJtcnpmOGNOZiIsImRrZS0yMzc2IjoiMCIsImRrZS0yMzgxIjoiMTA2MDYiLCJka2UtMjM4MyI6IjEwNjI3IiwiZGtlLTIzODQiOiIxMDYzNCIsImRraC0yMzg2IjoibWdqbUc1cEsiLCJka2UtMjM4NiI6IjAiLCJka2gtMjM4NyI6IlFQOWZKWWdFIiwiZGtlLTIzODciOiIwIiwiZGtlLTIzODgiOiIxMDY1MSIsImRraC0yMzkwIjoibkE1OFFuNGoiLCJka2UtMjM5MCI6IjAiLCJka2UtMjM5MSI6IjEwNjY0IiwiZGtoLTIzOTIiOiJUT1NuRlR6ZyIsImRrZS0yMzkyIjoiMCIsImRrZS0yMzkzIjoiMTA2ODUiLCJka2gtMjM5NSI6IkZHQjYtQVhuIiwiZGtlLTIzOTUiOiIwIiwibmJmIjoxNzUwMzM4MTMzLCJleHAiOjE3NTAzMzg0MzMsImlhdCI6MTc1MDMzODEzMywiaXNzIjoiZGsifQ.UViBALWfReuWkTJ3p_UTVCa0XopR4Xr_3tPhQMThMZ8; bm_sz=B5A75966907C60287E125B434EB00487~YAAQNnsyF0WGkWCXAQAAlchIiBysBhJmVIygM0TEUtqum62fKsstkOoz/mtOVGFOkMTL0e7cos8KBjpl/K2qyFOMkWlNyIBeoDqBzb2g79nWPC9AbhV20R2juTgSnAtn1kjx2Yy00YQIj1phg28suCNzHQyiRYWFDnUzSBx4FJWlDxJjmZwcaJjvgMKt8B16YIOvhfP/vvmREUvDTjwUa4Rb/8gk7BXPqALdTS5kpquSp+EoqouOvCDp4LZbd0kJlo+rqILcbnECigF4o6Hy3eslXdJMsPKRgSpEU53PlgCq/qfNLsIgG3wyW1HvmrP3H4Ijqy3ByBSxpS6GwEw4L2Slc5oSEuXlJy9RUw/GRXEUKwJot4f9biq2l5xrOHbxRjDUG8IRiiFAQSoXTCOxB4oPTbAzVXkOiXW9T1WLt2CLVnaEapdh+/DXVJeWYUmb7/2EXA==~4339526~3356473; _rdt_uuid=1719604931045.235f79f2-d88a-4b6e-b8f0-2bd58f1d9438; _scid_r=LkxT-DzpjaCXmmcl5qi6ig3RH3H0IGHzoZDQeA; ab.storage.sessionId.28eefbae-1976-4817-bed3-fdc2de3d28dc=%7B%22g%22%3A%227a6e7f09-f166-96ac-be35-346a40a34e1f%22%2C%22e%22%3A1750339961890%2C%22c%22%3A1750337274051%2C%22l%22%3A1750338161890%7D; ttcsid=1750337274257::MXzu5ABCybQ4iskrYhOP.384.1750338161945; ss-id=q0c2CZi6770DGTgzFmBS; ss-pid=5X1wMGMTpzEFpAESTX6o; _uetsid=ef828010457b11f0a8e67963ddc72616; _uetvid=f7d6bae034ab11efb898ab9bfc238976; ttcsid_C9QNSUJC77U8C02RC95G=1750337272323::ksncuvYjVvru58YZLffT.380.1750338162568; _dc_gtm_UA-28146424-1=1; _gat_UA-28146424-14=1; _ga=GA1.1.1845256652.1719509867; _ga_QG8WHJSQMJ=GS2.1.s1750337272$o1902$g1$t1750338161$j59$l0$h0; STE="2025-06-19T13:32:43.2284332Z"; bm_sv=E7FCA71D6543B096A1FEAC7FEED4169C~YAAQNnsyF2WGkWCXAQAAMtJIiByTwKiaFLjwiQ5E0cdZDI7nAuAagM3sHCF307drOQ5YokshbPyTM4lDgbQzYtEE7fe0hUghxt99+w6NI43Zv7QT/RBSTemq42+ZiS6CG3F4TVHj8mfQiFkZ+8T5sYKPMmx0YJJnjzd7anKJzT6KadJf3YrSBm82r7RZBP4ZOYQVC6MXlpQK5zQrhho93gMhcroR3ARYfMsCHhl4R+pt4VjBrBpjJWSdzqyFfsQ4w2oHYUb7~1; _ga_MCSJH508Q0=GS2.2.s1750337275$o1479$g1$t1750338162$j60$l0$h0; _ga_M8T3LWXCC5=GS2.2.s1750337275$o1496$g1$t1750338163$j60$l0$h0'
    # 1️⃣ parse the draft group ID
    dgid_str = sys.argv[1]
    try:
        dgid = int(dgid_str)
    except ValueError:
        print(f"Invalid draft_group_id: {dgid_str}")
        sys.exit(1)

    # 2️⃣ for each remaining arg, treat as a contest ID
    for cid_str in sys.argv[2:]:
        try:
            cid = int(cid_str)
        except ValueError:
            print(f"Skipping invalid contest_id: {cid_str}")
            continue

        print(f"→ Rebuilding contest {cid} (draft_group_id={dgid}) …")
        rebuild_contest_data(dgid, cid, raw_cookie)
        time.sleep(1)


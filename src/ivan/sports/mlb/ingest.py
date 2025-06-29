#!/usr/bin/env python
import os
import csv
import json
import re
from datetime import datetime

from sqlalchemy import delete

from ivan.database.session import init_db, get_session, Base
from ivan.sports.mlb.models import (
    BatterProjection,
    PitcherProjection,
    HeroProjection
)
from ivan.core.utils import normalize_name

# adjust this relative to your scripts/ directory
CONFIG_PATH = os.path.abspath(os.path.join(__file__, '../../../../../config/mlb.json'))
TEAM_ABBR = {
    'WAS': 'WSH', 'SFG': 'SF', 'TBR': 'TB',
    'KCR': 'KC',  'SDP': 'SD',  'CHW': 'CWS'
}

def clean_opponent(raw: str) -> str:
    """
    Strip leading '@' or 'vs' from opponent field,
    then apply our TEAM_ABBR mapping.
    """
    cleaned = re.sub(r'^(?:@|vs\.?\s*)', '', raw or '').strip()
    return TEAM_ABBR.get(cleaned, cleaned)

def to_int(val, default=0):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default

def to_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def ingest_projections():
    # 1) Generate a batch timestamp for this load
    batch_ts = datetime.utcnow()

    # 2) Load config and DB session
    cfg    = json.load(open(CONFIG_PATH, encoding='utf-8'))
    engine = init_db('mlb', cfg.get('database_url'))
    sess   = get_session(engine)

    proj_dir  = cfg['paths']['projections_dir']
    hero_path = cfg['paths']['hero_projections']

    # ─── NEW: read the player_ids.csv to know which teams are on our slate ───
    pid_fp = cfg['paths']['ids']
    slate_teams = set()
    with open(pid_fp, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            team_abbrev = row.get('TeamAbbrev', '').strip()
            if team_abbrev:
                slate_teams.add(team_abbrev)
    # ──────────────────────────────────────────────────────────────────────────

    # ─── STEP 0: scan b.csv to find any double‐header pairings ───
    bpath = os.path.join(proj_dir, 'b.csv')
    dh_map: dict[tuple[str, str], set[str]] = {}

    # First pass: collect all rows with a '2H' of '1' or '2', grouped by pairing
    with open(bpath, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_2h = (row.get('2H') or '').strip()
            if raw_2h in ('1', '2'):
                # normalize the two teams
                team_code = TEAM_ABBR.get(row.get('TEAM'), row.get('TEAM'))
                opp_code  = clean_opponent(row.get('OPP_TM'))
                # use a sorted tuple so “STL‐KCR” and “KCR‐STL” collapse to the same key
                pairing = tuple(sorted([team_code, opp_code]))
                dh_map.setdefault(pairing, set()).add(raw_2h)

    # For each pairing where both halves ('1' and '2') are present, ask user which half
    selected_halves: dict[tuple[str, str], str] = {}
    for pairing, halves in dh_map.items():
        if halves == {'1', '2'}:
            # prompt including both team codes
            t1, t2 = pairing
            prompt = f"DOUBLE HEADER DETECTED for {t1} vs {t2}. Which game is on the slate? (1 or 2) "
            choice = None
            while choice not in ('1', '2'):
                choice = input(prompt).strip()
            selected_halves[pairing] = choice
            print(f"→ Filtering to game {choice} for {t1} vs {t2}.\n")
    # ──────────────────────────────────────────────────────────────────────────

    missing_ol_all: set[str] = set()

    # ─── INGEST BATTERS ───
    with open(bpath, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_2h = (row.get('2H') or '').strip()
            team_code = TEAM_ABBR.get(row.get('TEAM'), row.get('TEAM'))
            opp_code  = clean_opponent(row.get('OPP_TM'))
            pairing   = tuple(sorted([team_code, opp_code]))

            # Determine if this row is part of a double‐header pairing
            if raw_2h in ('1', '2'):
                # if the pairing was prompted, only keep this half
                if pairing in selected_halves and raw_2h != selected_halves[pairing]:
                    continue
            # If raw_2h is blank or not '1'/'2', we treat it as “always include”:
            # that covers single‐game players or teams not in a detected DH.

            # Normalize team code again (already done above)
            # If OL == FALSE, remember that team
            if row.get('OL', '').strip().upper() == 'FALSE':
                missing_ol_all.add(team_code)

            # skip non‐starters
            if row.get('IPL', '').strip().upper() == 'FALSE':
                continue

            bp = BatterProjection(
                player_id       = to_int(row.get('PLAYERID')),
                name            = normalize_name(row.get('NAME')),
                hand            = row.get('HAND'),
                team            = team_code,
                opp_tm          = opp_code,
                salary          = to_int(row.get('SALARY')),
                pos             = row.get('POS'),
                lp              = to_int(row.get('LP')),
                ol              = row.get('OL','').upper() == 'TRUE',
                od              = row.get('OD','').upper() == 'TRUE',
                ipl             = row.get('IPL','').upper() == 'TRUE',
                error           = row.get('ERROR'),
                doubles_2h      = row.get('2H') or None,
                pitcher         = row.get('PITCHER'),
                catcher         = row.get('CATCHER'),
                umpire          = row.get('UMPIRE'),
                park            = row.get('PARK'),
                roof            = row.get('ROOF'),
                gvf             = row.get('GVF'),
                platoon         = row.get('PLATOON','').upper() == 'TRUE',
                hfa             = row.get('HFA','').upper() == 'TRUE',
                ph_pct          = to_float(row.get('PH%')),
                cold            = row.get('COLD','').upper() == 'TRUE',
                tilt_bias       = row.get('TILT_BIAS'),
                fpts            = to_float(row.get('FPTS')),
                fpts_per_dollar = to_float(row.get('FPTS/$')),
                r               = to_float(row.get('R')) + 0.01,
                sb              = to_float(row.get('SB')),
                cs              = to_float(row.get('CS')),
                sh              = to_float(row.get('SH')),
                sf              = to_float(row.get('SF')),
                gidp            = to_float(row.get('GIDP')),
                hr_dependence   = to_float(row.get('HR_DEPENDENCE')),
                ph_fpts_pct     = to_float(row.get('0%_PH%_FPTS')),
                slate           = row.get('SLATE'),
                floor           = to_float(row.get('FLOOR')),
                ceiling         = to_float(row.get('CEILING')),
                pown            = to_float(row.get('POWN')),
                pa              = to_float(row.get('PA')),
                pavssp          = to_float(row.get('PAVSSP')),
                pavssp_pct      = to_float(row.get('PAVSSP%')),
                ba              = to_float(row.get('BA')),
                obp             = to_float(row.get('_OBP')),
                slg             = to_float(row.get('SLG')),
                iso             = to_float(row.get('ISO')),
                woba            = to_float(row.get('WOBA')),
                cbwoba          = to_float(row.get('CNWOBA')),
                xwoba_cs        = to_float(row.get('XWOBA_CS')),
                xwoba_ls        = to_float(row.get('XWOBA_LS')),
                k               = to_float(row.get('K')),
                bb              = to_float(row.get('BB')),
                ibb             = to_float(row.get('IBB')),
                hbp             = to_float(row.get('HBP')),
                h               = to_float(row.get('H')),
                single          = to_float(row.get('1B')),
                double          = to_float(row.get('2B')),
                triple          = to_float(row.get('3B')),
                hr              = to_float(row.get('HR')),
                tb              = to_float(row.get('TB')),
                rbi             = to_float(row.get('RBI')),
                partnerid       = to_int(row.get('PARTNERID')),
                created_at      = batch_ts
            )
            sess.add(bp)

    # ─── INGEST PITCHERS (only one per team, but still filter by 2H) ───
    ppath = os.path.join(proj_dir, 'p.csv')
    best_pp: dict[str, dict] = {}   # team → raw row dict

    with open(ppath, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # skip non‐starters
            if row.get('IPL','').strip().upper() == 'FALSE':
                continue

            team_code = TEAM_ABBR.get(row.get('TEAM'), row.get('TEAM'))
            opp_code  = clean_opponent(row.get('OPP'))
            pairing   = tuple(sorted([team_code, opp_code]))
            raw_2h    = (row.get('2H') or '').strip()

            # If this pitcher row has raw_2h '1' or '2', enforce the same filter:
            if raw_2h in ('1', '2'):
                if pairing in selected_halves and raw_2h != selected_halves[pairing]:
                    continue
            # raw_2h blank → always include

            fpts = to_float(row.get('FPTS'))
            if team_code not in best_pp or fpts > to_float(best_pp[team_code]['FPTS']):
                best_pp[team_code] = row

    # write exactly one pitcher per team
    for row in best_pp.values():
        pp = PitcherProjection(
            player_id       = to_int(row.get('PLAYERID')),
            player          = normalize_name(row.get('PLAYER')),
            salary          = to_int(row.get('SALARY')),
            pos             = row.get('POS'),
            team            = TEAM_ABBR.get(row.get('TEAM'), row.get('TEAM')),
            opp             = clean_opponent(row.get('OPP')),
            schedule_id     = to_int(row.get('SCHEDULE_ID')),
            slate           = row.get('SLATE'),
            tm              = row.get('TM'),
            opp_tm          = clean_opponent(row.get('OPP_TM')),
            hand            = row.get('HAND'),
            ol              = row.get('OL','').upper() == 'TRUE',
            od              = row.get('OD','').upper() == 'TRUE',
            pcc             = row.get('PCC','').upper() == 'TRUE',
            error           = row.get('ERROR'),
            doubles_2h      = row.get('2H') or None,
            bpc             = to_float(row.get('BPC')),
            ppc             = to_float(row.get('PPC')),
            mpc             = to_float(row.get('MPC')),
            opener          = row.get('OPENER','').upper() == 'TRUE',
            catcher         = row.get('CATCHER'),
            umpire          = row.get('UMPIRE'),
            park            = row.get('PARK'),
            roof            = row.get('ROOF'),
            platoon         = row.get('PLATOON'),
            split           = row.get('SPLIT'),
            gvf             = row.get('GVF'),
            hfa             = row.get('HFA','').upper() == 'TRUE',
            dh              = row.get('DH','').upper() == 'TRUE',
            familiarity     = row.get('FAMILIARITY'),
            tilt_bias       = row.get('TILT_BIAS'),
            spin_ls         = to_int(row.get('SPIN_LS')),
            spin_cs         = to_int(row.get('SPIN_CS')),
            spin_rec        = to_int(row.get('SPIN_REC')),
            spin_change     = to_int(row.get('SPIN_CHANGE')),
            fpts            = to_float(row.get('FPTS')),
            fpts_per_dollar = to_float(row.get('FPTS/$')),
            ip              = to_float(row.get('IP')),
            outs            = to_float(row.get('OUTS')),
            era             = to_float(row.get('ERA')),
            cnera           = to_float(row.get('CNERA')),
            w               = to_float(row.get('W')),
            l               = to_float(row.get('L')),
            qs              = row.get('QS','').upper() == 'TRUE',
            cg              = row.get('CG','').upper() == 'TRUE',
            cgsh            = row.get('CGSH','').upper() == 'TRUE',
            tbf             = to_float(row.get('TBF')),
            k               = to_float(row.get('K')),
            bb              = to_float(row.get('BB')),
            ibb             = to_float(row.get('IBB')),
            hbp             = to_float(row.get('HBP')),
            h               = to_float(row.get('H')),
            hr              = to_float(row.get('HR')),
            tb              = to_float(row.get('TB')),
            sh              = to_float(row.get('SH')),
            sf              = to_float(row.get('SF')),
            gidp            = to_float(row.get('GIDP')),
            sb              = to_float(row.get('SB')),
            cs              = to_float(row.get('CS')),
            er              = to_float(row.get('ER')),
            floor           = to_float(row.get('FLOOR')),
            ceiling         = to_float(row.get('CEILING')),
            pown            = to_float(row.get('POWN')),
            rgid            = to_int(row.get('RGID')),
            partnerid       = to_int(row.get('PARTNERID')),
            ownership       = to_float(row.get('OWNERSHIP')),
            created_at      = batch_ts
        )
        sess.add(pp)

    # ─── INGEST HERO ───
    with open(hero_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            hp = HeroProjection(
                source_id    = to_int(row.get('ID')),
                name         = normalize_name(row.get('Name')),
                pos          = row.get('Pos'),
                team         = row.get('Team'),
                opp          = row.get('Opp'),
                salary       = to_int(row.get('Salary')),
                stdv         = to_float(row.get('STDV')),
                value        = to_float(row.get('Value')),
                hand         = row.get('Hand'),
                order_rank   = to_int(row.get('Order')),
                runs         = to_float(row.get('Runs')),
                era          = to_float(row.get('ERA')),
                k9           = to_float(row.get('K/9')),
                whip         = to_float(row.get('WHIP')),
                woba         = to_float(row.get('wOBA')),
                iso          = to_float(row.get('ISO')),
                floor        = to_float(row.get('Floor')),
                ceiling      = to_float(row.get('Ceiling')),
                rnkgpp       = to_int(row.get('RnkGPP')),
                rnkcash      = to_int(row.get('RnkCash')),
                top1         = to_float(row.get('Top 1')),
                top5         = to_float(row.get('Top 5')),
                top10        = to_float(row.get('Top 10')),
                two_x        = to_float(row.get('2X')),
                two_5_x      = to_float(row.get('2.5X')),
                three_x      = to_float(row.get('3X')),
                three_5_x    = to_float(row.get('3.5X')),
                four_x       = to_float(row.get('4X')),
                four_5_x     = to_float(row.get('4.5X')),
                five_x       = to_float(row.get('5X')),
                five_5_x     = to_float(row.get('5.5X')),
                optimal      = to_float(row.get('Optimal')),
                lev          = to_float(row.get('Lev')),
                hero_own     = to_float(row.get('Hero Own')),
                hero_proj    = to_float(row.get('Hero Proj.')),
                user_own     = to_float(row.get('User Own')),
                user_proj    = to_float(row.get('User Proj.')),
                agg_proj     = to_float(row.get('Agg Proj.')),
                rdm_pct      = to_float(row.get('Rdm %')),
                min_exp      = to_float(row.get('Min. Exp.')),
                max_exp      = to_float(row.get('Max. Exp.')),
                build_exp    = to_float(row.get('Build Exp.')),
                created_at   = batch_ts
            )
            sess.add(hp)

    # ─── COMMIT EVERYTHING ───
    sess.commit()
    print("✅ Flattened projections ingested into separate tables.")

    # ─── NOW FILTER missing_ol_all BY slate_teams ───
    missing_ol_slate = missing_ol_all.intersection(slate_teams)
    if missing_ol_slate:
        print(
            "⚠️  The following slate‐teams have OL=False (no official lineup loaded): "
            f"{', '.join(sorted(missing_ol_slate))}"
        )
    else:
        print("✅ Every slate‐team had OL=True.")

    return batch_ts


if __name__ == '__main__':
    ingest_projections()

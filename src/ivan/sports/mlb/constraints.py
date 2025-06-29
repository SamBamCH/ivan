# src/ivan/sports/mlb/constraints.py

import random
from collections import defaultdict
from pulp import lpSum, LpVariable, LpBinary
from ivan.core.utils import get_config_value

class Constraints:
    def __init__(self, site, problem, players, lp_variables, config, verbose=False):
        self.site      = site
        self.problem   = problem
        self.players   = players
        self.vars      = lp_variables      # now keyed by (player_obj, position)
        self.config    = config
        self.verbose   = verbose

    def _add_salary_constraints(self):
        max_sal = get_config_value(self.config, "lineup.max_salary", 50_000)
        min_sal = get_config_value(self.config, "lineup.min_salary",    0)
        expr = lpSum(p.salary * self.vars[(p, pos)]
                     for p in self.players for pos in p.position)
        self.problem += expr <= max_sal, "Max_Salary"
        self.problem += expr >= min_sal, "Min_Salary"

    def _add_position_constraints(self):
        if self.site != "dk":
            raise ValueError("Position constraints only set up for DK")
        # your per‐slot counts
        limits = {"P":2, "C":1, "1B":1, "2B":1, "3B":1, "SS":1, "OF":3}

        # enforce each slot exactly
        for slot, lim in limits.items():
            elig = [ self.vars[(p, slot)]
                     for p in self.players if slot in p.position ]
            self.problem += lpSum(elig) == lim, f"Pos_{slot}"

        # **new**: enforce total roster size = sum(limits)
        total_slots = sum(limits.values())
        all_vars    = [ self.vars[(p, pos)]
                        for p in self.players for pos in p.position ]
        self.problem += lpSum(all_vars) == total_slots, "TotalRosterSize"

    def _add_single_use_constraints(self):
        for p in self.players:
            vs = [ self.vars[(p, pos)] for pos in p.position ]
            self.problem += lpSum(vs) <= 1, f"SingleUse_{p.id}"

    def _add_team_batter_constraints(self):
        max_b = 5
        by_team = defaultdict(list)
        for p in self.players:
            if not any(pos.startswith("P") for pos in p.position):
                for pos in p.position:
                    by_team[p.team].append(self.vars[(p, pos)])
        for team, vs in by_team.items():
            self.problem += lpSum(vs) <= max_b, f"MaxBat_{team}"

    def _add_opposing_batter_constraints(self):
        max_opp = get_config_value(self.config,
                                   "lineup.max_opposing_batters", 0)
        M       = 5

        pitchers = defaultdict(list)
        for p in self.players:
            for pos in p.position:
                if pos.startswith("P"):
                    pitchers[p.opponent].append(self.vars[(p, pos)])

        batters = defaultdict(list)
        for p in self.players:
            for pos in p.position:
                if not pos.startswith("P"):
                    batters[p.team].append(self.vars[(p, pos)])

        for opp, p_vars in pitchers.items():
            if opp not in batters:
                continue
            y = LpVariable(f"PitchVs_{opp}", cat=LpBinary)
            for x_p in p_vars:
                self.problem += y >= x_p, f"LinkPitch_{x_p.name}_vs_{opp}"
            self.problem += (
                lpSum(batters[opp]) <= max_opp + M * (1 - y),
                f"MaxOppBat_{opp}"
            )

    def add_static_constraints(self):
        self._add_salary_constraints()
        self._add_position_constraints()
        self._add_single_use_constraints()
        self._add_team_batter_constraints()
        self._add_opposing_batter_constraints()

    def add_team_stack_constraints(self):
        stack_dist = get_config_value(self.config,
                                      "lineup.stack_probabilities", {})
        if not stack_dist:
            return

        # normalize and pick one stack shape
        types, weights = zip(*stack_dist.items())
        total = sum(weights)
        weights = [w/total for w in weights]
        sel = random.choices(types, weights=weights, k=1)[0]
        prim_n, sec_s = sel.split("-")
        prim_n = int(prim_n)
        sec_n  = None if sec_s.lower()=="x" else int(sec_s)

        # map team → list of batter‐vars
        by_team = defaultdict(list)
        for p in self.players:
            if not any(pos.startswith("P") for pos in p.position):
                for pos in p.position:
                    by_team[p.team].append(self.vars[(p, pos)])
        teams = list(by_team.keys())

        # primary
        prim = LpVariable.dicts("PrimaryStack", teams, cat=LpBinary)
        self.problem += lpSum(prim[t] for t in teams) == 1, "OnePrimaryStack"
        M = 5
        for t, vars_in_team in by_team.items():
            self.problem += (
                lpSum(vars_in_team) - prim_n <= M * (1 - prim[t]),
                f"PUp_{t}"
            )
            self.problem += (
                lpSum(vars_in_team) - prim_n >= -M * (1 - prim[t]),
                f"PLo_{t}"
            )

        # secondary (if any)
        if sec_n is not None:
            sec = LpVariable.dicts("SecondaryStack", teams, cat=LpBinary)
            self.problem += lpSum(sec[t] for t in teams) == 1, "OneSecondaryStack"
            for t, vars_in_team in by_team.items():
                self.problem += (prim[t] + sec[t] <= 1, f"DistinctStack_{t}")
                self.problem += (
                    lpSum(vars_in_team) - sec_n <= M * (1 - sec[t]),
                    f"SUp_{t}"
                )
                self.problem += (
                    lpSum(vars_in_team) - sec_n >= -M * (1 - sec[t]),
                    f"SLo_{t}"
                )

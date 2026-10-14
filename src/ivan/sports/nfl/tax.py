# src/ivan/sports/nfl/tax.py
"""
US federal gambling tax on a year of DFS results, as a pure function.

    compute_tax(W, L, cfg) -> TaxResult

W is gross winnings (sum over entries of payout - fee where positive), L is
gross losses (sum of fee - payout where positive). Everything else is in
TaxConfig; defaults are the 2026 rules (One Big Beautiful Bill Act: only 90%
of losses deductible) with the cap configurable because a retroactive repeal
is pending. Not tax advice; verify the figures in config/tax.json with a CPA.

Two reporting modes, side by side wherever the backtest reports:
  gross_165d  winnings are income, losses an itemized deduction (IRC 165(d))
  net_dfs     taxable = max(P, 0), as DraftKings' 1099-MISC reports DFS
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ivan.core.utils import PROJECT_ROOT

TAX_CONFIG_PATH = PROJECT_ROOT / "config" / "tax.json"

GROSS = "gross_165d"
NET = "net_dfs"
REPORTING_MODES = (GROSS, NET)
ITEMIZE_CHOICES = ("auto", "always", "never")

# Fallback standard deductions if config/tax.json is missing. Verify.
STANDARD_DEDUCTION_2026 = {"single": 16100.0, "married_joint": 32200.0, "married_separate": 16100.0,
                           "head_of_household": 24150.0}


@dataclass(frozen=True)
class TaxConfig:
    tax_year: int = 2026
    loss_cap_pct: float = 0.90                 # share of losses deductible; 1.0 = pre-2026 / repeal
    filing_status: str = "single"
    standard_deduction: Optional[float] = None  # None: looked up by filing status
    other_itemized: float = 0.0                # SALT, mortgage interest, charity ...
    other_taxable_income: float = 0.0          # wages etc. before deductions (bracket mode)
    marginal_rate: float = 0.22                # flat mode
    brackets: Optional[Tuple[Tuple[float, float], ...]] = None   # ((lower_bound, rate), ...) -> bracket mode
    state_rate: float = 0.0
    state_allows_loss_deduction: bool = True
    reporting_mode: str = GROSS
    itemize: str = "auto"                      # auto | always | never
    tax_reserve_pct: Optional[float] = None    # share of each win swept into a tax reserve (ledger only)
    label: str = ""

    def __post_init__(self):
        if self.reporting_mode not in REPORTING_MODES:
            raise ValueError(f"reporting_mode must be one of {REPORTING_MODES}, got {self.reporting_mode!r}")
        if self.itemize not in ITEMIZE_CHOICES:
            raise ValueError(f"itemize must be one of {ITEMIZE_CHOICES}, got {self.itemize!r}")
        if not 0.0 <= self.loss_cap_pct <= 1.0:
            raise ValueError("loss_cap_pct must be between 0 and 1")

    @property
    def mode(self) -> str:
        return "bracket" if self.brackets else "flat"

    @property
    def std_deduction(self) -> float:
        if self.standard_deduction is not None:
            return float(self.standard_deduction)
        return STANDARD_DEDUCTION_2026.get(self.filing_status, STANDARD_DEDUCTION_2026["single"])

    def with_(self, **changes) -> "TaxConfig":
        return replace(self, **changes)

    def describe(self) -> str:
        parts = ["net DFS" if self.reporting_mode == NET else f"gross, {self.loss_cap_pct:.0%} loss cap"]
        if self.reporting_mode == GROSS:
            parts.append({"auto": "itemize if it beats the standard deduction", "always": "itemizing",
                          "never": "standard deduction"}[self.itemize])
        parts.append(f"{self.mode} rate" + (f" {self.marginal_rate:.0%}" if self.mode == "flat" else ""))
        if self.state_rate:
            parts.append(f"state {self.state_rate:.1%}" + ("" if self.state_allows_loss_deduction else " (no loss deduction)"))
        return "; ".join(parts)


@dataclass
class TaxResult:
    W: float
    L: float
    P: float
    deductible: float
    itemized_total: float
    itemize: bool
    taxable_gambling: float
    incremental_taxable: float    # change in taxable income from the gambling activity
    phantom: float
    federal_tax: float
    state_tax: float
    est_tax: float
    after_tax_profit: float
    after_tax_roi: Optional[float]
    effective_rate: Optional[float]   # est_tax / taxable_gambling
    reporting_mode: str
    mode: str

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# --------------------------------------------------------------------------- brackets
def bracket_tax(taxable: float, brackets: Sequence[Tuple[float, float]]) -> float:
    """Tax on `taxable` under a table of (lower_bound, rate) rows sorted by lower bound."""
    taxable = max(0.0, float(taxable))
    rows = sorted((float(lo), float(r)) for lo, r in brackets)
    tax = 0.0
    for i, (lo, rate) in enumerate(rows):
        hi = rows[i + 1][0] if i + 1 < len(rows) else math.inf
        if taxable > lo:
            tax += (min(taxable, hi) - lo) * rate
    return tax


def marginal_rate_at(taxable: float, brackets: Sequence[Tuple[float, float]]) -> float:
    rows = sorted((float(lo), float(r)) for lo, r in brackets)
    rate = rows[0][1]
    for lo, r in rows:
        if taxable >= lo:
            rate = r
    return rate


# --------------------------------------------------------------------------- the engine
def compute_tax(W: float, L: float, cfg: TaxConfig, total_fees: Optional[float] = None) -> TaxResult:
    """
    Federal (and optional state) tax attributable to a year with gross
    winnings W and gross losses L. The federal number is incremental: tax
    with the gambling activity minus tax without it, holding other income and
    other deductions fixed, so a switch from the standard deduction to
    itemizing is priced in.
    """
    W, L = max(0.0, float(W)), max(0.0, float(L))
    P = W - L
    std = cfg.std_deduction

    if cfg.reporting_mode == NET:
        deductible = min(W, L)
        itemized_total = cfg.other_itemized
        itemize = False
        taxable_gambling = max(P, 0.0)
        incremental = taxable_gambling
    else:
        deductible = min(cfg.loss_cap_pct * L, W)
        itemized_total = deductible + cfg.other_itemized
        if cfg.itemize == "always":
            itemize, itemize_without = True, True
        elif cfg.itemize == "never":
            itemize, itemize_without = False, False
        else:
            itemize = itemized_total > std
            itemize_without = cfg.other_itemized > std
        taxable_gambling = W - deductible if itemize else W
        deduction_with = itemized_total if itemize else std
        deduction_without = cfg.other_itemized if itemize_without else std
        incremental = W - deduction_with + deduction_without

    phantom = taxable_gambling - max(P, 0.0)

    if cfg.mode == "bracket":
        other = cfg.other_taxable_income
        if cfg.reporting_mode == NET:
            with_ = other + taxable_gambling - max(std, cfg.other_itemized)
            without = other - max(std, cfg.other_itemized)
        else:
            with_ = other + W - deduction_with
            without = other - deduction_without
        federal = bracket_tax(with_, cfg.brackets) - bracket_tax(without, cfg.brackets)
    else:
        federal = cfg.marginal_rate * incremental

    state_taxable = taxable_gambling if cfg.state_allows_loss_deduction else W
    state = cfg.state_rate * state_taxable
    est = federal + state
    after = P - est
    return TaxResult(
        W=W, L=L, P=P, deductible=deductible, itemized_total=itemized_total, itemize=itemize,
        taxable_gambling=taxable_gambling, incremental_taxable=incremental, phantom=phantom,
        federal_tax=federal, state_tax=state, est_tax=est, after_tax_profit=after,
        after_tax_roi=(after / total_fees) if total_fees else None,
        effective_rate=(est / taxable_gambling) if taxable_gambling > 0 else None,
        reporting_mode=cfg.reporting_mode, mode=cfg.mode)


# --------------------------------------------------------------------------- closed forms
def after_tax_closed_form(P: float, L: float, t: float, cap: float = 0.90) -> float:
    """After-tax profit when itemizing and cap*L < W (flat rate t): P(1-t) - (1-cap) t L."""
    return P * (1 - t) - (1 - cap) * t * L


def breakeven_profit_ratio(t: float, cap: float = 0.90) -> float:
    """P/L needed for after-tax profit >= 0 when itemizing and cap*L < W: (1-cap) t / (1-t)."""
    return (1 - cap) * t / (1 - t)


# --------------------------------------------------------------------------- scenarios
def scenario_configs(base: TaxConfig) -> List[TaxConfig]:
    """The comparison table: cap 90% vs 100%, itemize vs standard, gross vs net, plus the user's own setting."""
    out = [base.with_(label="as configured")]
    for cap in (0.90, 1.00):
        for itemize in ("always", "never"):
            out.append(base.with_(reporting_mode=GROSS, loss_cap_pct=cap, itemize=itemize,
                                  label=f"gross cap {cap:.0%}, {'itemize' if itemize == 'always' else 'standard'}"))
    out.append(base.with_(reporting_mode=NET, label="net DFS (1099 net)"))
    return out


# --------------------------------------------------------------------------- config file
def load_tax_config(path: str | Path | None = None, **overrides) -> TaxConfig:
    """
    config/tax.json -> TaxConfig. The file holds defaults, a brackets table per
    filing status and standard deductions; `overrides` are applied last.
    bracket mode is used when overrides or the file set use_brackets true.
    """
    path = Path(path) if path else TAX_CONFIG_PATH
    raw = json.loads(path.read_text()) if path.exists() else {}
    defaults = dict(raw.get("defaults", {}))
    defaults.update({k: v for k, v in overrides.items() if v is not None})
    status = defaults.get("filing_status", "single")
    std_table = raw.get("standard_deduction", {})
    if defaults.get("standard_deduction") is None and status in std_table:
        defaults["standard_deduction"] = float(std_table[status])
    use_brackets = bool(defaults.pop("use_brackets", False))
    table = raw.get("brackets", {}).get(status)
    brackets = tuple((float(lo), float(r)) for lo, r in table) if (use_brackets and table) else None
    known = {f for f in TaxConfig.__dataclass_fields__}
    return TaxConfig(brackets=brackets, **{k: v for k, v in defaults.items() if k in known})

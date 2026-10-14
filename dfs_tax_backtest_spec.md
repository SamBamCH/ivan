# DFS Backtest: Tax-Aware Projection Module — Spec

Handoff spec for building a backtest module that projects **after-tax** DFS performance under 2026 US federal gambling tax rules. Hand this to a Claude Code agent along with the existing contest-simulation code.

> Not tax advice. Rule parameters are configurable because the law may change (see "Legal status"). Verify figures with a CPA before relying on output.

---

## 1. Goal

Given simulated or historical contest results over a tax year, compute:

- Gross winnings, gross losses, pre-tax net profit
- Deductible gambling losses under the selected rule set
- Taxable gambling income and the itemize-vs-standard decision
- Estimated federal tax attributable to gambling
- After-tax profit, **phantom income**, and after-tax ROI
- Running versions of all of the above, per slate/contest, so the user can see tax liability accrue as the bankroll moves

Primary use: decide how much volume to play, and in which contest formats, given that the tax cost scales with gross losses.

---

## 2. Definitions (per contest entry)

For each entry `i` with entry fee `fee_i` and payout `payout_i`:

- If `payout_i > fee_i`: **winnings** += `payout_i - fee_i`
- If `payout_i <= fee_i`: **losses** += `fee_i - payout_i`

Year totals:

- `W` = total gross winnings
- `L` = total gross losses
- `P = W - L` = pre-tax net profit (can be negative)

Also support a simplified **slate-level mode**, where the user supplies winnings and losses per slate directly (the convention used in the worked examples below: e.g. "$150 in, $100 out" = W += 100, L += 150).

---

## 3. Tax rules to implement

### 3a. Deductible gambling losses (itemizers)

```
cap_pct = 0.90            # 2026 rule (OBBBA). 1.00 = pre-2026 / repeal scenario
deductible = min(cap_pct * L, W)
```

Losses never create a deduction larger than winnings, and do not carry forward.

### 3b. Itemize vs. standard deduction

```
itemized_total = deductible + other_itemized    # SALT, mortgage interest, charity, etc.
use_itemize    = itemized_total > standard_deduction
```

- If itemizing: gambling losses reduce income by `deductible`.
- If taking the standard deduction: gambling losses are worth **$0**; all of `W` is taxable.

The module should report the decision and the **incremental** tax effect of gambling, i.e. compare total tax with vs. without the gambling activity, holding everything else constant.

### 3c. Taxable gambling income

```
taxable_gambling = W - deductible     # if itemizing
taxable_gambling = W                  # if standard deduction
```

### 3d. Phantom income

```
phantom = taxable_gambling - max(P, 0)
```

This is income taxed beyond actual profit.

### 3e. Tax

Support both:

1. **Flat marginal rate** (`marginal_rate`, e.g. 0.22), the simple mode.
2. **Bracket table** + `other_taxable_income`, to compute the true incremental tax from layering gambling income on top of wages. Bracket tables should be config, not hard-coded.

Optional: `state_rate` and a `state_allows_loss_deduction` flag, since some states do not allow gambling losses at all.

### 3f. Alternative reporting mode: DFS net (CPA-dependent)

DraftKings typically reports DFS on a 1099-MISC as net profit for the year. Some preparers report DFS on a net basis. Whether that holds under the 2026 rule is unsettled. Implement as a toggle:

```
reporting_mode = "gross_165d"   # default: rules 3a–3d
reporting_mode = "net_dfs"      # taxable = max(P, 0); no itemizing needed for losses
```

Always let the output show both modes side by side, so the user can see what's at stake.

---

## 4. Config / inputs

| Param | Default | Notes |
|---|---|---|
| `tax_year` | 2026 | |
| `loss_cap_pct` | 0.90 | 1.00 for repeal scenario |
| `filing_status` | single | |
| `standard_deduction` | 16100 | ~2026 single; ~32200 married filing jointly. **Verify.** |
| `other_itemized` | 0 | user's non-gambling itemized deductions |
| `other_taxable_income` | 0 | wages, etc. (bracket mode) |
| `marginal_rate` | 0.22 | flat mode |
| `brackets` | config file | bracket mode |
| `state_rate` | 0 | optional |
| `state_allows_loss_deduction` | true | optional |
| `reporting_mode` | gross_165d | or net_dfs |
| `tax_reserve_pct` | — | optional: simulate a % of each win swept out of the bankroll into a tax reserve |

---

## 5. Outputs

Per slate/contest (running) and year-end:

- `W`, `L`, `P`, `bankroll`
- `deductible`, `itemize?`, `taxable_gambling`, `phantom`
- `est_tax`, `after_tax_profit`, `after_tax_roi` (on total entry fees)
- `tax_reserve_needed` vs. `bankroll`: flag any point where projected tax owed > bankroll + reserve. **This is the key risk the user cares about: owing tax on money already lost.**

Scenario comparison table: cap 90% vs 100%, itemize vs standard, gross vs net mode.

Volume sweep: run the same edge (ROI distribution from the sim software) at multiple volume levels and plot after-tax expected profit and the probability of tax owed exceeding bankroll.

---

## 6. Useful closed forms (for validation)

After-tax profit when itemizing and `0.9L < W`:

```
after_tax = P * (1 - t) - 0.10 * t * L
```

Break-even condition (profit as a fraction of losses):

```
P / L > (0.10 * t) / (1 - t)
# t = 0.22 -> ~2.8%;  t = 0.24 -> ~3.2%
```

---

## 7. Test cases (must pass)

All flat-rate, itemizing unless noted, `loss_cap_pct = 0.90`.

1. **Single slate**: W=100, L=150 → 0.9L=135, deductible=min(135,100)=100, taxable=**0**.
2. **Single slate, standard deduction**: W=100, L=150 → taxable=**100**.
3. **Big win**: W=4000, L=100 → deductible=90, taxable=**3910**, phantom=10.
4. **Big win + n losing slates** (each slate: W+=100, L+=150):
   - W = 4000 + 100n, L = 100 + 150n
   - taxable = **3910 − 35n** while 0.9L < W (n < ~111.7)
   - Each $50 net-losing slate reduces taxable income by only $35 → $15 phantom per slate

   | n | W | L | deductible | taxable | P | phantom |
   |---|---|---|---|---|---|---|
   | 0 | 4000 | 100 | 90 | 3910 | 3900 | 10 |
   | 20 | 6000 | 3100 | 2790 | 3210 | 2900 | 310 |
   | 40 | 8000 | 6100 | 5490 | 2510 | 1900 | 610 |
   | 60 | 10000 | 9100 | 8190 | 1810 | 900 | 910 |
   | 78 | 11800 | 11800 | 10620 | 1180 | 0 | 1180 |
   | 80 | 12000 | 12100 | 10890 | 1110 | −100 | 1210 |

5. **Itemize vs standard**: W=12000, L=12100, other_itemized=5000, standard=16100 → itemized_total=15890 < 16100 → **use standard**; taxable gambling=12000 (offset by std deduction in total-income terms).
6. **Repeal scenario** (`loss_cap_pct=1.00`): W=12000, L=12100 → deductible=12000, taxable=**0**.
7. **Net DFS mode**: W=12000, L=12100 → P=−100 → taxable=**0**. W=4000, L=100 → taxable=**3900**.
8. **Break-even**: t=0.22, L=10000 → after-tax ≈ 0 when P ≈ 282.

---

## 8. Legal status (as of Oct 2026)

- 90% cap on gambling loss deductions is in effect for tax years beginning after Dec 31, 2025 (One Big Beautiful Bill Act, amending IRC §165(d)).
- House Ways and Means approved a repeal 38–5 on Sept 16, 2026, retroactive to 2026, bundled in a digital-asset tax bill. Not yet law; it still needs House floor, Senate, and signature.
- Keep `loss_cap_pct` configurable and default the UI to showing both 90% and 100% results.

---

## 9. Implementation notes

- Keep the tax engine a pure function: `(W, L, config) -> TaxResult`, so it can be called per-step during a backtest and unit-tested against section 7.
- Ingest results from the existing contest simulator. The agent should read the current codebase to find the result schema rather than assuming one.
- Contest-format breakdown (cash/50-50 vs GPP): report losses generated per $ of profit by format, since the tax cost scales with gross losses and low-margin high-churn formats are hit hardest.

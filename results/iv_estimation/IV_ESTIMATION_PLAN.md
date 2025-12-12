# Panel IV Estimation Plan: Broadband Demand Elasticity (EU vs EaP)

## Objective
Implement rigorous panel IV estimation to address price endogeneity when estimating broadband demand elasticity, comparing EU vs Eastern Partnership countries.

---

## 1. Instrument Strategy

### Primary Instruments (Recommended)
| Instrument | Variable | Availability | Justification |
|-----------|----------|--------------|---------------|
| Mobile broadband price | `mobile_broad_price_ppp` | 2013-2024 | Supply-side cost shifter; shares infrastructure costs with fixed |
| Regulatory quality | `regulatory_quality_estimate` | Full panel | Affects investment/supply but not demand directly |

### Secondary (Robustness)
- Lagged prices (`log_price_lag1`, `log_price_lag2`) - predetermined but weaker theoretically

---

## 2. Model Specification

### Structural Equation (Second Stage)
```
log(subs_it) = alpha_i + gamma_t + beta*log(price_it) + delta*log(price_it)*EaP_i + X'_it*theta + epsilon_it
```

### First Stage
```
log(price_it) = pi_0 + pi_1*Z_it + pi_2*X_it + mu_i + nu_t + v_it
```

### Key Variables
- **Dependent**: `fixed_broadband_subs_i992b` (per 100 people) or `internet_users_pct_i99H`
- **Endogenous**: `log_price`, `log_price_x_eap` (both instrumented)
- **Instruments**: Mobile price + Regulatory quality (and their EaP interactions)
- **Controls**: GDP per capita, R&D expenditure, secure servers

---

## 3. Implementation Steps

### Step 1: Data Preparation
- Create log transformations
- Generate regional indicators and interactions
- Create lagged price variables
- Set panel index (country, year)

### Step 2: First-Stage Diagnostics
- Run first-stage regression: `log_price ~ instruments + controls + FE`
- Calculate F-statistic for excluded instruments (threshold: F > 10)
- Calculate partial R-squared
- Test instrument relevance

### Step 3: Main IV Estimation
- **Approach A**: IV2SLS with interaction terms (instrument both `log_price` and `log_price_x_eap`)
- **Approach B**: Separate IV regressions for EU and EaP (robustness)
- **Approach C**: Control function approach (for complex interactions)

### Step 4: Diagnostic Tests
| Test | Purpose | Implementation |
|------|---------|----------------|
| First-stage F | Weak instruments | F > 10 (Stock-Yogo) |
| Durbin-Wu-Hausman | Endogeneity | Compare OLS vs IV |
| Hansen J-test | Overidentification | Valid if p > 0.05 |
| Anderson-Rubin | Weak-IV robust CI | If F < 10 |

### Step 5: Regional Elasticity Calculation
```python
EU elasticity = beta_price
EaP elasticity = beta_price + beta_interaction
SE(EaP) = sqrt(Var(beta_price) + Var(beta_int) + 2*Cov(beta_price, beta_int))
```

---

## 4. Robustness Checks

### 4.1 Alternative Instruments
- Mobile price only
- Regulatory quality only
- Lagged prices only
- All instruments combined (overidentification test)

### 4.2 Time Periods
- Pre-COVID (2010-2019) - primary
- Full period (2010-2024)
- Early (2010-2014) vs Late (2015-2019)

### 4.3 Alternative Dependent Variables
- Fixed broadband subscriptions per 100
- Internet users percentage
- Mobile subscriptions (falsification)

### 4.4 Control Variable Sensitivity
- Minimal: GDP only
- Baseline: GDP + R&D + secure servers
- Extended: + regulatory quality + urbanization + education

---

## 5. Output Deliverables

### Main Results Table (Pre-COVID 2010-2019)
| | (1) OLS | (2) IV Mobile | (3) IV RegQual | (4) IV Both | (5) IV Lagged |
|--|---------|---------------|----------------|-------------|---------------|
| EU Elasticity | coef (SE) | coef (SE) | coef (SE) | coef (SE) | coef (SE) |
| EaP Elasticity | coef (SE) | coef (SE) | coef (SE) | coef (SE) | coef (SE) |
| EaP/EU Ratio | X.Xx | X.Xx | X.Xx | X.Xx | X.Xx |
| First-stage F | - | F | F | F | F |
| Hausman p-val | - | p | p | p | p |
| Hansen J p-val | - | - | - | p | p |
| N | N | N | N | N | N |

### Main Results Table (Full Period 2010-2024)
| | (6) OLS | (7) IV Mobile | (8) IV RegQual | (9) IV Both | (10) IV Lagged |
|--|---------|---------------|----------------|-------------|----------------|
| EU Elasticity | coef (SE) | coef (SE) | coef (SE) | coef (SE) | coef (SE) |
| EaP Elasticity | coef (SE) | coef (SE) | coef (SE) | coef (SE) | coef (SE) |
| First-stage F | - | F | F | F | F |
| N | N | N | N | N | N |

**Best specification selection**: Choose the IV specification with (1) F > 10, (2) J-test p > 0.05, (3) economically meaningful elasticities

### Additional Outputs
- First-stage regression table
- Robustness matrix across specifications
- Diagnostic summary
- LaTeX-ready tables for paper

---

## 6. File to Create

**New file**: `code/analysis/05_panel_iv_estimation.py`

### Structure
```python
# 1. Imports and configuration
# 2. Data preparation functions
# 3. First-stage analysis
# 4. Main IV estimation (with interactions)
# 5. Diagnostic tests
# 6. Regional elasticity calculation
# 7. Robustness checks loop
# 8. Table generation
# 9. Main execution
```

---

## 7. Critical Files to Reference

- `code/analysis/00_comprehensive_method_diagnostic.py` - IV2SLS pattern (lines 388-548)
- `code/analysis/02_main_analysis.py` - Regional elasticity calculation (lines 59-99)
- `code/utils/config.py` - Country lists and variable definitions
- `code/analysis/04_comprehensive_table.py` - LaTeX table generation pattern

---

## 8. User Preferences (Confirmed)

1. **Instrument strategy**: **Try all and pick best** - Run multiple instrument combinations, report whichever has best diagnostics (F-stat > 10, valid J-test)
2. **Sample period**: **Both as co-equal** - Report pre-COVID (2010-2019) AND full period (2010-2024) as main results
3. **Regional approach**: **Pooled with interactions** - Single regression with Price x EaP interaction (more efficient)
4. **Weak instrument handling**: If F < 10, use LIML or Anderson-Rubin CIs

---

## 9. Actual Results (Post-Implementation)

### What We Found

**OLS Two-Way FE (Baseline):**
- Pre-COVID: EU = -0.061*, EaP = -0.458***, Ratio = 7.5x
- Full Period: EU = -0.098*, EaP = -0.101, Ratio = 1.0x

**IV Results:**
- Mobile price: WEAK instrument (F = 2.6-3.2)
- Regulatory quality: VERY WEAK (F ≈ 0)
- Lagged prices: STRONG (F = 48-131) but elasticities shrink to ~0

**Hausman Test:**
- p = 0.022 → Price IS endogenous
- OLS estimates are biased upward

### Key Insight

The original hypothesis was partially reversed:
- Expected: IV estimates larger than OLS (measurement error bias)
- Found: IV estimates SMALLER than OLS (reverse causality/omitted variables bias)

This suggests OLS elasticities are upper bounds, and true causal elasticities may be near zero.

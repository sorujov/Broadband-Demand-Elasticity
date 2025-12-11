# Complete Guide: Missing Data Analysis for Telecommunications Policy Paper

## 📋 Overview

This guide provides a **publication-grade framework** for handling missing data in your broadband demand elasticity study for EU and EaP countries.

---

## 🎯 Executive Summary

**Your Current Approach:** Too simplistic for Telecommunications Policy
- Forward fill for variables <10% missing
- Drop variables >70% missing
- No theoretical justification

**New Rigorous Approach:** Publication-ready
- Test missing data mechanisms (MCAR/MAR/MNAR)
- Multiple imputation with Rubin's rules
- Sensitivity analysis with 3+ methods
- Clear methodology documentation

---

## 📊 Complete Workflow

### **Step 1: Exploratory Analysis** (File: `03_rigorous_missing_analysis.py`)

Run this FIRST to understand your missing data:

```bash
python code/data_preparation/03_rigorous_missing_analysis.py
```

**What it does:**
1. **MCAR Test**: Tests if data is Missing Completely at Random
2. **Pattern Analysis**: Identifies which variables are missing together
3. **Method Comparison**: Compares 6 imputation methods
4. **Multiple Imputation**: Creates 5 imputed datasets using MICE
5. **Documentation**: Generates methodology text for your paper

**Outputs:** (saved to `data/processed/missing_data_analysis/`)
- `01_overall_missingness.xlsx` - Variable-level statistics
- `02_missing_by_year.xlsx` - Temporal patterns
- `03_comissingness_heatmap.png` - Which variables are missing together
- `04_imputation_method_comparison.xlsx` - Compare 6 methods
- `imputed_data_m1.xlsx` through `m5.xlsx` - 5 completed datasets
- `05_methodology_text_for_paper.txt` - Copy-paste to your paper

---

### **Step 2: Regression Analysis** (File: `06_regression_with_multiple_imputation.py`)

Run regressions on ALL 5 imputed datasets and pool results:

```bash
python code/analysis/06_regression_with_multiple_imputation.py
```

**What it does:**
1. Runs regression on each of the 5 imputed datasets
2. Pools results using **Rubin's Rules**:
   - Combines coefficient estimates
   - Adjusts standard errors for imputation uncertainty
   - Calculates proper confidence intervals
3. Compares with listwise deletion (robustness)
4. Reports diagnostics (FMI, RIV)

**Outputs:** (saved to `data/processed/mi_regression_results/`)
- `pooled_results_baseline.xlsx` - Main results
- `pooled_results_twoway.xlsx` - Two-way FE results
- Comparison with complete case analysis

---

## 📝 What to Report in Your Paper

### **Section 3.2: Data and Missing Values**

```
Our dataset exhibits missing values ranging from X% to Y% across key 
variables. We follow best practices from the missing data literature 
(Rubin, 1976; Little & Rubin, 2019) to handle this systematically.

Testing for Missing Data Mechanisms:
We test whether data is Missing Completely at Random (MCAR) using 
correlation-based diagnostics. Our analysis reveals evidence of 
Missing at Random (MAR), where missingness is related to observed 
covariates. Specifically, we find that [INSERT FINDINGS: e.g., 
"bandwidth data is more likely to be missing in earlier years and 
for smaller countries"].

Given the MAR assumption, we employ Multiple Imputation by Chained 
Equations (MICE) as our primary method (van Buuren & Groothuis-Oudshoorn, 
2011). This approach creates m=5 completed datasets by iteratively 
imputing each variable conditional on all others using Bayesian Ridge 
regression.

We pool results using Rubin's rules (Rubin, 1987):
  β̂_pooled = (1/m) Σ β̂_i
  SE(β̂_pooled) = √[W + (1+1/m)B]

where W is within-imputation variance and B is between-imputation 
variance.
```

### **Section 4: Results**

**Main Table - Elasticity Estimates:**

| Method | Price Elasticity | Std. Error | t-stat | 95% CI | FMI |
|--------|------------------|------------|--------|---------|-----|
| Multiple Imputation (MICE) | -0.XXX | 0.XXX | -X.XX | [-0.XX, -0.XX] | 0.XX |
| Forward Fill (robustness) | -0.XXX | 0.XXX | -X.XX | [-0.XX, -0.XX] | - |
| Listwise Deletion (robustness) | -0.XXX | 0.XXX | -X.XX | [-0.XX, -0.XX] | - |

**Notes:** 
- Multiple Imputation uses m=5 imputations with Rubin's pooling
- FMI = Fraction of Missing Information
- Standard errors clustered at country level
- All specifications include country and year fixed effects

**In text:**
```
Our primary estimates using multiple imputation indicate a price 
elasticity of -0.XXX (SE = 0.XXX, p < 0.01), suggesting [INTERPRETATION]. 
The Fraction of Missing Information (FMI = 0.XX) indicates that XX% 
of the variance in our estimates is attributable to missing data, 
which is moderate [or low/high] and within acceptable bounds 
(Schafer, 1997).

Robustness checks using forward fill and listwise deletion yield 
qualitatively similar results (elasticities of -0.XXX and -0.XXX, 
respectively), providing confidence that our findings are not driven 
by the imputation method.
```

---

## 🔬 Understanding the Diagnostics

### **Fraction of Missing Information (FMI)**
- **Range:** 0 to 1
- **Interpretation:**
  - FMI < 0.30: Low missingness impact → Results robust
  - 0.30 ≤ FMI < 0.50: Moderate impact → Multiple imputation essential
  - FMI ≥ 0.50: High impact → Consider data quality issues
- **Report:** Always report FMI in your main results

### **Relative Increase in Variance (RIV)**
- Measures how much imputation increases standard errors
- RIV = (1 + 1/m) × B / W
- Higher RIV → More uncertainty from missing data

### **Between-Imputation Variance (B)**
- Variability in estimates across the 5 imputations
- High B suggests:
  - Substantial uncertainty about missing values
  - Need for more imputations (increase m to 10)
  - Potential MNAR issues

---

## ✅ Quality Checks

Before submitting your paper, verify:

1. **□ Missing data mechanism tested**
   - Run MCAR test
   - Document findings in paper
   - Justify MAR assumption

2. **□ Multiple methods compared**
   - MICE (primary)
   - Forward fill (robustness)
   - Listwise deletion (robustness)
   - Show results are consistent

3. **□ Proper pooling implemented**
   - Used Rubin's rules
   - Reported FMI and RIV
   - Correct degrees of freedom

4. **□ Sensitivity analysis**
   - Vary m (5, 10, 20 imputations)
   - Try different imputation models
   - Check if results stable

5. **□ Documentation complete**
   - Method described in detail
   - References cited properly
   - Supplementary materials prepared

---

## 📚 Key References for Your Paper

**Must cite:**
1. **Rubin, D.B. (1976).** "Inference and Missing Data." *Biometrika*, 63(3), 581-592.
   - Foundational paper on missing data theory

2. **Rubin, D.B. (1987).** *Multiple Imputation for Nonresponse in Surveys*. Wiley.
   - Standard reference for MI methodology

3. **Little, R.J.A., & Rubin, D.B. (2019).** *Statistical Analysis with Missing Data* (3rd ed.). Wiley.
   - Comprehensive textbook - cite for general framework

4. **van Buuren, S., & Groothuis-Oudshoorn, K. (2011).** "mice: Multivariate Imputation by Chained Equations in R." *Journal of Statistical Software*, 45(3).
   - Cite for MICE method

5. **Schafer, J.L. (1997).** *Analysis of Incomplete Multivariate Data*. Chapman & Hall.
   - Cite for FMI interpretation

**Good to cite (telecommunications specific):**
6. **Greenstein, S., & McDevitt, R.C. (2011).** "Evidence of a Modest Price Decline in US Broadband Services." *Information Economics and Policy*, 23(2), 200-211.
   - Handling missing data in broadband research

---

## 🎯 Reviewer Expectations (Telecommunications Policy)

Top-tier journals expect:

1. **Theoretical Justification**
   - Why is data MAR vs MCAR vs MNAR?
   - What is the assumed missing mechanism?
   - Why is MI appropriate for your context?

2. **Multiple Methods**
   - Never rely on single imputation
   - Show robustness across approaches
   - Explain why results differ (if they do)

3. **Proper Uncertainty**
   - Don't treat imputed values as "known"
   - Report pooled standard errors
   - Show uncertainty from imputation

4. **Transparency**
   - Document all decisions
   - Provide supplementary materials
   - Make code/data available (if possible)

---

## 🚀 Next Steps

### **Immediate (Today):**
1. Run `03_rigorous_missing_analysis.py`
2. Review all outputs in `missing_data_analysis/` folder
3. Read the generated methodology text

### **This Week:**
1. Run `06_regression_with_multiple_imputation.py`
2. Compare pooled results with your current estimates
3. Draft missing data section of paper

### **Before Submission:**
1. Create supplementary table comparing methods
2. Add robustness checks to appendix
3. Prepare detailed documentation
4. Have co-author review methodology

---

## 💡 Pro Tips for Publication

1. **Be Transparent**
   - Show what you tried
   - Explain why you chose MICE
   - Document sensitivity to choices

2. **Use Appendix Strategically**
   - Main text: Concise methodology + main results
   - Appendix: Detailed diagnostics + robustness checks
   - Supplementary: Code + additional tables

3. **Anticipate Reviewer Questions**
   - "Why not just drop missing observations?"
   - "How do you know MAR holds?"
   - "Are results sensitive to imputation model?"
   - **Have answers ready!**

4. **Frame Positively**
   - Don't apologize for missing data
   - Emphasize rigorous handling
   - Show it doesn't affect conclusions

---

## ❓ Common Issues & Solutions

### Issue: "My results change across imputations"
**Solution:** This is expected! High between-imputation variance suggests uncertainty. Report pooled estimates with wider CIs.

### Issue: "FMI is too high (>0.50)"
**Solution:** 
- Check if key variables have >50% missing → exclude them
- Consider sensitivity analysis with different m
- Discuss as limitation if unavoidable

### Issue: "Reviewer asks about MNAR"
**Solution:**
- Acknowledge possibility
- Explain why MAR is reasonable assumption
- Run sensitivity analysis with pattern-mixture models

### Issue: "Results differ from listwise deletion"
**Solution:**
- This is actually good - shows imputation added information
- Explain which estimate is more reliable (usually MI)
- Discuss in robustness section

---

## 📧 Questions?

If you encounter issues:
1. Check output files in `missing_data_analysis/`
2. Review error messages carefully
3. Verify data format (country-year panel)
4. Ensure all required columns exist

**Good luck with your Telecommunications Policy submission!** 🎓

---

## 🔖 Quick Reference Card

```
WORKFLOW:
1. Run: 03_rigorous_missing_analysis.py
2. Review: missing_data_analysis/ folder
3. Run: 06_regression_with_multiple_imputation.py
4. Report: Pooled estimates with FMI
5. Robustness: Compare 3 methods
6. Document: Methodology section

REPORT IN PAPER:
- Missing percentages by variable
- MCAR test results (or MAR justification)
- MICE with m=5 imputations
- Pooled estimates (β, SE, CI, FMI)
- Robustness checks (forward fill, listwise)

KEY METRICS:
- FMI < 0.30 → Good
- 0.30 ≤ FMI < 0.50 → Acceptable
- FMI ≥ 0.50 → Concerning

CITATIONS:
- Rubin (1987) - MI methodology
- Little & Rubin (2019) - General framework
- van Buuren & Groothuis-Oudshoorn (2011) - MICE
- Schafer (1997) - FMI interpretation
```

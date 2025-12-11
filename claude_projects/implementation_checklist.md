# Implementation Checklist: Missing Data Analysis for Your Paper

## 🎯 Your Goal
Publish a rigorous broadband demand elasticity study in **Telecommunications Policy** (Q1 journal) with publication-grade missing data handling.

---

## ✅ Complete Implementation Checklist

### **Phase 1: Setup** (15 minutes)

- [ ] **1.1** Save all 4 Python scripts to your project:
  - `code/data_preparation/03_rigorous_missing_analysis.py`
  - `code/analysis/06_regression_with_multiple_imputation.py`
  - `code/analysis/07_create_publication_outputs.py`
  - Review: Complete Guide markdown

- [ ] **1.2** Install required packages:
```bash
pip install missingno scikit-learn scipy statsmodels linearmodels openpyxl
```

- [ ] **1.3** Verify your data file exists:
  - `data/processed/data_merged_with_series.xlsx`
  - Contains: country, year, and all ITU/World Bank variables

---

### **Phase 2: Missing Data Analysis** (1-2 hours)

- [ ] **2.1** Run exploratory analysis:
```bash
python code/data_preparation/03_rigorous_missing_analysis.py
```

- [ ] **2.2** Review outputs in `data/processed/missing_data_analysis/`:
  - [ ] `01_overall_missingness.xlsx` - Which variables have most missing?
  - [ ] `02_missing_by_year.xlsx` - Is missingness increasing over time?
  - [ ] `03_comissingness_heatmap.png` - Which variables are missing together?
  - [ ] `04_imputation_method_comparison.xlsx` - Which method performs best?
  - [ ] `imputed_data_m1.xlsx` through `m5.xlsx` - 5 completed datasets

- [ ] **2.3** Answer these questions from the analysis:
  - [ ] What % of data is missing for key variables (price, subscriptions)?
  - [ ] Is data MCAR or MAR? (Check correlation test results)
  - [ ] Which imputation method has lowest bias? (Check comparison table)
  - [ ] What's the FMI for key variables? (Will determine after regression)

**📝 DECISION POINT:** Based on analysis, confirm:
- MICE is appropriate method (usually yes if MAR detected)
- Missing < 50% for key variables (drop if >50%)
- Multiple imputations needed (yes if MAR/MNAR)

---

### **Phase 3: Regression Analysis** (2-3 hours)

- [ ] **3.1** Run regressions on imputed datasets:
```bash
python code/analysis/06_regression_with_multiple_imputation.py
```

- [ ] **3.2** Review regression outputs in `data/processed/mi_regression_results/`:
  - [ ] `pooled_results_baseline.xlsx` - Main price elasticity estimate
  - [ ] `pooled_results_twoway.xlsx` - With time fixed effects
  - [ ] Note the FMI values (fraction of missing information)

- [ ] **3.3** Record your main findings:
  - [ ] Price elasticity (MI): _______ (SE: _______)
  - [ ] FMI: _______ (< 0.30 is good, 0.30-0.50 acceptable)
  - [ ] Price elasticity (forward fill): _______ (for comparison)
  - [ ] Price elasticity (listwise): _______ (for robustness)

**📝 DECISION POINT:** Are results stable across methods?
- If YES → Good! MI is working well
- If NO → Investigate why (check FMI, between-imputation variance)

---

### **Phase 4: Publication Outputs** (1 hour)

- [ ] **4.1** Generate all tables and figures:
```bash
python code/analysis/07_create_publication_outputs.py
```

- [ ] **4.2** Check outputs in `data/processed/publication_outputs/`:
  - [ ] `Table1_Descriptives.xlsx` + `.tex`
  - [ ] `Table2_MissingPatterns.xlsx` + `.tex`
  - [ ] `Table3_MainResults.xlsx` + `.tex`
  - [ ] `Table4_Robustness.xlsx` + `.tex`
  - [ ] `Figure1_MissingPatterns.png` (300 DPI)
  - [ ] `Figure2_MethodComparison.png` (300 DPI)

- [ ] **4.3** Verify table quality:
  - [ ] Numbers are properly formatted (2-4 decimal places)
  - [ ] Statistical significance indicated (*, **, ***)
  - [ ] Sample sizes reported
  - [ ] All confidence intervals present

---

### **Phase 5: Paper Writing** (4-6 hours)

- [ ] **5.1** Write "Data and Methodology" section:
  - [ ] Describe your dataset (N, T, countries)
  - [ ] Report missing percentages (use Table 1)
  - [ ] Explain MCAR test results
  - [ ] Justify MI approach (cite Rubin 1987)
  - [ ] Describe MICE procedure (m=5, Bayesian Ridge)
  - [ ] Explain Rubin's pooling rules

**Template paragraph:**
```
Our dataset exhibits missing values ranging from X% to Y%. 
Testing for missing data mechanisms reveals evidence of MAR, 
where missingness correlates with observed covariates. We 
employ Multiple Imputation by Chained Equations (MICE) with 
m=5 imputations, pooling results using Rubin's rules.
```

- [ ] **5.2** Write "Results" section:
  - [ ] Report pooled elasticity estimate (Table 3)
  - [ ] Mention FMI in text ("FMI of 0.XX indicates...")
  - [ ] Compare with robustness checks (Table 4)
  - [ ] Discuss economic interpretation

**Template paragraph:**
```
Our primary estimates using multiple imputation indicate a 
price elasticity of -0.XXX (SE = 0.XXX, p < 0.01). The 
Fraction of Missing Information (FMI = 0.XX) is moderate, 
suggesting XX% of variance is due to missing data. Robustness 
checks using forward fill (-0.XXX) and listwise deletion 
(-0.XXX) yield qualitatively similar results.
```

- [ ] **5.3** Create "Robustness Checks" subsection:
  - [ ] Show Table 4 (method comparison)
  - [ ] Discuss why results are stable (or not)
  - [ ] Address sensitivity to m (number of imputations)

- [ ] **5.4** Write "Data Limitations" section:
  - [ ] Acknowledge missing data challenge
  - [ ] Explain how you addressed it rigorously
  - [ ] Mention any variables excluded (if >50% missing)

---

### **Phase 6: Supplementary Materials** (2 hours)

- [ ] **6.1** Create Appendix A: Detailed Missing Data Analysis
  - [ ] Include Figure 1 (missing patterns)
  - [ ] Include Table 2 (patterns by region/time)
  - [ ] Add text explaining patterns

- [ ] **6.2** Create Appendix B: Robustness Checks
  - [ ] Include Figure 2 (method comparison)
  - [ ] Include Table 4 (full comparison)
  - [ ] Show sensitivity to m (5 vs 10 vs 20 imputations)

- [ ] **6.3** Create Appendix C: Technical Details
  - [ ] MICE algorithm description
  - [ ] Rubin's pooling formulas
  - [ ] Convergence diagnostics

- [ ] **6.4** Prepare code/data availability statement:
```
Data and code are available upon request. Imputed datasets 
were created using the mice package in Python, following 
van Buuren & Groothuis-Oudshoorn (2011).
```

---

### **Phase 7: Quality Checks** (1 hour)

- [ ] **7.1** Verify citations:
  - [ ] Rubin (1976) - MCAR/MAR theory
  - [ ] Rubin (1987) - MI methodology
  - [ ] Little & Rubin (2019) - General framework
  - [ ] van Buuren & Groothuis-Oudshoorn (2011) - MICE
  - [ ] Schafer (1997) - FMI interpretation

- [ ] **7.2** Check consistency across tables:
  - [ ] Same sample sizes reported
  - [ ] Elasticity estimates consistent
  - [ ] Standard errors properly pooled
  - [ ] All decimal places match

- [ ] **7.3** Verify figure quality:
  - [ ] 300 DPI or higher
  - [ ] Clear labels and legends
  - [ ] Readable fonts (≥10pt)
  - [ ] Publication-quality formatting

- [ ] **7.4** Run sensitivity analysis:
  - [ ] Re-run with m=10 (check if results stable)
  - [ ] Try different estimator in MICE (Random Forest)
  - [ ] Exclude countries with >30% missing
  - [ ] Check if elasticity changes meaningfully

---

### **Phase 8: Pre-Submission Review** (2 hours)

- [ ] **8.1** Self-review checklist:
  - [ ] Missing data mechanism clearly identified
  - [ ] Imputation method justified theoretically
  - [ ] Multiple robustness checks shown
  - [ ] Uncertainty properly quantified (FMI, pooled SEs)
  - [ ] Results not overstated

- [ ] **8.2** Anticipate reviewer questions:
  - [ ] **Q:** "Why not just drop missing observations?"
    - **A:** "Would lose X% of data and introduce selection bias"
  - [ ] **Q:** "How do you know MAR holds?"
    - **A:** "MCAR test rejects (p<0.05), MAR is standard assumption"
  - [ ] **Q:** "Are results sensitive to imputation?"
    - **A:** "No, Table 4 shows consistent elasticities across methods"

- [ ] **8.3** Have co-author/colleague review:
  - [ ] Missing data section clear?
  - [ ] Tables self-explanatory?
  - [ ] Figures publication-quality?
  - [ ] All technical details correct?

---

## 📊 Expected Outputs for Journal

### **Main Manuscript:**
1. **Table 1:** Descriptive statistics (in text or appendix)
2. **Table 2:** Main regression results (MI estimates)
3. **Figure 1:** Missing data patterns (in methodology section)

### **Appendix:**
1. **Table A1:** Missing data patterns by region/time
2. **Table A2:** Robustness checks (method comparison)
3. **Figure A1:** Distribution comparison across methods

### **Supplementary Materials:**
1. Detailed MICE algorithm description
2. Convergence diagnostics
3. Additional robustness checks (m=10, m=20)

---

## ⚠️ Common Pitfalls to Avoid

- [ ] **DON'T** treat imputed values as "true" values
  - Always acknowledge uncertainty (report FMI)

- [ ] **DON'T** rely on single imputation
  - Multiple imputations essential for valid inference

- [ ] **DON'T** ignore missing data mechanism
  - Test MCAR vs MAR, justify your assumption

- [ ] **DON'T** cherry-pick best imputation
  - Always pool across all m imputations

- [ ] **DON'T** hide robustness checks
  - Show multiple methods, discuss differences

---

## 🎯 Success Criteria

Your missing data handling is publication-ready if:

✅ **Theoretically justified**
- MCAR/MAR tested and documented
- MI method appropriate for your mechanism
- Pooling follows Rubin's rules

✅ **Empirically robust**
- Results stable across methods (MI, LOCF, listwise)
- FMI < 0.50 for key variables
- Elasticity estimates have reasonable CIs

✅ **Clearly communicated**
- Methodology section explains approach
- Tables show comparison across methods
- Figures illustrate missing patterns
- Limitations acknowledged

---

## 📧 Final Pre-Submission Checklist

Before clicking "Submit" to Telecommunications Policy:

- [ ] All tables formatted per journal guidelines
- [ ] All figures meet DPI requirements (usually 300+)
- [ ] Supplementary materials prepared
- [ ] Code documentation complete
- [ ] Co-authors have approved
- [ ] Missing data section peer-reviewed
- [ ] All references cited properly
- [ ] Responded to internal reviewers

---

## 🚀 Estimated Timeline

| Phase | Time | Cumulative |
|-------|------|------------|
| Setup | 15 min | 15 min |
| Missing data analysis | 2 hours | 2h 15min |
| Regression analysis | 3 hours | 5h 15min |
| Publication outputs | 1 hour | 6h 15min |
| Paper writing | 6 hours | 12h 15min |
| Supplementary materials | 2 hours | 14h 15min |
| Quality checks | 1 hour | 15h 15min |
| Pre-submission review | 2 hours | 17h 15min |

**Total: ~17 hours** (2-3 working days)

---

## 💡 Pro Tips

1. **Start early:** Don't wait until submission deadline
2. **Document everything:** Keep detailed notes on decisions
3. **Seek feedback:** Have methodologist review your approach
4. **Be transparent:** Show what you tried, not just what worked
5. **Stay current:** Check recent TP papers for examples

---

## 🎓 You're Ready When...

✅ You can explain your missing data mechanism to a reviewer

✅ Your elasticity estimate has proper uncertainty quantification

✅ Your robustness checks show consistent results

✅ Your tables and figures are publication-quality

✅ Your methodology section follows best practices

---

## 📞 Need Help?

If you encounter issues:
1. Check output files for error messages
2. Verify data format (country-year panel)
3. Ensure all required columns exist
4. Review Python package versions
5. Check this guide's troubleshooting section

**Good luck with your Telecommunications Policy submission!** 🎯

---

**Remember:** Rigorous missing data handling is what separates Q1 publications from desk rejects. You've got this! 💪

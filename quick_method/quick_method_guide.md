# 🎯 QUICK METHOD SELECTION GUIDE
## Your Shortest Path to Published Results

---

## 🚀 **RUN THIS FIRST** (10 minutes)

```bash
python code/analysis/00_comprehensive_method_diagnostic.py
```

**What it does:** Tests ALL econometric methods on your data and tells you exactly which one(s) to use.

---

## 📊 **What You'll Get**

### Output Files (in `data/processed/method_diagnostic/`):

1. **`method_recommendations.xlsx`** ← START HERE
   - Ranked list of suitable methods
   - Elasticity estimates from each
   - Issues and priorities

2. **`method_comparison_table.xlsx`**
   - Side-by-side comparison
   - Statistical tests
   - Diagnostic statistics

3. **`decision_tree.txt`**
   - Visual flowchart
   - Step-by-step logic
   - Clear recommendation

---

## 🎯 **Most Likely Outcomes & Your Path**

### **SCENARIO A: You Have Strong Instruments** (Best Case)
✅ **First-stage F-stat > 10**

**YOUR PATH:**
1. **PRIMARY:** IV/2SLS
   - Use regulatory quality and/or mobile price as instruments
   - Report this as main result
   - Elasticity will be your best estimate

2. **ROBUSTNESS:** Two-way FE
   - Compare to show results are stable
   - Include in Table 2 of paper

3. **COMPARISON:** Pooled OLS
   - Show how much bias there was
   - Proves IV was necessary

**PAPER STRUCTURE:**
- Table 1: Descriptives
- Table 2: Main results (IV/2SLS)
- Table 3: Robustness (FE, OLS comparison)
- Text: "We use IV/2SLS as our primary estimation strategy due to endogeneity concerns. Our instruments pass all validity tests (F > 10)..."

**TIME TO RESULTS:** 2-3 days

---

### **SCENARIO B: Weak Instruments** (Common)
⚠️ **First-stage F-stat < 10**

**YOUR PATH:**
1. **PRIMARY:** Two-Way Fixed Effects
   - Controls for country + time effects
   - Report as main result
   - Most robust non-IV method

2. **ROBUSTNESS:** One-Way FE
   - Show results consistent
   - Include in appendix

3. **ACKNOWLEDGE:** Endogeneity
   - Discuss as limitation
   - Explain why IV not feasible

**PAPER STRUCTURE:**
- Table 1: Descriptives
- Table 2: Main results (Two-way FE)
- Table 3: Robustness (One-way FE, OLS)
- Text: "While IV estimation would be ideal, our instruments are weak (F < 10). We therefore employ two-way fixed effects as our primary approach, which controls for..."

**TIME TO RESULTS:** 1-2 days

---

### **SCENARIO C: Limited Within Variation** (Tricky)
⚠️ **Price varies mostly between countries, not within**

**YOUR PROBLEM:**
- Fixed effects "throw away" most price variation
- Elasticity estimates imprecise (large SEs)
- Hard to identify causal effect

**YOUR PATH:**
1. **ACKNOWLEDGE:** Limited variation
   - "Price exhibits limited within-country variation..."

2. **USE:** Between effects or pooled OLS
   - Compare countries with different price levels
   - Not ideal but only option

3. **ROBUST CHECKS:**
   - Try different time periods
   - Sub-sample analysis
   - Non-linear specifications

**PAPER STRUCTURE:**
- Emphasize descriptive analysis
- Focus on cross-country patterns
- Acknowledge causal identification challenges
- Position as "suggestive evidence"

**TIME TO RESULTS:** 2-3 days (more exploratory)

---

## 🔍 **How to Interpret Diagnostic Output**

### Key Metrics to Check:

| Metric | Good | Acceptable | Problematic |
|--------|------|------------|-------------|
| **First-stage F** | > 10 | 5-10 | < 5 |
| **Within variation ratio** | > 20% | 10-20% | < 10% |
| **Panel completeness** | > 90% | 70-90% | < 70% |
| **N × T** | > 300 | 150-300 | < 150 |

### What Each Test Tells You:

**✅ TEST 1: Data Structure**
- Tells you: Is panel balanced? Enough obs?
- If YES → All methods available
- If NO → Some methods restricted

**✅ TEST 2: Pooled OLS**
- Always works, but likely biased
- Use as baseline comparison
- Never report as main result

**✅ TEST 3: Fixed Effects**
- Checks within-country variation
- If variation low → FE inefficient
- If variation good → FE is viable

**✅ TEST 4: Hausman Test**
- Decides FE vs RE
- Almost always says FE
- Don't worry about RE for your paper

**✅ TEST 5: Instruments**
- **MOST CRITICAL TEST**
- Determines if IV is possible
- F > 10 = strong, F < 5 = useless

**✅ TEST 6: Endogeneity**
- Compares OLS vs IV
- Large difference = endogeneity present
- Justifies using IV

**✅ TEST 7: Two-Way FE**
- Best non-IV method
- Controls time + country effects
- Safe default if IV doesn't work

---

## ⚡ **Quick Decision Flowchart**

```
START
  │
  ├─ Run diagnostic script (10 min)
  │
  ├─ Open method_recommendations.xlsx
  │
  ├─ Check Rank 1 method
  │  │
  │  ├─ "IV/2SLS" ?
  │  │  │
  │  │  ├─ YES → Use IV as primary ✅
  │  │  │        Add FE as robustness
  │  │  │        Done in 2-3 days
  │  │  │
  │  │  └─ NO → Check Rank 2
  │  │     │
  │  │     ├─ "Two-Way FE" ?
  │  │     │  │
  │  │     │  ├─ YES → Use 2WFE as primary ✅
  │  │     │  │        Add 1WFE as robustness
  │  │     │  │        Done in 1-2 days
  │  │     │  │
  │  │     │  └─ NO → Review issues in output
  │  │     │           May need advanced methods
  │  │     │           Consult with advisor
  │
  └─ DONE: You now know your path!
```

---

## 📝 **What to Write in Your Paper**

### If IV/2SLS Works:

**Methodology Section:**
```
We employ IV/2SLS to address endogeneity in broadband pricing. 
Our instruments are [regulatory quality / mobile broadband price], 
which satisfy both relevance (first-stage F = XX.X) and exogeneity 
conditions. The Hausman test confirms endogeneity (p < 0.05), 
validating our IV approach.
```

**Results Section:**
```
Table 2 presents our main findings. The IV/2SLS estimate indicates 
a price elasticity of -X.XXX (SE = 0.XXX, p < 0.01), suggesting 
[INTERPRETATION]. This estimate is substantially larger than the 
OLS estimate (-0.XXX), confirming measurement error and simultaneity 
bias in the price variable.
```

### If Two-Way FE is Your Best Option:

**Methodology Section:**
```
We employ two-way fixed effects regression, controlling for both 
country-specific and time-specific unobserved factors. This 
specification accounts for permanent differences across countries 
(supply infrastructure, regulation) and common time trends 
(technological progress, economic cycles).
```

**Results Section:**
```
Table 2 presents our main findings using two-way fixed effects. 
The price elasticity is -X.XXX (SE = 0.XXX, p < 0.01), indicating 
[INTERPRETATION]. Results are robust to alternative specifications 
(one-way FE) as shown in Table 3.
```

### Always Include:

**Limitations Section:**
```
While we employ rigorous econometric methods, several limitations 
remain. [IF NO STRONG IV: "We cannot fully address potential 
endogeneity in pricing decisions, though our fixed effects 
specification mitigates this concern."] [IF LIMITED VARIATION: 
"Price exhibits limited within-country variation, which reduces 
the precision of our estimates."]
```

---

## 🎯 **Success Checklist**

Before you start writing, ensure:

- [ ] Diagnostic script ran successfully
- [ ] You have elasticity estimates from 2+ methods
- [ ] You understand why one method is preferred
- [ ] You can explain instrument validity (if using IV)
- [ ] You have robustness checks ready
- [ ] You know what to write in methodology

---

## ⏱️ **Realistic Timelines**

| Your Situation | Method | Time to Results | Time to Paper |
|----------------|--------|-----------------|---------------|
| Strong instruments | IV/2SLS | 2-3 days | 1-2 weeks |
| Weak instruments | Two-way FE | 1-2 days | 1 week |
| Limited variation | Exploratory | 3-4 days | 2 weeks |

---

## 🚨 **Common Mistakes to Avoid**

1. **❌ Don't** spend weeks trying to make IV work if F < 5
   - **✅ Do** accept that FE is your best option

2. **❌ Don't** report only one method
   - **✅ Do** show robustness with 2-3 specifications

3. **❌ Don't** ignore diagnostic warnings
   - **✅ Do** address them in limitations section

4. **❌ Don't** cherry-pick best elasticity
   - **✅ Do** use theoretically justified method

5. **❌ Don't** hide negative results
   - **✅ Do** report honestly (reviewers will respect it)

---

## 💡 **Pro Tips**

1. **Run diagnostic FIRST** - Don't commit to a method before testing

2. **Document everything** - Save all test outputs for appendix

3. **Be honest about limitations** - Reviewers appreciate transparency

4. **Focus on robustness** - Multiple methods > perfect single method

5. **Ask for help early** - If diagnostic shows problems, consult advisor

---

## 📧 **When to Get Help**

Contact an econometrician if:
- All F-stats < 5 (very weak instruments)
- Within variation < 5% (almost no variation)
- Panel completeness < 50% (lots of missing data)
- Diagnostic script crashes repeatedly
- Results are economically implausible (elasticity > 0 or < -10)

---

## 🎓 **Bottom Line**

**Your shortest path to publication:**

1. **Run diagnostic** (10 minutes)
2. **Read recommendations** (5 minutes)  
3. **Implement top method** (1-3 days)
4. **Add robustness checks** (1 day)
5. **Write paper** (1-2 weeks)

**Total time: 2-3 weeks from now to submission-ready manuscript**

---

## ✅ **You're Ready When...**

- [ ] You know which method to use (IV or FE)
- [ ] You have elasticity estimates from 2+ methods
- [ ] You understand why results differ (if they do)
- [ ] You can defend your method choice to reviewers
- [ ] Your robustness checks confirm main results

---

**NOW RUN THE DIAGNOSTIC AND GET YOUR ANSWER! 🚀**

```bash
python code/analysis/00_comprehensive_method_diagnostic.py
```

**Time to answer: ~10-15 minutes**
**Time saved: 2-4 weeks of trial-and-error**

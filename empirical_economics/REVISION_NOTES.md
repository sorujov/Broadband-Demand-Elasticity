# Revision notes: from the JRE/IEP rejections to the Empirical Economics submission

**Date:** 16 September 2026
**New target:** *Empirical Economics* (Springer). ABDC A / AJG 2 → ADA Category 5: 2,000 AZN base, which becomes **1,200 AZN** and **2.0 recognition points** with two external co-authors. No submission fee.
**Alternative with the same category:** *Applied Economics*. It charges a USD 270 submission fee and caps papers at 8,000 words including references.

**New title:** *What can country panels tell us about broadband price responsiveness? Evidence from the European Union and the Eastern Partnership, 2010–2024*

---

## 1. What changed in the message, and why

The rejected paper made three claims:
- EaP demand was elastic before COVID (ε = −0.60).
- Both regions became inelastic by 2020–24.
- The change began in 2015.

The re-analysis shows that **these claims do not survive**:

| Check | Result |
|---|---|
| Correct log price (the old pipeline used ln(1+p)) | EaP static −0.46, not −0.60 |
| Country-specific trends | Pooled −0.23 → −0.08; post-2019 shift disappears |
| First differences | Pooled −0.07; EU −0.01 (90% CI [−0.04, 0.02]) |
| Simulation with **zero** price effect | Levels TWFE gives −0.48, and its year-specific slopes rise from −0.64 to −0.16, the same "fading elasticity" shape |
| ITU basket revision (1 GB → 5 GB in 2018) | Levels coefficient changes sign exactly there |
| EaP with EaP-specific year effects | +0.13 (n.s.): the EaP association is bloc-wide co-movement, not identified within the group |
| Few-cluster inference for EaP–EU | RI p 0.005–0.03; WCR p 0.01–0.52 depending on specification |
| Timing placebo (next year's price) | EaP −0.16 (p = 0.11): adoption growth is also related to *future* affordability gains |

The paper is therefore now a careful **"what can this design identify?"** paper:
- **EU:** a tightly bounded zero.
- **EaP:** co-movement during catch-up, not a causal price effect.
- **2020:** no break.

This answers the IEP editor's objection head-on instead of arguing against it. It is the version least likely to be rejected again for the same reason.

## 2. Data problems found and fixed

1. **Log transform.** `log_fixed_broad_price` was ln(1+p) (e.g. ARM 2010: 11.52 → 2.527 = ln 12.52). The coefficients in the submitted paper were therefore not elasticities. It is now ln p.
2. **Imputation.** `step3_process_raw_data.py` forward-filled and then interpolated in both directions (i.e. also extrapolated), across prices, subscriptions and controls. Cells affected include the ROU 2013 price and the BGR 2024 subscriptions. No price or subscription value is imputed now.
3. **2019 prices.** The ITU workbook repeats the 2018 values in 2019 for 22 countries. These are now treated as missing and flagged.
4. **Mobile price.** The prefix `i271mb_` averaged several different ITU baskets. It is now a proper chain: 1 GB postpaid (2013–17), 1.5 GB (2018–20), 2 GB (2021–24).
5. **Dependent variable.** The old DV was the log *count* of subscriptions while the text said "per 100". It is now per 100 (ITU via World Bank); the count is kept as a robustness check.
6. **Data refresh.**
   - ITU price workbook 2008–2025.
   - World Bank WDI/WGI, release of 13 July 2026.
   - 2025 cannot be added yet, because subscriptions stop at 2024.
7. **New open data sources.**
   - Harmonised DMSP/VIIRS night-time lights 2009–2024 (Li et al. 2020, 2024 release), used as a control that doesn't depend on national income statistics.
   - Eurostat NUTS-2 household broadband access and regional GDP (175 regions, 2010–2021).

## 3. Point-by-point: earlier referees → where addressed

### JRE Editor
- **Six countries.** The EaP result is now presented as fragile. We add randomisation inference, WCR, leave-one-out, group-specific year effects and country slopes (§4.5, §5.2, Table 4, Fig. 2).
- **Endogeneity.**
  - Hausman-type IV: weak (F = 12 in levels, < 3 in FD); Anderson–Rubin set reported (§4.6, Table S4).
  - Arellano–Bond difference GMM with collapsed instruments (Table 3).
  - Timing placebo and tariff/GNI decomposition (Table 6).
  - We say clearly that no credible cost-side instrument exists.

### JRE Reviewer 1
- **2.1 Aggregation.** NUTS-2 panel of 175 regions × national price × less-developed indicator, including country-by-year fixed effects (§5.5, Table 7). CI reported honestly: [−0.18, 0.14].
- **2.2 TWFE heterogeneity.**
  - New literature paragraph citing de Chaisemartin & D'Haultfœuille, Goodman-Bacon and Callaway et al. (continuous treatment).
  - Formal bias derivation, eq. (3), and a simulation.
  - Country-specific slopes, unweighted mean group, and TWFE weights (EaP = 32%).
- **2.3 Standard errors.** Clustered SEs everywhere, plus the wild cluster restricted bootstrap. Driscoll–Kraay only in Table S1.
- **2.4 N = 6.** Randomisation inference is the primary check. The EaP result is qualified throughout, including the abstract.
- **2.5 "Several studies published in this journal".** Removed. This was the JRE/IEP citation mix-up.
- **2.6 Saturation vs. inelasticity.**
  - Logistic (log-odds) outcome with K = 55/70/100.
  - Headroom interaction.
  - New §6.2, which states honestly that headroom and EaP membership coincide in this sample.

### JRE Reviewer 2 — general comments
- **1 Regulation.** New §2.2 covers EU (EECC Art. 84, BCRD → GIA), US (Lifeline, ACP) and EaP (EU4Digital, EaPeReg, Belarus suspension). Also new §6.4 on regulatory implications.
- **2 Clarity of the main specification.** §4 now opens with the estimand and the main specification; the null for the post-2019 test is stated explicitly (§4.4).
- **3 Nesting of specifications / IV.** Specifications (1)–(3) are collapsed into one interaction equation. Arellano–Bond and Hausman IV are added, and the lagged-price IV is dropped.

### JRE Reviewer 2 — specific comments
- **1** Literature table added (Table 1), including Rosston et al., Liu et al., Mendez et al. and Wilson. All references were verified against Crossref.
- **2** Relation to Mendez et al. discussed.
- **3** Why constant elasticity is used; linear, log-linear and linear-log forms reported (§4.1, Table S2).
- **4** ChatGPT / Ukraine war: sample restrictions (Table S3), plus a sentence on the direction in which adding 2020–24 moves the levels coefficient (§5.3).
- **5** Sign predictions corrected: β₂ has no sign restriction (§4.4).
- **6** Each control is tied to the demand or supply factor it captures; post-treatment controls are removed (§3.2).
- **7** ITU basket definition described: allowance, speed, operator, installation excluded, bundles allowed, 2018 revision (§3.1). The mobile basket is a robustness control.
- **8** Comparison with electricity and water elasticities (§6.3).
- **9** COVID and policy attention: stated as a conjecture (§6.4).

### IEP Editor ("price elasticity cannot be credibly estimated with aggregate country-level panel data")
- The paper now agrees with this, demonstrates *why*, and shows what the panel can and cannot say. The title, abstract and conclusion are written accordingly.

## 4. Internal adversarial review

A separate referee-style review was run on the draft. Its main points and the fixes:
- **Eq. (3) does not predict the year-by-year path.** Claim removed; the simulation added.
- **FD does not fully remove trends.** FD bias formula added, plus an FD model with country effects.
- **The EaP result comes from bloc-level co-movement.** Group-specific year effects added, and the text rewritten around this.
- **The "mean group" column was precision-weighted.** Now the unweighted Pesaran–Smith mean, with the Belarus weight disclosed.
- **Unreported trend-model EU shift.** Now reported (−0.08, WCR p = 0.008).
- **EU GNI decomposition.** Now discussed.
- **Smaller fixes.** Inference wording, the CI mismatch and several jargon terms.

An automated audit checks every decimal number in the text against `results/results.json`. The only unmatched values are figures quoted from other papers.

## 5. Things you should check before submitting

- **Co-authors.** All authors have approved the current text (confirmed 16 September 2026).
- **Author contributions.** Confirm the CRediT roles.
- **Competing interests.** None declared (confirmed by the authors). Confirm ORCID details in the submission system.
- **GitHub.** The repository should contain the new `empirical_economics/` folder before the Data availability statement goes live.
- **EaP facts.** State ownership of incumbents and the Belarus EaP suspension (2021) are stated without a citation; add a source if the editor asks.
- **Online Resource.** Submit `ESM_1.pdf` as "Online Resource 1".
- **Blind review.** Empirical Economics may use double-blind review; `manuscript_anonymous.pdf` is ready if so.

## 6. How to reproduce (run from `empirical_economics/code`)

```
python build_data_v2.py      # downloads/uses ITU, WB (cached), NTL, builds panel.pkl
python regional.py           # Eurostat regional panel (needs eurostat/*.json)
python analysis.py           # Tables 3–5, 7, S1–S4 (≈2 min)
python analysis2.py          # long differences, decomposition, placebo, lights, sub-periods
python analysis3_ivfd.py     # FD Hausman IV
python analysis4_referee.py  # FD+FE, group-year FE, mean group, EaP by year, K=70 headroom
python simulate.py           # Figure S1
python figures.py; python make_tables.py; python make_esm.py
python verify_numbers.py     # audit of numbers in the text
```

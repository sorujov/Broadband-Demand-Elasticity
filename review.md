Read the paper and evaluate its publishability! I need careful analysis of the paper to be publishabe in Information Economics and Policy journal

paper.pdf
Completed 1 step
Your revised paper is now quite close to being publishable in Information Economics and Policy; I would expect it to be sent out for review and to have a realistic chance of acceptance after one substantial revision round, provided you tighten a few remaining points.
​

Fit with IEP and contribution
Strong topical fit: broadband demand, price elasticity, digital transformation, and COVID are core IEP themes, and you engage several recent IEP papers (Liu–Prince–Wallsten, Lindlacher, Sinclair, Grzybowski et al.).
​

Clear and focused contribution: you offer (i) time‑varying elasticities over 15 years, (ii) EU vs EaP heterogeneity, and (iii) a careful COVID vs secular trend narrative, all grounded in standard panel methods.
​

Policy relevance is explicit, detailed, and aligned with IEP’s “economics + policy” orientation (affordability vs infrastructure, region‑specific implications, ITU affordability metrics).
​

On “fit”, I see no red flags; if anything, the combination of ITU‑style measurement issues and IEP‑backbone references makes it attractive.

Methodology and identification
Strengths:

Clean two‑way FE with Driscoll–Kraay SE on a balanced N=33, T=15 panel, with detailed controls and a transparent equation (2)/(3).
​

You explicitly discuss TWFE criticisms and explain why Goodman‑Bacon / staggered DiD issues do not apply to your continuous price regressor and common COVID timing.
​

Identification threats are handled systematically: simultaneity (lagged prices IV and mobile‑price IV robustness), omitted variables (rich control sets and robustness matrix), and cross‑sectional dependence (DK SE vs clustered alternatives).
​

Remaining issues referees are likely to raise:

Endogeneity of prices / instrument strength

The mobile‑price IV results are informative but the first‑stage for EU is borderline and full‑sample SEs are huge; a referee may see this as weak‑instrument territory and ask you to down‑weight the IV narrative.
​

Practical fix: keep IV as robustness, but in the text frame your preferred estimates as OLS‑FE with DK SE, explicitly acknowledging that IV is suggestive but not precise for the EU subsample.

Functional‑form and constant elasticity

You assume log‑log constant elasticity; this is standard and you justify it, but you already show time‑variation, so a referee might ask about non‑linearity in levels (e.g. elasticity at different penetration levels) beyond the year‑by‑year plots.
​

Low‑cost fix: add one robustness table with spline or interaction of log price with initial penetration deciles, or with an income–price interaction, to show no strong non‑linear pattern beyond what you already capture via time dummies.

Data quality / basket change

You now document the 1GB→5GB basket change and state explicitly that you use the harmonised 5GB series, validated with ITU staff; this is excellent and will reassure referees.
​

Make sure you keep that paragraph concise but prominent; it directly addresses potential “measurement break” criticism.

Interpretation of elasticity and COVID story
The empirical story is compelling: EaP elasticity around −0.6 vs EU −0.12 pre‑COVID, convergence to ~0 in 2020–24, and placebo plus year‑by‑year results showing the decline starts in 2015.
​

You carefully distinguish acute vs post‑acute COVID, show near‑zero elasticity in both, and argue for a secular transformation interpretation rather than pure COVID shock.
​

What I would still soften or sharpen:

Strength of “necessity good, price immune” language

“Immune to price changes” and “perfectly inelastic” appear in the framing; yet estimates are “near zero with wide confidence intervals” rather than statistically indistinguishable from exactly 0 everywhere.
​

Suggested adjustment: emphasise “near‑zero and statistically insignificant elasticities consistent with a necessity‑like behaviour” rather than absolute inelasticity.

Revenue interpretation and |ε| thresholds

Your revenue discussion around |ε|≈0.6 for EaP is fine, but be explicit that this is based on average elasticity and that operator‑specific pricing objectives could differ. 
​

Causal language about COVID

You already say “accelerated rather than initiated”; keep this carefully non‑causal (you show timing evidence but not a structural COVID instrument).
​

Writing, structure, and positioning
Positives:

The paper reads very clearly, with a strong narrative thread from abstract through Section 6; figures and tables are nicely integrated and labelled.
​

You explicitly tie each empirical section back to the research questions and to the policy angle, which matches what IEP editors want.
​

Targeted improvements before submission:

Trim some repetition

Sections 5.5–5.7 and Appendix B repeat the “declining since 2015 / robust across 24 specs” message several times. Condensing a bit will help keep the main text tight (IEP papers tend not to be extremely long).
​

Correct remaining literature over‑claims

You have already fixed the main mis‑matches (Goolsbee & Klenow now correctly cited as AER P&P on time‑use / Internet; Hausman et al. 2001 broadband demand; Nevo removed or re‑positioned). Still ensure the sentences about Brynjolfsson et al. (2003) don’t over‑state “declining elasticity” and instead emphasise consumer surplus / information‑goods value.
​

Tighten Section 4.5 (TWFE and heterogeneity)

This section is very good, but long relative to its payoff. Consider moving some of the references (Goodman‑Bacon, de Chaisemartin–D’Haultfœuille, Callaway–Sant’Anna) and the detailed argument to an appendix, keeping only a concise explanation in the main text.
​

Overall publication outlook and concrete steps
My editorial assessment, if this came as a new submission to IEP:

Desk‑reject risk: low. Topic and methods fit well; abstract, introduction, and data sections are strong and polished.

Referee expectations: one referee focused on telecom demand and ITU/World Bank data will look at price measurement, mobile substitution, and IV; another more econometric referee will scrutinise FE, DK SE, and TWFE issues. Your current draft anticipates both reasonably well.
​

Most likely outcome: “Revise and resubmit” with comments on (i) causal wording, (ii) IV strength / interpretation, and (iii) slight streamlining of the narrative. With a careful response, acceptance is realistically attainable.

Before you submit, I recommend you:

Slightly soften “perfectly inelastic / immune to price” language and keep “near‑zero, statistically insignificant elasticities consistent with necessity‑like behaviour.”
​

Re‑frame the IV section as robustness only; emphasise your main claims are based on FE+DK.
​

Prune a bit of repetition in Sections 5–6 and move some heavy econometric discussion to an appendix.
​

Would you like me to mark up specific paragraphs (e.g. abstract, key sentences in Sections 1.1, 2.1, 5.3, and 6.1) to make them “referee‑proof” in terms of wording and literature claims?

yes go ahead
Completed 1 step
Here are concrete, referee‑safe rewrites for the key places where wording is now a bit too strong. I’ll give each original fragment followed by a suggested replacement you can paste directly into the TeX file.

Abstract and highlights
Original first sentence of abstract

Has broadband become a necessity good immune to price changes?

Suggested

Has broadband become a necessity‑like good with very low sensitivity to price changes?

This keeps the question sharp but avoids implying literally zero elasticity.

Original highlight bullet

Broadband demand evolved from price-elastic (ε = −0.61 in EaP) to price-inelastic (ε ≈0) over 2010–2024

Suggested

Broadband demand evolved from clearly price‑elastic (ε = −0.61 in EaP) to near‑zero price elasticity (ε ≈ 0) over 2010–2024

Abstract middle sentences

Current:

By 2020–2024, both regions converged to near-zero elasticity, with price changes having no detectable effect on adoption. … These findings have immediate policy relevance: as broadband transitions from discretionary service to essential utility, policy emphasis must shift from affordability subsidies to universal infrastructure deployment.

Suggested:

By 2020–2024, both regions converged to near‑zero elasticity, with price changes having no statistically detectable effect on adoption. … These findings have immediate policy relevance: as broadband transitions from discretionary service to essential utility, policy emphasis should gradually shift from affordability subsidies toward universal infrastructure deployment.

That “statistically detectable” and “should gradually shift” language will read well to referees.

Introduction (Section 1)
Paragraph 1

Current:

… if demand becomes price-inelastic, subsidies and price regulations … lose their effectiveness.

Suggested:

… if demand becomes highly inelastic, subsidies and price regulations … become much less effective as tools for expanding adoption.

Later sentence (around line 20)

Current:

… transitioned to price-inelastic demand (ε ≈0) by 2020–2024, indicating broadband's evolution from discretionary service to essential necessity.

Suggested:

… transitioned to very low price sensitivity (ε ≈ 0) by 2020–2024, indicating broadband's evolution from discretionary service toward an essential necessity.

Literature review: elasticity becoming a necessity
Section 2.1 – Brynjolfsson et al. (2003) and Goolsbee & Klenow (2006)
Current text

A key theoretical insight is that demand elasticity varies with necessity status. As connectivity integrates into economic and social life, price sensitivity declines as services shift from discretionary to essential (Hausman et al., 2001). Brynjolfsson et al. (2003) show similar patterns for information goods, where high initial elasticity declines as network effects and complementarities increase adoption. … Evidence from broadband markets suggests that price sensitivity declines as services become embedded in daily economic life and as complementary technologies … raise the opportunity cost of disconnection (Goolsbee and Klenow, 2006).

Suggested rewrite

A key theoretical insight is that demand elasticity varies with necessity status. As connectivity integrates into economic and social life, price sensitivity can decline as services shift from discretionary to essential (Hausman et al., 2001). Brynjolfsson et al. (2003) document large consumer surplus from digital information goods, consistent with high valuations once such goods become integral to daily activities. Goolsbee and Klenow (2006) show that internet use generates substantial consumer surplus when measured via time use, highlighting that the perceived value of connectivity can be high even relative to observed expenditures. Our finding of declining elasticity from 2015–2024 fits naturally into this framework, suggesting that broadband has been moving toward necessity‑like status.

This keeps both citations but aligns what you claim with what the papers actually do.

Results and interpretation
Section 5.3 – COVID period
Key sentences

Current:

Both regions exhibit near-zero or slightly positive elasticity during COVID. … This pattern is consistent with broadband becoming an essential necessity during the pandemic.

Suggested:

Both regions exhibit near‑zero or slightly positive elasticity during COVID, with wide confidence intervals around zero. … This pattern is consistent with broadband behaving increasingly like an essential service during the pandemic.

Section 5.5–5.7 – structural transformation
Where you write:

… suggesting broadband followed a similar transition.
… structural transformation as broadband transitioned from luxury to necessity good.

I suggest softening to:

… suggesting broadband followed a similar transition toward necessity‑like behaviour.
… structural transformation as broadband moved from a discretionary service toward a necessity‑like good.

You can apply this small wording change wherever you use “luxury to necessity” in a strong way.

Policy and economic interpretation
Section 5.8 – revenue and “perfectly inelastic”
Current paragraph end

The shift toward near-zero elasticity during COVID dramatically changes the policy calculus. If demand becomes perfectly inelastic (ε = 0), price reductions have no effect on adoption. … The COVID period suggests broadband became an essential necessity where price ceased to be the binding constraint.

Suggested rewrite

The shift toward near‑zero elasticity during COVID dramatically changes the policy calculus. When demand becomes very inelastic, further price reductions have little effect on adoption. … The COVID period suggests broadband increasingly behaved like an essential necessity, with price no longer the primary binding constraint for most households.

This avoids the literal ε=0 statement while keeping the message.

Section 6.1 – opening of policy implications
Current:

However, the shift toward inelastic demand during COVID-19 fundamentally alters the policy toolkit. When price sensitivity approaches zero, further price reductions yield minimal adoption gains.

Suggested:

However, the shift toward very low price sensitivity during COVID‑19 substantially alters the policy toolkit. When elasticity is close to zero, further price reductions yield only limited adoption gains.

Section 6.5 – COVID vs secular trends
Where you write:

Instead, zero post-acute elasticity supports the secular transformation hypothesis: broadband has permanently crossed the threshold from discretionary to essential service.

Suggested:

Instead, the near‑zero post‑acute elasticities support the secular transformation hypothesis: broadband appears to have crossed the threshold from discretionary to effectively essential service in these markets.

Limitations and IV section
Section 6.6 – endogeneity paragraph
You can slightly re‑frame IV as robustness:

Replace:

We address this directly using 2SLS … The EaP 2SLS estimate … The EU 2SLS estimate is imprecise …

With something like:

To gauge the direction of potential bias, we also estimate 2SLS models using the mobile broadband price as an instrument for fixed broadband price (Table B.7). For EaP countries, the 2SLS coefficient is larger in magnitude than the FE estimate, consistent with OLS attenuation from measurement error. For the EU subsample, the instrument is weaker and standard errors are very large, so we treat these IV results as robustness checks rather than a basis for our main quantitative conclusions.

This tells referees you know IV is noisy and that your core claims rest on FE+DK.

Section 6.2 – time‑varying elasticity
A light softening:

Current:

The near-zero COVID-era elasticity implies that infrastructure availability and service quality – not affordability – now determine adoption.

Suggested:

The near‑zero COVID‑era elasticities suggest that infrastructure availability and service quality – rather than affordability alone – now play the central role in determining adoption.

If you want, you can paste your updated abstract and Section 2.1 after making these edits, and I can do a final “journal‑editor style” pass for tone, word count, and punchiness specifically for Information Economics and Policy.
# Information Economics and Policy — Submission Requirements

Source: https://www.sciencedirect.com/journal/information-economics-and-policy/publish/guide-for-authors

---

## Abstract
- **Word limit: 250 words maximum**
- Must be self-contained (can stand alone without article)
- State purpose, principal results, and major conclusions
- Avoid references; if essential, cite author(s) and year
- Avoid non-standard abbreviations; define any used at first mention

## Keywords
- **1–7 keywords** (written in English)
- Used for indexing and discoverability

## Highlights
- **3–5 bullet points**
- **Maximum 85 characters per bullet, including spaces**
- Capture novel results and new methods
- Use terms readers search for online
- Avoid jargon, acronyms, abbreviations
- Note: not part of editorial evaluation; submitted as a separate file at final stage

## Article Structure (required sections)
1. Title page (title, authors, affiliations, corresponding author, email)
2. Abstract (≤250 words)
3. Keywords (1–7)
4. Highlights (3–5 bullets, ≤85 chars each)
5. Main text (Introduction, Literature, Data, Methodology, Results, Discussion, Conclusion)
6. CRediT Authorship Contribution Statement
7. Declaration of competing interest
8. Data availability statement
9. Acknowledgments
10. References
11. Appendices (if applicable)

## Reference Style
- **Author-year (Harvard) format** — consistent with elsarticle-harv
- In-text: `(Author, Year)` or `Author (Year)`
- Reference list: alphabetical by first author surname
- Elsevier uses natbib with `\bibliographystyle{elsarticle-harv}`

## Mandatory Declarations

### CRediT Authorship Statement
- List each author's contributions using CRediT taxonomy
- Roles: Conceptualization, Methodology, Software, Validation, Formal analysis,
  Investigation, Resources, Data curation, Writing (original draft / review & editing),
  Visualization, Supervision, Project administration, Funding acquisition

### Declaration of Competing Interest
- Required for all submissions
- Standard phrase if none: "The authors declare that they have no known competing
  financial interests or personal relationships that could have appeared to influence
  the work reported in this paper."

### Data Availability Statement
- Required when data were used
- Specify where data can be accessed (URL, repository, or explanation)

### Generative AI Declaration (if applicable)
- **Required if AI tools were used in manuscript preparation**
- Must appear in a new section before the References
- Standard text: *"During the preparation of this work the author(s) used [TOOL] in order
  to [REASON]. After using this tool/service, the author(s) reviewed and edited the content
  as needed and take(s) full responsibility for the content of the publication."*
- If no AI was used, no section is needed

## Formatting
- **Document class**: elsarticle (Elsevier's LaTeX class)
- **Review mode**: `\documentclass[preprint,review,12pt,authoryear]{elsarticle}`
- **Final mode**: `\documentclass[final,3p,times,authoryear]{elsarticle}`
- **Line numbers**: Required for review submission (`\usepackage{lineno}`, `\linenumbers`)
- **Font**: Times (12pt for review)
- **Journal declaration**: `\journal{Information Economics and Policy}`

## Submission Platform
- **Editorial Manager** (Elsevier's submission system)
- File formats accepted: LaTeX (.tex) + PDF, or Word (.docx)
- Figures: separate high-resolution files (≥300 DPI for halftones, ≥600 DPI for line art)
- Highlights: submitted as separate Word document or plain text

## Peer Review
- **Single anonymized review** — reviewers are blinded, authors are not
- Author names and affiliations appear in the manuscript
- Minimum two independent reviewers

## Article Types
- Research papers
- Short contributions
- Survey articles

## Publishing Options
- Subscription (no APC for authors)
- Open Access: APC = USD 4,520 (before taxes)

## Journal Scope (IEP-specific)
Topics relevant to *Information Economics and Policy*:
- Economics of telecommunications, mass media, information industries
- Innovation and intellectual property economics
- Information's role in economic development
- Information technology in market operations
- Policy-oriented theoretical and empirical research

---

## Cross-Check: Paper vs. Requirements

### PASS ✓
| Requirement | Status |
|---|---|
| Journal set (`\journal{...}`) | `Information Economics and Policy` ✓ |
| Abstract word count | ~165 words (under 250 limit) ✓ |
| Keywords count | **FIXED → 7** ✓ |
| Reference style | elsarticle-harv, author-year ✓ |
| Line numbers | `\usepackage{lineno}` + `\linenumbers` enabled ✓ |
| CRediT statement | Present ✓ |
| Competing interests | Present ✓ |
| Data availability | Present ✓ |
| Acknowledgments | Present ✓ |
| All figures present | fig1–fig6 all exist ✓ |
| Bib keys (previous fixes) | hausman2001price→hausman2001private, katz1985network, goolsbee fixed ✓ |

### FIXED IN THIS SESSION ✓
| Issue | Fix Applied |
|---|---|
| **8 keywords** (max 7) | Removed "Two-way fixed effects" → 7 keywords |
| **Highlight 1: 104 chars** (max 85) | Rewritten to 81 chars |
| **Highlight 2: 86 chars** (max 85) | "price-sensitive" → "price-elastic" = 84 chars |
| **Highlight 4: 106 chars** (max 85) | Rewritten to 84 chars |
| **Highlight 5: 93 chars** (max 85) | Rewritten to 85 chars |

### ACTION REQUIRED BY AUTHORS
| Item | Action Needed |
|---|---|
| **Generative AI declaration** | Added: Claude (Anthropic) used for scripting and manuscript editing ✓ |
| **Cover letter** | Not mandatory but strongly recommended; should state novelty, target journal fit, and confirm the paper is not under review elsewhere |

### MINOR BIB TYPE ISSUES (non-critical)
| Entry | Issue |
|---|---|
| `angrist2009mostly` | @article but is a book (Princeton UP) |
| `hsiao2014analysis` | @article but is a book (Cambridge UP) |
| `katz2010impact` | @article but is an ITU report (consider @techreport) |

These will not affect editorial review but may affect reference list formatting.

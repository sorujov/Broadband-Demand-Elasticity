import sys, os, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import fitz

pdfs_dir = r"C:\Users\samir.orucov\Desktop\GIT Projects\Broadband-Demand-Elasticity\springer\pdfs"

def get_text(fname):
    doc = fitz.open(os.path.join(pdfs_dir, fname))
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# =====================
# 1. GOLDFARB - Does it mention COVID/pandemic? Get abstract + intro
# =====================
print("\n\n===== 1. GOLDFARB 2020 - first 3000 chars =====")
t = get_text("goldfarb2020digital.pdf")
print(t[:3000])
print("\n--- SEARCHING for accelerat/trend in first 10000 ---")
for kw in ["accelerat", "COVID", "pandemic", "trend"]:
    hits = [m.start() for m in re.finditer(kw, t[:10000], re.IGNORECASE)]
    print(f"  {kw}: {len(hits)} hits in first 10k chars")

# =====================
# 2. BRYNJOLFSSON - lockdown, necessity, convenience 
# =====================
print("\n\n===== 2. BRYNJOLFSSON - COVID remote work =====")
t2 = get_text("brynjolfsson2020covid.pdf")
print(t2[:4000])
print("\n--- All occurrences of necessity/convenience/lockdown ---")
for kw in ["necessity", "convenience", "lockdown", "shelter", "stay.at.home", "essential"]:
    hits = [m.start() for m in re.finditer(kw, t2, re.IGNORECASE)]
    if hits:
        pos = hits[0]
        print(f"\n  [{kw}] {len(hits)} hits, first: ...{t2[max(0,pos-100):pos+250]}...")
    else:
        print(f"  [{kw}] NOT FOUND")

# =====================
# 3. FAVALE - videoconferencing 
# =====================
print("\n\n===== 3. FAVALE - campus traffic =====")
t3 = get_text("favale2020campus.pdf")
print(t3[:3000])
print("\n--- Searching video/conferenc/zoom/teams ---")
for kw in ["video", "conferenc", "Zoom", "Teams", "traffic", "shift", "dramatic"]:
    hits = [m.start() for m in re.finditer(kw, t3, re.IGNORECASE)]
    if hits:
        pos = hits[0]
        print(f"\n  [{kw}] {len(hits)} hits: ...{t3[max(0,pos-100):pos+250]}...")
    else:
        print(f"  [{kw}] NOT FOUND")

# =====================
# 4. ROLLER - income/developing country adoption
# =====================
print("\n\n===== 4. ROLLER - telecom infrastructure =====")
t4 = get_text("roller2001telecommunications.pdf")
print(t4[:3000])
print("\n--- Income effects and developing countries ---")
for kw in ["income", "develop", "low.income", "afford", "dominate", "adoption", "poor", "tighter"]:
    hits = [m.start() for m in re.finditer(kw, t4, re.IGNORECASE)]
    if hits:
        pos = hits[0]
        print(f"\n  [{kw}] {len(hits)} hits: ...{t4[max(0,pos-100):pos+300]}...")
    else:
        print(f"  [{kw}] NOT FOUND")

# =====================
# 5. DAUVIN - log-log, elasticity, middle-income
# =====================
print("\n\n===== 5. DAUVIN - broadband diffusion EU =====")
t5 = get_text("dauvin2014estimating.pdf")
print(t5[:3000])
print("\n--- Elasticity and methodology ---")
for kw in ["log", "elasticit", "price", "middle", "linear", "logistic", "double.log"]:
    hits = [m.start() for m in re.finditer(kw, t5, re.IGNORECASE)]
    if hits:
        pos = hits[0]
        print(f"\n  [{kw}] {len(hits)} hits: ...{t5[max(0,pos-100):pos+300]}...")
    else:
        print(f"  [{kw}] NOT FOUND")

# =====================
# 6. KOUTROUMPIS - geography/country scope
# =====================
print("\n\n===== 6. KOUTROUMPIS - broadband impact =====")
t6 = get_text("koutroumpis2009impact.pdf")
print(t6[:4000])
print("\n--- Country scope and elasticity ---")
for kw in ["OECD", "countr", "Eastern", "elasticit", "demand", "develop", "panel", "sample"]:
    hits = [m.start() for m in re.finditer(kw, t6, re.IGNORECASE)]
    if hits:
        pos = hits[0]
        print(f"\n  [{kw}] {len(hits)} hits: ...{t6[max(0,pos-100):pos+300]}...")
    else:
        print(f"  [{kw}] NOT FOUND")

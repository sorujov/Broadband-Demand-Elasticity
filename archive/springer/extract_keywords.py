import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import re, os

pdfs_dir = r"C:\Users\samir.orucov\Desktop\GIT Projects\Broadband-Demand-Elasticity\springer\pdfs"

def extract_text(pdf_path):
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
            return text
        except Exception as e:
            return "ERROR: " + str(e)

tasks = [
    ("goldfarb2020digital.pdf", ["pandemic", "COVID", "accelerat", "trend", "digital economics"]),
    ("brynjolfsson2020covid.pdf", ["remote work", "lockdown", "necessity", "convenience", "shift", "rapid"]),
    ("favale2020campus.pdf", ["videoconferenc", "streaming", "traffic", "pattern", "shift", "e-learning"]),
    ("roller2001telecommunications.pdf", ["income effect", "lower-income", "developing", "affordab", "adoption", "dominate"]),
    ("dauvin2014estimating.pdf", ["log-log", "log transform", "elasticit", "middle-income", "EU", "NUTS", "specification"]),
    ("koutroumpis2009impact.pdf", ["Eastern Partnership", "Armenia", "Azerbaijan", "Belarus", "Georgia", "Moldova", "Ukraine", "elasticit", "demand"]),
]

for fname, keywords in tasks:
    path = os.path.join(pdfs_dir, fname)
    text = extract_text(path)
    print("\n" + "="*60)
    print("FILE: " + fname + " | Length: " + str(len(text)))
    print("="*60)
    for kw in keywords:
        hits = [m.start() for m in re.finditer(kw, text, re.IGNORECASE)]
        if hits:
            pos = hits[0]
            ctx = text[max(0, pos-150):pos+300]
            print("\n  [" + kw + "] (" + str(len(hits)) + " hits) -- first occurrence:")
            print("  ..." + ctx + "...")
        else:
            print("\n  [" + kw + "] NOT FOUND")

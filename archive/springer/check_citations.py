import sys, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open(r'extracted_texts.txt', encoding='utf-8', errors='replace') as f:
    text = f.read()

dav = text[317508:359243]
kout = text[359243:]

print("=== DAUVIN GNI context (first 3 hits) ===")
hits = [m.start() for m in re.finditer('GNI', dav)]
for h in hits[:3]:
    print(f'--- pos {h} ---')
    print(dav[max(0,h-100):h+250])
    print()

print("\n=== DAUVIN results section (Table 3 context) ===")
hits = [m.start() for m in re.finditer('Table 3', dav, re.IGNORECASE)]
for h in hits:
    print(dav[max(0,h-100):h+800])
    print()

print("\n\n=== KOUTROUMPIS first 4000 ===")
print(kout[300:4000])

print("\n=== KOUTROUMPIS country search ===")
for kw in ['OECD', 'countr', 'Eastern', 'elasticit', 'demand', 'panel', 'sample', 
           'Armenia', 'Azerbaijan', 'Belarus', 'Georgia', 'Moldova', 'Ukraine',
           'CIS', 'transition', 'develop']:
    hits = [m.start() for m in re.finditer(kw, kout, re.IGNORECASE)]
    if hits:
        pos = hits[0]
        print(f'\n[{kw}] {len(hits)} hits: ...{kout[max(0,pos-100):pos+300]}...')
    else:
        print(f'\n[{kw}] NOT FOUND')

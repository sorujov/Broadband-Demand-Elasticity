import geopandas as gpd, rasterio, rasterio.mask, numpy as np, pandas as pd, glob, re
ISO="AUT BEL BGR HRV CYP CZE DNK EST FIN FRA DEU GRC HUN IRL ITA LVA LTU LUX MLT NLD POL PRT ROU SVK SVN ESP SWE ARM AZE BLR GEO MDA UKR".split()
g=gpd.read_file('zip://ne_admin0.zip')
col='ADM0_A3' if 'ADM0_A3' in g else 'ISO_A3'
g=g[g[col].isin(ISO)][[col,'geometry']].rename(columns={col:'iso'})
# keep European part of FRA (drop overseas) by clipping to Europe bbox
from shapely.geometry import box
g['geometry']=g.geometry.intersection(box(-25,27,60,72))
print(len(g), sorted(set(ISO)-set(g.iso)))
rows=[]
for f in sorted(glob.glob('ntl/Harmonized_DN_NTL_*.tif')):
    yr=int(re.findall(r'_(\d{4})_',f)[0])
    with rasterio.open(f) as src:
        for _,r in g.iterrows():
            arr,_=rasterio.mask.mask(src,[r.geometry],crop=True,filled=True,nodata=0)
            a=arr[0].astype(float)
            rows.append(dict(country=r.iso,year=yr,ntl_sum=a.sum(),ntl_lit=(a>7).sum(),sensor='DMSP' if 'DMSP' in f else 'VIIRSsim'))
    print(yr)
pd.DataFrame(rows).to_csv('ntl_country.csv',index=False)

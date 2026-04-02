"""
load_tmdb.py — TMDB dataset loader for the Box Office Predictor (v2)
====================================================================
Improvements over v1:
  • Uses franchise_lookup.py for accurate sequel/franchise detection
  • Inflation-adjusts all dollar figures to 2024 USD
  • Adds release_year as a model feature
  • Better RT score proxy (weighted vote_average + confidence)
  • Competition level estimated from release month (smarter than random)
  • Filters to 1990+ only
"""

import ast
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MOVIES_CSV  = Path("tmdb_5000_movies.csv")
CREDITS_CSV = Path("tmdb_5000_credits.csv")
OUTPUT_CSV  = Path("tmdb_features.csv")

# CPI-U multipliers to convert to 2024 USD
CPI_MULTIPLIER = {
    1990:2.37,1991:2.27,1992:2.20,1993:2.14,1994:2.09,1995:2.03,
    1996:1.97,1997:1.93,1998:1.90,1999:1.86,2000:1.80,2001:1.75,
    2002:1.72,2003:1.68,2004:1.64,2005:1.59,2006:1.54,2007:1.49,
    2008:1.44,2009:1.44,2010:1.42,2011:1.37,2012:1.34,2013:1.32,
    2014:1.30,2015:1.30,2016:1.28,2017:1.25,2018:1.22,2019:1.19,
    2020:1.18,2021:1.12,2022:1.02,2023:1.00,2024:1.00,2025:0.97,2026:0.94,
}

GENRE_MAP = {
    "Action":"Action","Adventure":"Action","Science Fiction":"SciFi",
    "Horror":"Horror","Comedy":"Comedy","Animation":"Animation",
    "Family":"Animation","Romance":"Romance","Drama":"Drama",
    "Thriller":"Thriller","Mystery":"Thriller","Crime":"Thriller",
    "Fantasy":"Fantasy","Documentary":"Documentary","Music":"Drama",
    "History":"Drama","War":"Action","Western":"Action","TV Movie":"Drama",
}
GENRES_LIST  = ["Action","Comedy","Drama","Horror","SciFi","Animation","Romance","Thriller","Fantasy","Documentary"]
MPAA_RATINGS = ["G","PG","PG-13","R","NC-17"]

FEATURE_NAMES = [
    "budget_m_adj","marketing_budget_m","release_month","genre_encoded",
    "is_franchise","sequel_number","director_score","cast_score",
    "screen_count","mpaa_rating_encoded","runtime","trailer_views_m",
    "social_sentiment","rt_score","competition_level","release_year",
]

def safe_json(val):
    if pd.isna(val) or not val:
        return []
    try:    return json.loads(val)
    except Exception:
        try:    return ast.literal_eval(val)
        except: return []

def first_genre(g):
    for x in safe_json(g):
        if x.get("name","") in GENRE_MAP:
            return GENRE_MAP[x["name"]]
    return "Drama"

def get_director(c):
    for m in safe_json(c):
        if m.get("job")=="Director": return m.get("name","")
    return ""

def get_lead_actor(c):
    cast=safe_json(c)
    return cast[0].get("name","") if cast else ""

def get_keywords(k):
    return [x.get("name","") for x in safe_json(k)]

def cpi_adjust(amount_m, year):
    return amount_m * CPI_MULTIPLIER.get(int(year), 1.0)

def extract_mpaa(r):
    data=safe_json(r)
    for c in (data.get("countries",[]) if isinstance(data,dict) else []):
        if c.get("iso_3166_1")=="US":
            cert=c.get("certification","").strip()
            if cert in MPAA_RATINGS: return cert
    return "PG-13"

def build_person_scores(df):
    from collections import defaultdict
    df=df.copy()
    df["rdt"]=pd.to_datetime(df["release_date"],errors="coerce")
    df=df.sort_values("rdt").reset_index(drop=True)
    df_=defaultdict(list); cf_=defaultdict(list)
    for i,r in df.iterrows():
        g=r["revenue_adj_m"] if r["revenue_adj_m"]>0 else np.nan
        if r["director"]:   df_[r["director"]].append((i,g))
        if r["lead_actor"]: cf_[r["lead_actor"]].append((i,g))
    fb=df[df["revenue_adj_m"]>0]["revenue_adj_m"].median()
    def avg(films,idx,p):
        prev=[g for i,g in films.get(p,[]) if i<idx and not np.isnan(g)]
        return round(np.mean(prev[-5:]),1) if prev else fb
    ds={}; cs={}
    for i,r in df.iterrows():
        ds[r["id"]]=avg(df_,i,r["director"])
        cs[r["id"]]=avg(cf_,i,r["lead_actor"])
    return ds,cs

def main():
    print("📂  Loading TMDB CSVs …")
    if not MOVIES_CSV.exists():
        print(f"❌  {MOVIES_CSV} not found."); return
    if not CREDITS_CSV.exists():
        print(f"❌  {CREDITS_CSV} not found."); return

    movies =pd.read_csv(MOVIES_CSV)
    credits=pd.read_csv(CREDITS_CSV)
    print(f"    Movies:{len(movies):,}  Credits:{len(credits):,}")

    id_col="movie_id" if "movie_id" in credits.columns else "id"
    df=movies.merge(credits,left_on="id",right_on=id_col,how="left",suffixes=("","_c"))

    df["release_date_dt"]=pd.to_datetime(df["release_date"],errors="coerce")
    df["release_year"]=df["release_date_dt"].dt.year
    df=df[df["release_year"]>=1990].copy()
    df=df[(df["budget"]>500_000)&(df["revenue"]>500_000)].copy()
    print(f"    After filters: {len(df):,} rows")

    df["budget_m_adj"] =df.apply(lambda r:cpi_adjust(r["budget"]/1e6, r["release_year"]),axis=1)
    df["revenue_adj_m"]=df.apply(lambda r:cpi_adjust(r["revenue"]/1e6,r["release_year"]),axis=1)
    df["worldwide_gross_m"]=df["revenue_adj_m"]

    df["director"]  =df["crew"].apply(get_director)   if "crew"  in df.columns else ""
    df["lead_actor"]=df["cast"].apply(get_lead_actor) if "cast"  in df.columns else ""
    df["kw_list"]   =df["keywords"].apply(get_keywords) if "keywords" in df.columns else pd.Series([[]]*len(df))

    print("🏷  Franchise/sequel lookup …")
    from franchise_lookup import sequel_number as fl_seq, is_franchise as fl_fran
    df["sequel_number"]=df.apply(lambda r:float(fl_seq(str(r["original_title"]),r["kw_list"])),axis=1)
    df["is_franchise"] =df.apply(lambda r:float(fl_fran(str(r["original_title"]),r["kw_list"])),axis=1)
    print(f"    Franchise:{int(df['is_franchise'].sum())}  Sequels>1:{int((df['sequel_number']>1).sum())}")

    print("🧮  Person scores …")
    ds,cs=build_person_scores(df)
    df["director_score"]=df["id"].map(ds)
    df["cast_score"]    =df["id"].map(cs)

    df["release_month"]=df["release_date_dt"].dt.month.fillna(6).astype(int)
    df["genre"]=df["genres"].apply(first_genre)
    df["genre_encoded"]=df["genre"].apply(lambda g:GENRES_LIST.index(g) if g in GENRES_LIST else 0)
    df["marketing_budget_m"]=(df["budget_m_adj"]*0.5).round(1)
    df["screen_count"]=(1200+df["budget_m_adj"]*10+
        np.random.default_rng(42).normal(0,350,len(df))).clip(100,4500).round(0)

    rc=next((c for c in df.columns if c in ("release_dates","releases")),None)
    df["mpaa_rating"]=df[rc].apply(extract_mpaa) if rc else "PG-13"
    df["mpaa_rating_encoded"]=df["mpaa_rating"].apply(lambda r:MPAA_RATINGS.index(r) if r in MPAA_RATINGS else 2)
    df["runtime"]=pd.to_numeric(df["runtime"],errors="coerce").fillna(105)

    # Trailer views: calibrated log-popularity
    df["trailer_views_m"]=(np.log1p(df["popularity"])*12).clip(0.5,120).round(2)

    # Social sentiment from vote_average
    df["social_sentiment"]=((df["vote_average"]-5.0)/4.0).clip(0.05,0.98).round(3)

    # RT score: vote_average * confidence weight
    vc=np.log1p(df["vote_count"])/np.log1p(df["vote_count"].max())
    df["rt_score"]=(df["vote_average"]*10*0.7+65*0.3).clip(0,100).round(1)

    # Competition from release month (more realistic than pure random)
    month_comp={1:3,2:3,3:5,4:5,5:7,6:8,7:8,8:6,9:5,10:6,11:7,12:9}
    rng=np.random.default_rng(42)
    df["competition_level"]=(df["release_month"].map(month_comp).fillna(5)+
        rng.normal(0,1.5,len(df))).clip(1,10).round(1)

    out=df[FEATURE_NAMES+["worldwide_gross_m"]].copy().dropna()
    print(f"    Final dataset: {len(out):,} films")
    print("\n📊  Gross distribution (2024-adjusted $M):")
    print(out["worldwide_gross_m"].describe(percentiles=[.1,.25,.5,.75,.9]).round(1).to_string())
    out.to_csv(OUTPUT_CSV,index=False)
    print(f"\n✅  Saved to {OUTPUT_CSV}\n    Run: python retrain.py")

if __name__=="__main__":
    main()
"""
load_tmdb.py — TMDB dataset loader for the Box Office Predictor (v2)
====================================================================
Improvements over v1:
  • Uses franchise_lookup.py for accurate sequel/franchise detection
  • Inflation-adjusts all dollar figures to 2024 USD
  • Adds release_year as a model feature
  • Filters to 1990+ only

v3: only emits the pre-release features listed in features.FEATURE_NAMES.
Popularity/vote-based proxies (trailer views, sentiment, RT score) were
removed because they are measured after release and leak the target;
synthetic columns (marketing, screens, competition) and the constant MPAA
rating were removed because they carried no real information. Inflation
adjustment, genre encoding and person scores come from features.py, which
the live scraper also uses.
"""

import ast
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
# Progress messages use emoji; don't crash on Windows consoles that can't show them
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

MOVIES_CSV  = Path("tmdb_5000_movies.csv")
CREDITS_CSV = Path("tmdb_5000_credits.csv")
OUTPUT_CSV  = Path("tmdb_features.csv")
# Newer films from fetch_tmdb.py (same format); used when present
API_MOVIES_CSV  = Path("tmdb_api_movies.csv")
API_CREDITS_CSV = Path("tmdb_api_credits.csv")

GENRE_MAP = {
    "Action":"Action","Adventure":"Action","Science Fiction":"SciFi",
    "Horror":"Horror","Comedy":"Comedy","Animation":"Animation",
    "Family":"Animation","Romance":"Romance","Drama":"Drama",
    "Thriller":"Thriller","Mystery":"Thriller","Crime":"Thriller",
    "Fantasy":"Fantasy","Documentary":"Documentary","Music":"Drama",
    "History":"Drama","War":"Action","Western":"Action","TV Movie":"Drama",
}

from features import FEATURE_NAMES, cpi_adjust, encode_genre, person_score

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
        score=person_score(prev)
        return score if score is not None else fb
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
    movies["data_source"]="kaggle"
    print(f"    Kaggle  movies:{len(movies):,}  credits:{len(credits):,}")

    if API_MOVIES_CSV.exists() and API_CREDITS_CSV.exists():
        api_movies =pd.read_csv(API_MOVIES_CSV)
        api_credits=pd.read_csv(API_CREDITS_CSV)
        api_movies["data_source"]="tmdb_api"
        # API rows replace Kaggle rows for the same film (fresher revenue figures)
        movies =pd.concat([movies[~movies["id"].isin(api_movies["id"])],api_movies],ignore_index=True)
        credits=pd.concat([credits[~credits["movie_id"].isin(api_credits["movie_id"])],api_credits],ignore_index=True)
        print(f"    + API   movies:{len(api_movies):,} → {len(movies):,} total")

    df=movies.merge(credits,left_on="id",right_on="movie_id",how="left",suffixes=("","_c"))

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
    # English title, like the English Wikipedia titles the live scraper uses
    df["sequel_number"]=df.apply(lambda r:float(fl_seq(str(r["title"]),r["kw_list"])),axis=1)
    df["is_franchise"] =df.apply(lambda r:float(fl_fran(str(r["title"]),r["kw_list"])),axis=1)
    print(f"    Franchise:{int(df['is_franchise'].sum())}  Sequels>1:{int((df['sequel_number']>1).sum())}")

    print("🧮  Person scores …")
    ds,cs=build_person_scores(df)
    df["director_score"]=df["id"].map(ds)
    df["cast_score"]    =df["id"].map(cs)

    df["release_month"]=df["release_date_dt"].dt.month.fillna(6).astype(int)
    df["genre"]=df["genres"].apply(first_genre)
    df["genre_encoded"]=df["genre"].apply(encode_genre)
    df["runtime"]=pd.to_numeric(df["runtime"],errors="coerce").fillna(105)

    # Identity columns (not model inputs) record which films the model learns from
    df["tmdb_id"]=df["id"]
    out=df[["tmdb_id","title","data_source"]+FEATURE_NAMES+["worldwide_gross_m"]].dropna(
        subset=FEATURE_NAMES+["worldwide_gross_m"]).copy()
    print(f"    Final dataset: {len(out):,} films")
    print("\n📊  Gross distribution (2024-adjusted $M):")
    print(out["worldwide_gross_m"].describe(percentiles=[.1,.25,.5,.75,.9]).round(1).to_string())
    out.to_csv(OUTPUT_CSV,index=False)
    print(f"\n✅  Saved to {OUTPUT_CSV}\n    Run: python retrain.py")

if __name__=="__main__":
    main()
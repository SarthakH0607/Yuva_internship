#!/usr/bin/env python3
"""
Week 3 - cleaning and preprocessing pipeline for the skincare mismatch project.

Input : Sephora product table (product_info.csv) and one review file (reviews_*.csv)
Output: cleaned reviews and products (csv.gz), a step log and metrics (json)

Run   : python3 -I clean_pipeline.py <product_info.csv> <reviews.csv> <output_dir>

Design rules
  1. Raw files are never edited. Every change is made in code, in order, and logged.
  2. Rows are removed only when they are provably duplicated. Everything else is
     flagged, so the analyst can choose how to treat it.
  3. Missing outcomes are never imputed (that would invent the thing being measured).
  4. No feature is computed from the whole dataset before the time split (no leakage).
"""
import ast
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

prod_path, rev_path, out_dir = (Path(a) for a in sys.argv[1:4])
out_dir.mkdir(parents=True, exist_ok=True)

LOG, METRICS = [], {}


def log_step(name, before, after, flagged=None, note=""):
    LOG.append({
        "step": name,
        "rows_before": int(before),
        "rows_after": int(after),
        "flagged_or_changed": None if flagged is None else int(flagged),
        "note": note,
    })


def iqr_fence(x, k=1.5):
    q1, q3 = np.nanpercentile(x, [25, 75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


# =============================================================================
# PART A - PRODUCT TABLE
# =============================================================================
p = pd.read_csv(prod_path)
METRICS["products_raw"] = {
    "rows": len(p), "cols": p.shape[1],
    "missing_pct": (p.isna().mean() * 100).round(1).to_dict(),
}

# A1. drop columns that are mostly empty or not needed
sparse = [c for c in p.columns if p[c].isna().mean() > 0.80]
unneeded = ["child_count", "child_max_price", "child_min_price"]
p = p.drop(columns=sparse + unneeded)
METRICS["products_dropped_cols"] = {"sparse_over_80pct_missing": sparse, "not_needed": unneeded}

# A2. rename scrape-time popularity fields so nobody uses them as features by accident
p = p.rename(columns={
    "rating": "site_rating_at_scrape",
    "reviews": "site_review_count",
    "loves_count": "site_loves_at_scrape",
})
p["brand_name"] = p["brand_name"].str.strip()
p["product_name"] = p["product_name"].str.strip()

# A3. size -> millilitre equivalent, then price per 100 mL
OZ_ML = 29.5735


def size_ml(s):
    if not isinstance(s, str) or " x " in s.lower():
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:fl\.?\s*)?(?:mL|ml|ML)\b", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:fl\.?\s*)?oz", s, re.I)
    if m:
        return float(m.group(1)) * OZ_ML
    return np.nan


p["size_ml_equiv"] = p["size"].apply(size_ml)
p["price_per_100ml"] = p["price_usd"] / p["size_ml_equiv"] * 100
METRICS["size_parsed_pct"] = round(float(p["size_ml_equiv"].notna().mean() * 100), 1)
METRICS["size_missing_raw_pct"] = round(float(p["size"].isna().mean() * 100), 1)


# A4. highlights -> claim flags and the skin types a product says it is best for
def parse_list(s):
    if not isinstance(s, str):
        return []
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except Exception:
        return []


hl_txt = p["highlights"].apply(parse_list).apply(lambda L: " | ".join(L).lower())
p["claim_vegan"] = hl_txt.str.contains("vegan", regex=False).astype(int)
p["claim_cruelty_free"] = hl_txt.str.contains("cruelty-free", regex=False).astype(int)
p["claim_clean"] = hl_txt.str.contains("clean at sephora", regex=False).astype(int)
p["claim_fragrance_free"] = hl_txt.str.contains("fragrance free", regex=False).astype(int)
p["claim_paraben_free"] = hl_txt.str.contains("without parabens", regex=False).astype(int)

SKIN_MAP = {"dry": "dry", "oily": "oily", "combo": "combination", "combination": "combination", "normal": "normal"}


def best_for(t):
    if "all skin types" in t:
        return "combination|dry|normal|oily"
    m = re.search(r"best for ([a-z, ]+?) skin", t)
    if not m:
        return np.nan
    toks = sorted({SKIN_MAP[x.strip()] for x in m.group(1).split(",") if x.strip() in SKIN_MAP})
    return "|".join(toks) if toks else np.nan


p["best_for_skin"] = hl_txt.apply(best_for)

# A5. ingredients: parse the stringified list, drop kit headings, split, normalise
def split_top_level(s):
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def parse_ingredients(raw):
    """returns (tokens, is_kit) or (None, False) when there is no list."""
    if not isinstance(raw, str):
        return None, False
    try:
        items = ast.literal_eval(raw)
    except Exception:
        return None, False
    is_kit = False
    toks = []
    for el in items:
        el = str(el).strip()
        if el.endswith(":") and len(el) < 90:  # sub-product heading inside a set
            is_kit = True
            continue
        el = re.sub(r"(?i)\b(active|inactive|other)?\s*ingredients?\s*:", ",", el)
        el = re.sub(r"\d+(?:\.\d+)?\s*%", "", el)
        el = el.replace("[", "").replace("]", "").replace("+/-", ",")
        for t in split_top_level(el):
            t = re.sub(r"\s+", " ", t).strip(" .*†").lower()
            if t and len(t) <= 80 and not t.startswith("may contain"):
                toks.append(t)
    seen, uniq = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return (uniq or None), is_kit


parsed = p["ingredients"].apply(parse_ingredients)
p["ing_tokens"] = parsed.apply(lambda x: x[0])
p["is_kit"] = parsed.apply(lambda x: int(x[1]))
p["n_ingredients"] = p["ing_tokens"].apply(lambda x: len(x) if isinstance(x, list) else np.nan)
p["ingredients_clean"] = p["ing_tokens"].apply(lambda x: "|".join(x) if isinstance(x, list) else np.nan)
p["ing_suspect_parse"] = ((p["n_ingredients"] < 3) | (p["n_ingredients"] > 100)).astype(int)

GROUPS = {
    "ing_fragrance": r"\b(parfum|fragrance|aroma)\b",
    "ing_fragrance_allergen": r"\b(limonene|linalool|citronellol|geraniol|eugenol|coumarin|citral|benzyl salicylate|benzyl benzoate|hexyl cinnamal|farnesol|cinnamal|isoeugenol|amyl cinnamal|hydroxycitronellal|isomethyl ionone)\b",
    "ing_drying_alcohol": r"alcohol denat|sd alcohol|\bethanol\b|isopropyl alcohol",
    "ing_retinoid": r"\bretin(?:ol|al|yl|oic|oid)\w*|adapalene|tretinoin",
    "ing_acid_exfoliant": r"glycolic|lactic acid|mandelic|salicylic|gluconolactone|lactobionic",
    "ing_humectant": r"glycerin|hyaluron|panthenol|\burea\b|butylene glycol|propanediol|sodium pca|betaine",
    "ing_barrier_lipid": r"ceramide|cholesterol|squalane|squalene",
    "ing_oil_butter": r"\boil\b|\bbutter\b",
    "ing_uv_filter": r"avobenzone|butyl methoxydibenzoylmethane|homosalate|octocrylene|octisalate|ethylhexyl salicylate|ethylhexyl methoxycinnamate|zinc oxide|titanium dioxide|oxybenzone|tinosorb",
    "ing_niacinamide": r"niacinamide",
    "ing_vitamin_c": r"ascorb",
}
ing_txt = p["ingredients_clean"].fillna("")
has_ing = p["ingredients_clean"].notna()
for col, pat in GROUPS.items():
    p[col] = np.where(has_ing, ing_txt.str.contains(pat, regex=True).astype(float), np.nan)

# A6. price tiers (cut points from skincare products only), outlier flags
sk_mask = p["primary_category"].eq("Skincare")
q1, q2, q3 = p.loc[sk_mask, "price_usd"].quantile([0.25, 0.5, 0.75])
p["price_tier"] = pd.cut(p["price_usd"], [-np.inf, q1, q2, q3, np.inf],
                         labels=["budget", "mid", "premium", "luxury"]).astype(str)
lo, hi = iqr_fence(np.log1p(p["price_usd"]))
p["price_outlier"] = (np.log1p(p["price_usd"]) > hi).astype(int)
lo2, hi2 = iqr_fence(np.log1p(p["price_per_100ml"].dropna()))
p["price_per_ml_outlier"] = (np.log1p(p["price_per_100ml"]) > hi2).astype(int)

METRICS["products_clean"] = {
    "rows": len(p),
    "skincare_products": int(sk_mask.sum()),
    "skincare_with_ingredients": int((sk_mask & has_ing).sum()),
    "kits_flagged": int(p["is_kit"].sum()),
    "ingredient_parse_suspect": int(p["ing_suspect_parse"].sum()),
    "ingredient_median_tokens": float(p["n_ingredients"].median()),
    "price_quartiles_skincare": [float(q1), float(q2), float(q3)],
    "price_outliers_flagged": int(p["price_outlier"].sum()),
    "price_outlier_fence_usd": round(float(np.expm1(hi)), 1),
    "price_per_100ml_outliers": int(p["price_per_ml_outlier"].sum()),
    "best_for_skin_available": int(p["best_for_skin"].notna().sum()),
    "ingredient_group_prevalence_pct_skincare_nonkit": {
        c: round(float(p.loc[sk_mask & has_ing & (p["is_kit"] == 0), c].mean() * 100), 1) for c in GROUPS
    },
    "claims_pct": {c: round(float(p[c].mean() * 100), 1) for c in
                   ["claim_vegan", "claim_cruelty_free", "claim_clean", "claim_fragrance_free", "claim_paraben_free"]},
}
ex_row = p.loc[has_ing & (p["is_kit"] == 0) & (p["n_ingredients"].between(8, 20))].iloc[0]
METRICS["ingredient_example"] = {
    "product": ex_row["product_name"],
    "raw": str(ex_row["ingredients"])[:230],
    "tokens": ex_row["ing_tokens"][:8],
}

# =============================================================================
# PART B - REVIEW TABLE
# =============================================================================
r = pd.read_csv(rev_path, low_memory=False)
r = r.drop(columns=[c for c in r.columns if c.startswith("Unnamed")])
n_raw = len(r)
METRICS["reviews_raw"] = {
    "rows": n_raw, "cols": r.shape[1],
    "distinct_products": int(r["product_id"].nunique()),
    "distinct_authors": int(r["author_id"].nunique()),
    "missing_pct": (r.isna().mean() * 100).round(1).to_dict(),
}

# B1. types
r["author_id"] = r["author_id"].astype(str)
r["submission_time"] = pd.to_datetime(r["submission_time"], errors="coerce")
r["is_recommended"] = r["is_recommended"].astype("Int8")
log_step("Fix data types (dates, ids, recommend flag)", n_raw, len(r), int(r["submission_time"].isna().sum()),
         "dates that could not be parsed")

# B2. exact duplicates only
key = ["author_id", "product_id", "submission_time", "review_text"]
near_dups = int(r.duplicated(["author_id", "product_id"], keep=False).sum())
before = len(r)
r = r.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
log_step("Remove exact duplicate reviews", before, len(r), before - len(r),
         f"{near_dups} rows share author+product (kept: may be genuine re-reviews)")

# B3. text cleaning (raw text is kept in review_text)
RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_REP = re.compile(r"(.)\1{3,}")


def clean_text(s):
    if not isinstance(s, str):
        return np.nan
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = RE_HTML.sub(" ", s)
    s = RE_URL.sub(" ", s)
    s = RE_REP.sub(r"\1\1\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else np.nan


raw_txt = r["review_text"].fillna("")
METRICS["text_issues_raw"] = {
    "missing_text": int(r["review_text"].isna().sum()),
    "empty_or_blank_text": int((raw_txt.str.strip() == "").sum() - r["review_text"].isna().sum()),
    "html_tags": int(raw_txt.apply(lambda x: bool(RE_HTML.search(x))).sum()),
    "urls": int(raw_txt.apply(lambda x: bool(RE_URL.search(x))).sum()),
    "stretched_words_4plus_repeats": int(raw_txt.apply(lambda x: bool(RE_REP.search(x))).sum()),
}
r["review_text_clean"] = r["review_text"].apply(clean_text)
ex = r.loc[raw_txt.apply(lambda x: bool(RE_REP.search(x)) and len(x) < 160)].head(1)
if len(ex):
    METRICS["text_example"] = {"raw": ex["review_text"].iloc[0], "clean": ex["review_text_clean"].iloc[0]}
log_step("Clean review text (unicode, html, urls, stretched words, blanks)", len(r), len(r),
         int((r["review_text_clean"] != r["review_text"]).sum()), "rows whose text changed; raw text kept")

r["n_chars"] = r["review_text_clean"].fillna("").str.len()
r["n_words"] = r["review_text_clean"].fillna("").str.split().str.len()
r["has_text"] = r["review_text_clean"].notna().astype(int)
r["short_text"] = ((r["n_chars"] > 0) & (r["n_chars"] < 15)).astype(int)
r["non_ascii_suspect"] = r["review_text_clean"].fillna("").apply(
    lambda x: int(len(x) > 0 and sum(ord(c) > 127 for c in x) / len(x) > 0.30))

# B4. categorical standardisation + explicit 'not stated'
r["eye_color"] = r["eye_color"].str.strip().str.lower().replace({"grey": "gray"})
r["hair_color"] = r["hair_color"].str.strip().str.lower().replace({"brunette": "brown"})
TONE_MAP = {"porcelain": "porcelain", "fair": "fair", "fairLight": "fair_light", "light": "light",
            "lightMedium": "light_medium", "medium": "medium", "mediumTan": "medium_tan", "tan": "tan",
            "olive": "olive", "deep": "deep", "rich": "rich", "dark": "dark", "ebony": "ebony"}
TONE_GROUP = {"porcelain": "light", "fair": "light", "fair_light": "light", "light": "light",
              "light_medium": "medium", "medium": "medium", "medium_tan": "medium",
              "tan": "deep", "deep": "deep", "rich": "deep", "dark": "deep", "ebony": "deep"}
n_not_sure = int((r["skin_tone"] == "notSureST").sum())
r["skin_tone"] = r["skin_tone"].map(TONE_MAP)  # 'notSureST' and anything unknown -> missing
r["skin_tone_group"] = r["skin_tone"].map(TONE_GROUP)  # olive is an undertone, left ungrouped
for c in ["skin_type", "skin_tone", "skin_tone_group", "eye_color", "hair_color"]:
    r[c + "_stated"] = r[c].notna().astype(int)
    r[c] = r[c].fillna("not_stated")
log_step("Standardise categories and mark 'not stated'", len(r), len(r), n_not_sure,
         "merged grey/gray, brunette/brown; 'notSureST' treated as not stated")

# B5. helpfulness: missing only when a review has no votes -> structural, not random
t = r["total_feedback_count"]
pos = r["total_pos_feedback_count"]
r["has_votes"] = (t > 0).astype(int)
n_safe = t.where(t > 0, 1).astype(float)
phat = pos / n_safe
z = 1.96
wilson = (phat + z * z / (2 * n_safe) - z * np.sqrt((phat * (1 - phat) + z * z / (4 * n_safe)) / n_safe)) / (1 + z * z / n_safe)
r["helpful_wilson"] = wilson.clip(0, 1).where(t > 0)
r = r.rename(columns={"helpfulness": "helpful_ratio"})

# B6. outliers in vote counts and text length: winsorise / flag, never delete
cap = float(t.quantile(0.99))
r["feedback_w99"] = t.clip(upper=cap)
r["log_feedback"] = np.log1p(t)
lo_len, hi_len = iqr_fence(np.log1p(r.loc[r["n_chars"] > 0, "n_chars"]))
r["length_outlier"] = ((np.log1p(r["n_chars"]) > hi_len) & (r["n_chars"] > 0)).astype(int)

# B7. consistency flags
rec = r["is_recommended"]
rec_f = rec.fillna(-1).astype(int)
r["rating_rec_conflict"] = (((r["rating"] <= 2) & (rec_f == 1)) | ((r["rating"] >= 4) & (rec_f == 0))).astype(int)

REACT = re.compile(
    r"\b(break(?:ing|s)?[ -]?(?:me |my face |my skin )?out|broke (?:me |my face |my skin )?out|breakouts?|rash(?:es)?|"
    r"burn(?:ed|ing|s|t)?|irritat\w*|allerg\w*|itch\w*|sting(?:s|ing)?|stung|hives|redness|clogg?\w*|purg\w*|"
    r"swollen|peeling|flak\w*)\b", re.I)
NEG = re.compile(
    r"\b(no|not|never|without|nothing|zero|free of|\w+n't|didnt|doesnt|dont|wont)\b[^.!?]{0,30}$", re.I)


def reaction_flag(txt):
    if not isinstance(txt, str):
        return 0
    for m in REACT.finditer(txt):
        if not NEG.search(txt[max(0, m.start() - 35):m.start()]):
            return 1
    return 0


r["reaction_mention"] = r["review_text_clean"].apply(reaction_flag)
r["star_text_conflict"] = ((r["rating"] >= 4) & (r["reaction_mention"] == 1)).astype(int)
INCENT = re.compile(r"received (?:this |the )?(?:product |item )?(?:for free|free of charge|complimentary)|"
                    r"in exchange for|incentivi[sz]ed|#ad\b|free sample|sent (?:to )?me (?:this|by)", re.I)
r["incentivized_mention"] = r["review_text_clean"].fillna("").apply(lambda x: int(bool(INCENT.search(x))))

# B8. duplicated text: same author posting the same long text on several product pages
long_txt = r["review_text_clean"].where(r["n_chars"] >= 40)
r["dup_text_same_author"] = long_txt.notna() & r.duplicated(["author_id", "review_text_clean"], keep=False)
r["dup_text_same_author"] = r["dup_text_same_author"].astype(int)
r["dup_text_other_author"] = (long_txt.notna() & r.duplicated("review_text_clean", keep=False)).astype(int) - r["dup_text_same_author"]
r["dup_text_other_author"] = r["dup_text_other_author"].clip(lower=0)
r["text_first_occurrence"] = (~(long_txt.notna() & r.duplicated("review_text_clean", keep="first"))).astype(int)

# B9. the outcome: mismatch flags (two definitions + both)
r["mismatch_rating"] = (r["rating"] <= 2).astype(int)
r["mismatch_rec"] = (1 - rec).astype("float")  # missing stays missing
r["mismatch_any"] = ((r["mismatch_rating"] == 1) | (r["mismatch_rec"].fillna(0) == 1)).astype(int)
r["mismatch_both"] = ((r["mismatch_rating"] == 1) & (r["mismatch_rec"] == 1)).astype(int)
r["recommend_known"] = rec.notna().astype(int)

# B10. join product features (many reviews -> one product)
keep_cols = (["product_id", "brand_id", "secondary_category", "tertiary_category", "size_ml_equiv", "price_per_100ml",
              "price_tier", "price_outlier", "is_kit", "n_ingredients", "ing_suspect_parse", "ingredients_clean",
              "best_for_skin", "new", "limited_edition", "online_only", "out_of_stock", "sephora_exclusive"]
             + list(GROUPS) + ["claim_vegan", "claim_cruelty_free", "claim_clean", "claim_fragrance_free", "claim_paraben_free"])
before = len(r)
r = r.merge(p[keep_cols], on="product_id", how="left", validate="m:1", indicator=True)
matched = int((r["_merge"] == "both").sum())
r = r.drop(columns="_merge")
log_step("Join product features", before, len(r), matched, "reviews matched to a product record")


def claimed_fit(row):
    if not isinstance(row["best_for_skin"], str) or row["skin_type"] == "not_stated":
        return np.nan
    return int(row["skin_type"] in row["best_for_skin"].split("|"))


r["claimed_fit"] = r.apply(claimed_fit, axis=1)

# B11. reviewer history and the time split (cut-off chosen from dates only, no model involved)
r["author_n_reviews"] = r.groupby("author_id")["author_id"].transform("size")
r["low_history_author"] = (r["author_n_reviews"] == 1).astype(int)
cutoff = r["submission_time"].quantile(0.80)
r["split"] = np.where(r["submission_time"] > cutoff, "test", "train")

# =============================================================================
# PART C - VALIDATION AND SUMMARY
# =============================================================================
checks = {
    "ratings_between_1_and_5": bool(r["rating"].between(1, 5).all()),
    "all_dates_parsed": bool(r["submission_time"].notna().all()),
    "dates_in_2008_2023": bool(r["submission_time"].between("2008-01-01", "2023-12-31").all()),
    "votes_add_up": bool((r["total_pos_feedback_count"] + r["total_neg_feedback_count"] == r["total_feedback_count"]).all()),
    "wilson_between_0_and_1": bool(r["helpful_wilson"].dropna().between(0, 1).all()),
    "no_exact_duplicates": bool(not r.duplicated(key).any()),
    "every_review_matched_to_product": bool(matched == len(r)),
    "mismatch_flags_binary": bool(set(r["mismatch_any"].unique()) <= {0, 1}),
    "train_test_do_not_overlap_in_time": bool(r.loc[r["split"] == "train", "submission_time"].max() <= cutoff
                                                < r.loc[r["split"] == "test", "submission_time"].min()),
}
METRICS["validation_checks"] = checks
assert all(checks.values()), f"validation failed: {checks}"

st = r["skin_type"]
mm = r.groupby("skin_type")["mismatch_any"].agg(["mean", "size"])
tone_mm = r.groupby("skin_tone_group")["mismatch_any"].agg(["mean", "size"])
METRICS["reviews_clean"] = {
    "rows": len(r), "cols": r.shape[1],
    "rows_removed_total": int(n_raw - len(r)),
    "skin_type_stated_pct": round(float(r["skin_type_stated"].mean() * 100), 1),
    "skin_tone_stated_pct": round(float(r["skin_tone_stated"].mean() * 100), 1),
    "skin_type_counts": r["skin_type"].value_counts().to_dict(),
    "skin_tone_group_counts": r["skin_tone_group"].value_counts(dropna=False).to_dict(),
    "recommend_missing_pct": round(float((1 - r["recommend_known"]).mean() * 100), 1),
    "helpful_missing_equals_no_votes": bool(r["helpful_ratio"].isna().equals((r["has_votes"] == 0))),
    "has_text_pct": round(float(r["has_text"].mean() * 100), 1),
    "short_text": int(r["short_text"].sum()),
    "non_ascii_suspect": int(r["non_ascii_suspect"].sum()),
    "feedback_cap_99pct": cap,
    "feedback_rows_capped": int((t > cap).sum()),
    "length_outliers_flagged": int(r["length_outlier"].sum()),
    "length_outlier_chars_above": round(float(np.expm1(hi_len))),
    "rating_rec_conflict": int(r["rating_rec_conflict"].sum()),
    "reaction_mention": int(r["reaction_mention"].sum()),
    "reaction_mention_pct": round(float(r["reaction_mention"].mean() * 100), 1),
    "star_text_conflict_rating4plus_with_reaction": int(r["star_text_conflict"].sum()),
    "incentivized_mention": int(r["incentivized_mention"].sum()),
    "dup_text_same_author_rows": int(r["dup_text_same_author"].sum()),
    "dup_text_other_author_rows": int(r["dup_text_other_author"].sum()),
    "text_first_occurrence_rows": int(r["text_first_occurrence"].sum()),
    "mismatch_rating_pct": round(float((r["mismatch_rating"] == 1).mean() * 100), 1),
    "mismatch_rec_pct_of_known": round(float((r.loc[r["recommend_known"] == 1, "mismatch_rec"] == 1).mean() * 100), 1),
    "mismatch_any_pct": round(float(r["mismatch_any"].mean() * 100), 1),
    "mismatch_both_pct": round(float(r["mismatch_both"].mean() * 100), 1),
    "mismatch_any_by_skin_type_pct": {k: round(float(v * 100), 1) for k, v in mm["mean"].items()},
    "mismatch_any_by_skin_type_n": {k: int(v) for k, v in mm["size"].items()},
    "mismatch_any_by_tone_group_pct": {k: round(float(v * 100), 1) for k, v in tone_mm["mean"].items()},
    "claimed_fit_available": int(r["claimed_fit"].notna().sum()),
    "claimed_fit_match_pct": round(float(r["claimed_fit"].mean() * 100), 1),
    "authors": int(r["author_id"].nunique()),
    "authors_2plus_reviews": int((r.groupby("author_id").size() >= 2).sum()),
    "low_history_rows_pct": round(float(r["low_history_author"].mean() * 100), 1),
    "split_cutoff_date": str(cutoff.date()),
    "split_counts": r["split"].value_counts().to_dict(),
    "date_min": str(r["submission_time"].min().date()),
    "date_max": str(r["submission_time"].max().date()),
}


def native(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


r.drop(columns=[]).to_csv(out_dir / "reviews_clean.csv.gz", index=False)
p.drop(columns=["ing_tokens"]).to_csv(out_dir / "products_clean.csv.gz", index=False)
with open(out_dir / "cleaning_log.json", "w") as f:
    json.dump({"steps": LOG, "metrics": METRICS}, f, indent=2, default=native)

print(json.dumps({"steps": LOG, "metrics": METRICS}, indent=1, default=native))

"""Book Recommender — Item-Based CF (Streamlit Cloud ready, title-only).

Best model from Recom_Week2.ipynb (RMSE 1.70):
  pivot(book_title x user_id).fillna(0) + cosine_similarity
  recommend_books(title) = top similar titles, excluding itself.

Covers/metadata are KEPT (unlike the notebook which dropped image_url_*).
Popularity table is the fallback for unknown titles / cold start.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data"
BOOKS_CSV = DATA_DIR / "Books.csv"
RATINGS_CSV = DATA_DIR / "Ratings.csv"

MIN_USER_RATINGS = 10
MIN_BOOK_RATINGS = 20


@st.cache_data(show_spinner="Loading Books & Ratings...")
def load_raw():
    books = pd.read_csv(BOOKS_CSV, low_memory=False, dtype={"ISBN": str})
    ratings = pd.read_csv(RATINGS_CSV, dtype={"User-ID": int, "ISBN": str, "Book-Rating": int})
    # Same standardization as Recom_Week2: lower + '-' -> '_'
    books.columns = books.columns.str.lower().str.replace("-", "_")
    ratings.columns = ratings.columns.str.lower().str.replace("-", "_")
    # Keep covers/metadata: strip text but do NOT drop image_url_* columns
    books["book_title"] = books["book_title"].astype(str).str.strip()
    books["book_author"] = books["book_author"].astype(str).str.strip()
    return books, ratings


@st.cache_resource(show_spinner="Building item-similarity matrix (one-time)...")
def build_model():
    books, ratings = load_raw()

    explicit = ratings[ratings["book_rating"] > 0].copy()
    user_counts = explicit["user_id"].value_counts()
    book_counts = explicit["isbn"].value_counts()
    keep_users = user_counts[user_counts >= MIN_USER_RATINGS].index
    keep_isbns = book_counts[book_counts >= MIN_BOOK_RATINGS].index
    filt = explicit[
        explicit["user_id"].isin(keep_users) & explicit["isbn"].isin(keep_isbns)
    ].copy()
    filt["user_total_ratings"] = filt["user_id"].map(user_counts)
    filt["book_total_ratings"] = filt["isbn"].map(book_counts)

    keep_cols = [
        "isbn", "book_title", "book_author", "year_of_publication",
        "publisher", "image_url_s", "image_url_m", "image_url_l",
    ]
    keep_cols = [c for c in keep_cols if c in books.columns]
    df_merged = filt.merge(books[keep_cols], on="isbn", how="inner")

    # user_id x book pivot aggregated by title (merges editions of same title)
    pivot = df_merged.pivot_table(
        index="book_title", columns="user_id", values="book_rating", aggfunc="mean"
    ).fillna(0)

    sim = cosine_similarity(pivot).astype(np.float32)
    book_sim = pd.DataFrame(sim, index=pivot.index, columns=pivot.index)

    num = df_merged.groupby("book_title")["book_rating"].count().reset_index()
    num.columns = ["book_title", "num_ratings"]
    avg = df_merged.groupby("book_title")["book_rating"].mean().reset_index()
    avg.columns = ["book_title", "avg_rating"]
    popular = pd.merge(num, avg, on="book_title")
    popular = popular[popular["num_ratings"] >= MIN_BOOK_RATINGS].sort_values(
        "avg_rating", ascending=False
    ).reset_index(drop=True)

    # One display row per title: prefer the edition with most ratings + a cover
    ranked = df_merged.sort_values("book_total_ratings", ascending=False)
    lookup = ranked.drop_duplicates(subset="book_title").set_index("book_title")

    return {
        "similarity": book_sim,
        "popular": popular,
        "lookup": lookup,
        "n_books": len(book_sim),
        "n_interactions": len(df_merged),
    }


def recommend(title, model, top_n=5):
    sim = model["similarity"]
    if title not in sim.index:
        return None
    scores = sim[title].drop(title, errors="ignore").sort_values(ascending=False)
    return scores.head(top_n)


def book_card(title, score, model):
    lookup = model["lookup"]
    pop = model["popular"].set_index("book_title")
    row = lookup.loc[title] if title in lookup.index else None
    img = None
    author = publisher = year = ""
    if row is not None:
        # row may be Series (unique) — handle both
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        for col in ("image_url_m", "image_url_l", "image_url_s"):
            v = row.get(col, "")
            if isinstance(v, str) and v.startswith("http"):
                img = v
                break
        author = str(row.get("book_author", ""))
        publisher = str(row.get("publisher", ""))
        year = str(row.get("year_of_publication", ""))
    avg = num = None
    if title in pop.index:
        r = pop.loc[title]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        avg, num = float(r["avg_rating"]), int(r["num_ratings"])
    return {"img": img, "author": author, "publisher": publisher,
            "year": year, "avg": avg, "num": num, "score": float(score)}


# ---------------- UI ----------------
st.set_page_config(page_title="Book Recommender — Item-Based CF", layout="wide")
st.title("📚 Book Recommender — Item-Based Collaborative Filtering")
st.caption(
    "Best model from Recom_Week2 (Item-Based CF, RMSE 1.70). "
    "Pick a title → get similar books with covers & metadata. "
    "Unknown titles fall back to the popularity list."
)

try:
    model = build_model()
except FileNotFoundError:
    st.error(f"Data not found. Expected `{BOOKS_CSV}` and `{RATINGS_CSV}` next to app.py.")
    st.stop()

st.sidebar.success(f"Catalog: {model['n_books']} books · {model['n_interactions']:,} ratings")
titles = sorted(model["similarity"].index.tolist())
default_ix = next(
    (i for i, t in enumerate(titles) if "Harry Potter and the Goblet of Fire" in t),
    0,
)
book = st.selectbox("Choose a book you like", titles, index=default_ix)
top_n = st.slider("Recommendations", 3, 10, 5)

recs = recommend(book, model, top_n=top_n)
if recs is None or recs.empty:
    st.warning(f"'{book}' not in the filtered catalog — showing popular books instead.")
    recs = model["popular"].head(top_n).set_index("book_title")["avg_rating"]

st.subheader(f"Because you liked: {book}")
cols = st.columns(min(len(recs), 5) if len(recs) > 0 else 1)
for i, (t, s) in enumerate(recs.items()):
    card = book_card(t, s, model)
    with cols[i % len(cols)]:
        if card["img"]:
            st.image(card["img"], use_container_width=True)
        else:
            st.info("No cover available")
        st.markdown(f"**{t}**")
        st.caption(f"{card['author']} · {card['publisher']} {card['year']}".strip(" ·"))
        meta = f"similarity {card['score']:.3f}"
        if card["avg"] is not None:
            meta += f" · ⭐ {card['avg']:.2f} ({card['num']} ratings)"
        st.caption(meta)

with st.expander("🔥 Popular right now (fallback list)"):
    pop = model["popular"].head(10)
    for _, r in pop.iterrows():
        st.write(f"**{r['book_title']}** — ⭐ {r['avg_rating']:.2f} ({r['num_ratings']} ratings)")

st.divider()
st.caption(
    f"Item-CF on filtered set (users ≥ {MIN_USER_RATINGS}, books ≥ {MIN_BOOK_RATINGS} ratings). "
    "Covers via original Image-URL-M/L (old Amazon links may be dead)."
)

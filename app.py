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
USERS_CSV = DATA_DIR / "Users.csv"

MIN_USER_RATINGS = 10
MIN_BOOK_RATINGS = 20
AGE_BINS = [0, 18, 35, 50, 65, 100]
AGE_LABELS = ["Teens", "Young Adult", "Adult", "Middle Aged", "Senior"]


@st.cache_data(show_spinner="Loading Books, Ratings & Users...")
def load_raw():
    books = pd.read_csv(BOOKS_CSV, low_memory=False, dtype={"ISBN": str})
    ratings = pd.read_csv(RATINGS_CSV, dtype={"User-ID": int, "ISBN": str, "Book-Rating": int})
    users = pd.read_csv(USERS_CSV, dtype={"User-ID": int}, low_memory=False)
    # Same standardization as Recom_Week2: lower + '-' -> '_'
    books.columns = books.columns.str.lower().str.replace("-", "_")
    ratings.columns = ratings.columns.str.lower().str.replace("-", "_")
    users.columns = users.columns.str.lower().str.replace("-", "_")
    # Keep covers/metadata: strip text but do NOT drop image_url_* columns
    books["book_title"] = books["book_title"].astype(str).str.strip()
    books["book_author"] = books["book_author"].astype(str).str.strip()
    # Same age cleaning as Recom_Week2: outliers -> NaN -> median
    # Raw missing snapshot first (mirrors notebook Cell 6, before cleaning)
    raw_missing = {
        "users.age": float(users["age"].isna().mean()),
        "books.book_author": float(books["book_author"].isna().mean()),
        "books.publisher": float(books["publisher"].isna().mean()),
    }
    users["age"] = pd.to_numeric(users["age"], errors="coerce")
    users.loc[(users["age"] < 5) | (users["age"] > 90), "age"] = np.nan
    users["age"] = users["age"].fillna(users["age"].median())
    users["age_group"] = pd.cut(
        users["age"], bins=AGE_BINS, labels=AGE_LABELS
    )
    # Location split: city / state / country. Rows have 2-9 comma parts
    # ("suburb, city, state, country"), so country = last non-empty part.
    # (Notebook Cell 16's iloc[:,-1] on the 9-wide frame yields NaN for
    # normal 3-part rows; this follows the intent of its parse_location().)
    loc_split = users["location"].astype(str).str.split(",", expand=True)
    nparts = loc_split.notna().sum(axis=1)
    users["city"] = loc_split[0].str.strip()
    users["state"] = loc_split[1].str.strip().where(nparts >= 2)
    users["country"] = (
        loc_split.ffill(axis=1).iloc[:, -1].str.strip().where(nparts >= 2)
    )
    return books, ratings, users, raw_missing


@st.cache_resource(show_spinner="Building item-similarity matrix (one-time)...")
def build_model():
    books, ratings, _, _ = load_raw()

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


@st.cache_data(show_spinner="Computing EDA highlights...")
def compute_eda():
    """Lightweight EDA snapshot mirroring Recom_Week2 cells 5-12."""
    books, ratings, users, raw_missing = load_raw()

    n_users, n_books, n_ratings = len(users), len(books), len(ratings)
    n_explicit = int((ratings["book_rating"] > 0).sum())
    n_implicit = n_ratings - n_explicit
    explicit_pct = 100.0 * n_explicit / n_ratings if n_ratings else 0.0
    n_active_users = int(
        (ratings[ratings["book_rating"] > 0]["user_id"].value_counts() >= MIN_USER_RATINGS).sum()
    )

    # Rating distribution 0-10 (notebook Cell 10 countplot)
    rating_dist = ratings["book_rating"].value_counts().sort_index()
    rating_dist.index = rating_dist.index.astype(str)

    # Top authors / publishers by explicit-rating count
    explicit = ratings[ratings["book_rating"] > 0].merge(
        books[["isbn", "book_author", "publisher"]], on="isbn", how="inner"
    )
    top_authors = explicit["book_author"].value_counts().head(10)
    top_publishers = explicit["publisher"].value_counts().head(10)

    # Users: age groups + geography (notebook Cells 11/13/14/16)
    age_groups = users["age_group"].value_counts().reindex(AGE_LABELS, fill_value=0)
    median_age = float(users["age"].median())
    top_countries = users["country"].value_counts().head(10)
    top_cities = users["city"].value_counts().head(10)
    missing = pd.DataFrame({
        "missing_pct": [100.0 * raw_missing["users.age"],
                        100.0 * raw_missing["books.book_author"],
                        100.0 * raw_missing["books.publisher"]],
    }, index=["users.age", "books.book_author", "books.publisher"])

    return {
        "n_users": n_users, "n_books": n_books, "n_ratings": n_ratings,
        "n_explicit": n_explicit, "n_implicit": n_implicit,
        "explicit_pct": explicit_pct, "n_active_users": n_active_users,
        "rating_dist": rating_dist,
        "top_authors": top_authors, "top_publishers": top_publishers,
        "age_groups": age_groups, "median_age": median_age,
        "top_countries": top_countries, "top_cities": top_cities,
        "missing": missing,
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

tab_rec, tab_eda = st.tabs(["Recommend", "🔍 Explore the data"])

with tab_rec:
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

with tab_eda:
    eda = compute_eda()
    n_filt_books = model["n_books"]
    density = model["n_interactions"] / max(n_filt_books * max(eda["n_active_users"], 1), 1)

    st.subheader("Data snapshot")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Users", f"{eda['n_users']:,}")
    k2.metric("Books (catalog)", f"{eda['n_books']:,}")
    k3.metric("Ratings (raw)", f"{eda['n_ratings']:,}")
    k4.metric("Explicit ratings", f"{eda['explicit_pct']:.1f}%")
    k5.metric("Median age", f"{eda['median_age']:.0f}")
    st.caption(
        f"Implicit (0) ratings: {eda['n_implicit']:,}. "
        f"Filtered modeling set: {model['n_interactions']:,} ratings across "
        f"{n_filt_books} books × {eda['n_active_users']:,} active users "
        f"(density {density:.2%}, users ≥ {MIN_USER_RATINGS}, books ≥ {MIN_BOOK_RATINGS})."
    )

    st.subheader("Rating distribution")
    st.bar_chart(eda["rating_dist"])
    st.caption("0 = implicit (no score given); 1–10 = explicit scores. Mirrors the notebook countplot.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top authors (explicit ratings)")
        st.bar_chart(eda["top_authors"])
    with c2:
        st.subheader("Top publishers (explicit ratings)")
        st.bar_chart(eda["top_publishers"])

    st.subheader("Total ratings vs average rating")
    st.scatter_chart(
        model["popular"][["num_ratings", "avg_rating"]],
        x="num_ratings", y="avg_rating",
    )
    st.caption("Each dot is a book (≥20 ratings). Mirrors the notebook scatter plot.")

    st.subheader("Readers by age group")
    st.bar_chart(eda["age_groups"])
    st.caption("Ages <5 or >90 treated as outliers → median, then binned (notebook Cells 13–14).")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Top reader countries")
        st.bar_chart(eda["top_countries"])
    with c4:
        st.subheader("Top reader cities")
        st.bar_chart(eda["top_cities"])

    with st.expander("Missing values"):
        st.dataframe(eda["missing"].style.format({"missing_pct": "{:.2f}%"}))

st.divider()
st.caption(
    f"Item-CF on filtered set (users ≥ {MIN_USER_RATINGS}, books ≥ {MIN_BOOK_RATINGS} ratings). "
    "Covers via original Image-URL-M/L (old Amazon links may be dead)."
)

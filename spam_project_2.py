# app.py — MERGED: Project1 (Spam+News+Auth) + Project2 (Updated Movie Recommender replacing old one)
import os
import urllib.parse
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import base64
import joblib
import re
import time
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict
from hashlib import md5
from sklearn.preprocessing import LabelEncoder

# ---------------- CONFIG ----------------
DB_PATH = "app_data.db"
SPAM_MODEL_PATH = "spam_sgd_clf.pkl"
NEWS_MODEL_PATH = "news_classification.pkl"
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASS = "Admin@123"  # change for production
MAX_LOG_DISPLAY = 500

OMDB_API_KEY = "48972038"
DEFAULT_POSTER = "default_poster.jpg"

# Canonical default fallback categories (uppercase)
DEFAULT_CANONICAL_NEWS = ["SPORTS", "NEWS", "ECONOMICS", "ENTERTAINMENT", "POLITICS", "BUSINESS", "TECHNOLOGY", "WORLD"]

# ---------------- UTIL: DB ----------------
def get_db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        username TEXT,
        password_hash TEXT,
        is_admin INTEGER DEFAULT 0,
        is_blocked INTEGER DEFAULT 0,
        created_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        message TEXT,
        result TEXT,
        created_at TEXT,
        model TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS contacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        sender TEXT,
        email TEXT,
        message TEXT,
        resolved INTEGER DEFAULT 0,
        created_at TEXT
    )""")
    conn.commit()
    conn.close()

def create_admin_if_not_exists():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE is_admin=1 LIMIT 1")
    if cur.fetchone() is None:
        pw = hash_password(ADMIN_PASS)
        created_at = datetime.utcnow().isoformat()
        cur.execute(
            "INSERT OR IGNORE INTO users (email, username, password_hash, is_admin, created_at) VALUES (?,?,?,?,?)",
            (ADMIN_EMAIL, "admin", pw, 1, created_at),
        )
        conn.commit()
    conn.close()

def create_user(email, username, password):
    conn = get_db_conn()
    cur = conn.cursor()
    pw = hash_password(password)
    created_at = datetime.utcnow().isoformat()
    try:
        cur.execute("INSERT INTO users (email, username, password_hash, is_admin, created_at) VALUES (?,?,?,?,?)",
                    (email, username, pw, 0, created_at))
        conn.commit()
        return True, cur.lastrowid
    except sqlite3.IntegrityError:
        return False, "Email already registered"
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def get_user_by_email(email):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email=?", (email,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_id(uid):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (uid,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def log_prediction(user_id, message, result, model_type="spam"):
    conn = get_db_conn()
    cur = conn.cursor()
    created_at = datetime.utcnow().isoformat()
    try:
        cur.execute("INSERT INTO predictions (user_id, message, result, created_at, model) VALUES (?,?,?,?,?)",
                    (user_id, message, result, created_at, model_type))
        conn.commit()
    except Exception as e:
        print("log_prediction error:", e)
    finally:
        conn.close()

def get_user_predictions(user_id, limit=1000):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT ?", (user_id, limit))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_all_predictions(limit=5000):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT p.*, u.email, u.username FROM predictions p LEFT JOIN users u ON p.user_id=u.id ORDER BY p.created_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def send_contact(user_id, sender, email, message):
    conn = get_db_conn()
    cur = conn.cursor()
    created_at = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO contacts (user_id, sender, email, message, created_at) VALUES (?,?,?,?,?)",
                (user_id, sender, email, message, created_at))
    conn.commit()
    conn.close()

def get_contacts(limit=500):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT c.*, u.email AS user_email, u.username FROM contacts c LEFT JOIN users u ON c.user_id=u.id ORDER BY c.created_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def toggle_resolve_contact(contact_id, resolved):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("UPDATE contacts SET resolved=? WHERE id=?", (1 if resolved else 0, contact_id))
    conn.commit()
    conn.close()

# ---------------- AUTH ----------------
def hash_password(password: str, salt: str = None) -> str:
    if salt is None:
        salt = base64.urlsafe_b64encode(os.urandom(9)).decode()
    digest = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${digest}"

def verify_password(stored: str, provided: str) -> bool:
    if not stored or "$" not in stored:
        return False
    try:
        salt, digest = stored.split("$", 1)
    except Exception:
        return False
    check = hashlib.sha256((salt + provided).encode()).hexdigest()
    return check == digest

# ---------------- MODEL LOADING (robust) & PREDICTION HELPERS ----------------
@st.cache_resource
def _cached_load(path, mtime):
    try:
        if os.path.exists(path):
            model = joblib.load(path)
            print(f"Loaded model from {path}: type={type(model)}")
            return model
        else:
            print(f"Model path not found: {path}")
            return None
    except Exception as e:
        print("Model load error:", e)
        return None

def load_pipeline(path):
    """Keep this available for spam model and simple pipelines (cached)."""
    if os.path.exists(path):
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            mtime = None
        return _cached_load(path, mtime)
    return None

def save_uploaded_model(uploaded_file, dest_path):
    try:
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True, None
    except Exception as e:
        return False, str(e)

# Preprocessing for news text — keep simple & deterministic
def preprocess_news(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()
    # keep letters, numbers and spaces
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

class ModelWrapper:
    """
    Wraps various saved formats into a unified object exposing .predict(list_of_texts)
    Handles pipelines, (vectorizer, clf), dicts with 'classes', LabelEncoders, etc.
    """
    def __init__(self, raw):
        self.raw = raw
        self.pipeline = None
        self.vectorizer = None
        self.clf = None
        self.label_encoder = None
        self.class_map = None

        # If it's a sklearn pipeline or estimator with predict
        if hasattr(raw, "predict") and hasattr(raw, "fit"):
            self.pipeline = raw
        # tuple/list: try to detect vectorizer+clf or clf+le
        elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
            for item in raw:
                if hasattr(item, "predict"):
                    self.clf = item
                if hasattr(item, "transform"):
                    self.vectorizer = item
                if isinstance(item, LabelEncoder) or hasattr(item, "inverse_transform"):
                    self.label_encoder = item
        elif isinstance(raw, dict):
            self.pipeline = raw.get("pipeline") or raw.get("model") or raw.get("clf") or raw.get("estimator")
            self.vectorizer = raw.get("vectorizer") or raw.get("vec")
            self.clf = raw.get("clf") or raw.get("classifier") or raw.get("estimator")
            self.label_encoder = raw.get("label_encoder") or raw.get("le")
            if "classes" in raw and raw["classes"] is not None:
                classes = list(raw["classes"])
                self.class_map = {i: classes[i] for i in range(len(classes))}
            if "class_map" in raw:
                self.class_map = raw.get("class_map")

        # attempt to discover attributes if still missing
        if self.pipeline is None and self.clf is None and hasattr(raw, "clf"):
            self.clf = getattr(raw, "clf")
        if self.pipeline is None and self.vectorizer is None and hasattr(raw, "vectorizer"):
            self.vectorizer = getattr(raw, "vectorizer")

    def _preprocess_inputs(self, inputs, is_news=False):
        # normalize input list
        if isinstance(inputs, (pd.Series, np.ndarray)):
            arr = inputs.tolist()
        elif isinstance(inputs, str):
            arr = [inputs]
        else:
            arr = list(inputs)
        arr = [("" if x is None else str(x)) for x in arr]
        if is_news:
            arr = [preprocess_news(x) for x in arr]
        return arr

    def predict(self, inputs, is_news=False):
        arr = self._preprocess_inputs(inputs, is_news=is_news)
        # pipeline
        if self.pipeline is not None:
            preds = self.pipeline.predict(arr)
            return self._maybe_decode(preds)
        # vectorizer + clf
        if self.vectorizer is not None and self.clf is not None:
            X = self.vectorizer.transform(arr)
            preds = self.clf.predict(X)
            return self._maybe_decode(preds)
        # only clf
        if self.clf is not None:
            try:
                preds = self.clf.predict(arr)
                return self._maybe_decode(preds)
            except Exception:
                try:
                    X = np.array(arr, dtype=object).reshape(-1,1)
                    preds = self.clf.predict(X)
                    return self._maybe_decode(preds)
                except Exception as e:
                    raise RuntimeError(f"Classifier predict failed: {e}")
        raise RuntimeError("ModelWrapper: no usable pipeline/clf/vectorizer found")

    def _maybe_decode(self, preds):
        out = []
        for p in preds:
            if isinstance(p, (bytes, bytearray)):
                p = p.decode("utf-8")
            out.append(p)
        # if label encoder present
        if self.label_encoder is not None:
            try:
                inv = self.label_encoder.inverse_transform(np.array(out).astype(object))
                return [str(x) for x in inv]
            except Exception:
                try:
                    inv = self.label_encoder.inverse_transform(np.array(out, dtype=int))
                    return [str(x) for x in inv]
                except Exception:
                    pass
        # map numeric indices via class_map
        if self.class_map is not None:
            mapped = []
            for p in out:
                try:
                    k = int(p)
                    mapped.append(self.class_map.get(k, str(p)))
                except Exception:
                    mapped.append(self.class_map.get(p, str(p)))
            return mapped
        return [str(x) for x in out]

def load_raw(path):
    """Load raw object from path using joblib (cached loader above)."""
    if os.path.exists(path):
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            mtime = None
        return _cached_load(path, mtime)
    return None

def load_model_flexible(path):
    raw = load_raw(path)
    if raw is None:
        return None
    try:
        wrapper = ModelWrapper(raw)
        # quick sanity call (don't raise on fail)
        try:
            _ = wrapper.predict(["test"], is_news=False)
        except Exception:
            pass
        return wrapper
    except Exception as e:
        print("Failed to wrap model:", e)
        return None

def _try_predict(model, inputs, is_news=False):
    """
    Try to predict using `model`. Accepts:
      - sklearn-like object with .predict(list)
      - ModelWrapper (our wrapper)
      - raw joblib pipeline
    `is_news` toggles simple preprocessing for news text.
    """
    if model is None:
        raise RuntimeError("Model is None")
    # if it's our ModelWrapper, use its predict with is_news flag
    if isinstance(model, ModelWrapper):
        return model.predict(inputs, is_news=is_news)
    # else fallback to normal predict, but try to preprocess if requested
    arr = inputs
    if is_news:
        arr = [preprocess_news(x) for x in inputs]
    # try several shapes like original code
    errors = []
    try:
        preds = model.predict(list(arr))
        return [p.decode('utf-8') if isinstance(p, (bytes,bytearray)) else str(p) for p in preds]
    except Exception as e1:
        errors.append(e1)
    try:
        X = np.array(list(arr), dtype=object).reshape(-1,1)
        preds = model.predict(X)
        return [p.decode('utf-8') if isinstance(p, (bytes,bytearray)) else str(p) for p in preds]
    except Exception as e2:
        errors.append(e2)
    try:
        X = np.array(list(arr), dtype=object)
        preds = model.predict(X)
        return [p.decode('utf-8') if isinstance(p, (bytes,bytearray)) else str(p) for p in preds]
    except Exception as e3:
        errors.append(e3)
    raise RuntimeError(f"All prediction attempts failed. Errors: {errors}")

def chunked_predict(model, inputs: List[str], chunk: int = 2000, is_news: bool = False):
    out = []
    n = len(inputs)
    for i in range(0, n, chunk):
        batch = inputs[i:i+chunk]
        out_batch = _try_predict(model, batch, is_news=is_news)
        out.extend(out_batch)
    return out

def normalize_spam_label(label):
    """Return canonical lowercase spam/ham/error labels for programmatic use."""
    if label is None:
        return "error"
    s = str(label).strip().lower()
    if s in ("spam", "1", "true", "t"):
        return "spam"
    if s in ("ham", "0", "false", "f", "not spam"):
        return "ham"
    # fallback: return stripped lowercase string so comparisons remain predictable
    return s

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ---------------- NEWS NORMALIZATION HELPERS ----------------
def _extract_model_classes(news_model):
    try:
        if isinstance(news_model, ModelWrapper):
            raw = news_model.raw
            if hasattr(raw, "classes_"):
                return [str(x).strip().upper() for x in list(getattr(raw, "classes_"))]
            if news_model.class_map is not None:
                vals = list(news_model.class_map.values())
                return [str(x).strip().upper() for x in vals]
            if news_model.pipeline is not None and hasattr(news_model.pipeline, "classes_"):
                return [str(x).strip().upper() for x in list(news_model.pipeline.classes_)]
        else:
            if hasattr(news_model, "classes_"):
                return [str(x).strip().upper() for x in list(getattr(news_model, "classes_"))]
            if hasattr(news_model, "named_steps"):
                for step in news_model.named_steps.values():
                    if hasattr(step, "classes_"):
                        return [str(x).strip().upper() for x in list(step.classes_)]
    except Exception:
        pass
    return DEFAULT_CANONICAL_NEWS

def normalize_news_output(raw_label, news_model=None, canonical_list=None):
    if raw_label is None:
        return "NEWS"
    s = str(raw_label).strip()
    if s == "":
        return "NEWS"
    if canonical_list is None:
        try:
            canonical_list = _extract_model_classes(news_model) if news_model is not None else DEFAULT_CANONICAL_NEWS
        except Exception:
            canonical_list = DEFAULT_CANONICAL_NEWS
    canonical_set = set([c.strip().upper() for c in canonical_list if c is not None])
    if s.upper() in canonical_set:
        return s.upper()
    if s.title().upper() in canonical_set:
        return s.title().upper()
    lower = s.lower()
    sports_keywords = ["sport", "football", "cricket", "match", "tournament", "league", "score", "goal", "athlete"]
    for k in sports_keywords:
        if k in lower:
            return "SPORTS" if "SPORTS" in canonical_set else next(iter(canonical_set))
    econ_keywords = ["econom", "finance", "business", "stock", "market", "economy", "gdp", "inflation"]
    for k in econ_keywords:
        if k in lower:
            return "ECONOMICS" if "ECONOMICS" in canonical_set else next(iter(canonical_set))
    ent_keywords = ["entertain", "movie", "music", "film", "celebr", "tv", "show", "actor", "actress", "concert"]
    for k in ent_keywords:
        if k in lower:
            return "ENTERTAINMENT" if "ENTERTAINMENT" in canonical_set else next(iter(canonical_set))
    news_keywords = ["news", "politic", "government", "election", "president", "minister", "world", "breaking"]
    for k in news_keywords:
        if k in lower:
            return "NEWS" if "NEWS" in canonical_set else next(iter(canonical_set))
    try:
        if s.isdigit():
            idx = int(s)
            sorted_list = list(canonical_set)
            if 0 <= idx < len(sorted_list):
                return sorted_list[idx]
    except Exception:
        pass
    up = s.upper()
    if up in canonical_set:
        return up
    return "NEWS"

# ---------------- NEWS CATEGORY STYLING ----------------
NEWS_COLORS = {
    "SPORTS": ("#E0F2FE", "#0284C7", "#032B47"),
    "POLITICS": ("#F3E8FF", "#7C3AED", "#32104D"),
    "ENTERTAINMENT": ("#FFE4E6", "#E11D48", "#4A0716"),
    "NEWS": ("#FEF9C3", "#CA8A04", "#453906"),
    "ECONOMICS": ("#ECFCCB", "#65A30D", "#17360F"),
    "BUSINESS": ("#E6FFFA", "#059669", "#052E1F"),
    "TECHNOLOGY": ("#FFF7ED", "#FB923C", "#2B1208"),
    "WORLD": ("#F0F9FF", "#0EA5A4", "#042F2F"),
}

def category_style(label):
    if label is None:
        label = "Unknown"
    key = str(label).strip().upper()
    if key in NEWS_COLORS:
        bg, border, text = NEWS_COLORS[key]
    else:
        palette = list(NEWS_COLORS.values())
        idx = int(md5(key.encode()).hexdigest(), 16) % len(palette)
        bg, border, text = palette[idx]
    return f"background:{bg}; border-left:6px solid {border}; color:{text}; padding:10px; border-radius:10px; margin-bottom:8px; font-weight:700;"

# ---------------- CSS (Dark Premium + input fix) ----------------
st.set_page_config(page_title="AI Duo Classifier", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

:root {
  --bg1: #071026;
  --bg2: #081227;
  --card: rgba(255,255,255,0.03);
  --muted: #9fb0c8;
  --accent1: #7dd3fc;
  --accent2: #ff9ecf;
  --text: #e6eef6;
  --input-bg: rgba(255,255,255,0.06);
}

/* App background */
.stApp {
  background: radial-gradient(circle at 10% 20%, rgba(125,211,252,0.04) 0%, transparent 10%),
              radial-gradient(circle at 90% 80%, rgba(255,158,207,0.03) 0%, transparent 12%),
              linear-gradient(180deg, var(--bg1), var(--bg2));
  color: var(--text);
  font-family: 'Poppins', sans-serif;
  min-height:100vh;
  padding-bottom:40px;
}

/* Header */
header, header > div, div[role="banner"] {
  background: transparent !important;
  color: var(--text) !important;
}

/* Sidebar */
section[data-testid="stSidebar"], div[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)) !important;
  color: var(--text) !important;
  border-right: 1px solid rgba(255,255,255,0.03);
}

/* Cards */
.card {
  background: var(--card);
  padding:18px;
  border-radius:16px;
  border: 1px solid rgba(255,255,255,0.04);
  box-shadow: 0 8px 30px rgba(3,7,18,0.6);
  margin-bottom:12px;
  color: var(--text);
}

/* Titles */
.title { font-size:28px; font-weight:800; color:var(--text); }
.subtitle { color: var(--muted); margin-top:4px; }

/* Buttons */
.stButton>button {
  background: linear-gradient(90deg, var(--accent1), var(--accent2));
  color: #02161c;
  padding:8px 14px;
  border-radius:10px;
  font-weight:800;
  border: none;
}
.stButton>button:hover { transform: translateY(-2px); }

/* Inputs, textarea - ensure visible on dark background */
.stTextInput>div>input,
.stTextArea>div>textarea,
textarea, input[type="text"], input[type="email"] {
  background: var(--input-bg) !important;
  color: #0b0b0b !important;  /* force visible text color */
  border: 1px solid rgba(255,255,255,0.06) !important;
  padding: 10px !important;
  border-radius:8px !important;
}

/* Table text */
.stDataFrame table { color: var(--text) !important; }

/* Spam / Ham badges */
.spam-bulk { padding:12px; background: linear-gradient(90deg,#ffd9d9,#ffecec); border-left:6px solid #ff4b4b; color:#6b1515; border-radius:10px; margin-bottom:8px; word-wrap: break-word; }
.ham-bulk  { padding:12px; background: linear-gradient(90deg,#dfffe5,#effff5); border-left:6px solid #22c55e; color:#044b2b; border-radius:10px; margin-bottom:8px; word-wrap: break-word; }

/* KPI */
.kpi { display:inline-block; padding:12px 16px; border-radius:12px; background: rgba(255,255,255,0.02); margin-right:8px; font-weight:700; color:var(--text); }

</style>
""", unsafe_allow_html=True)

# ---------------- Movie recommender helpers (from Project 2) ----------------
@st.cache_data
def fetch_omdb_by_id(movie_id: str) -> Dict:
    if not movie_id:
        return {"Response": "False"}
    url = f"http://www.omdbapi.com/?i={movie_id}&apikey={OMDB_API_KEY}&plot=full"
    try:
        return requests.get(url, timeout=6).json()
    except:
        return {"Response": "False"}

@st.cache_data
def fetch_omdb_by_title(title: str) -> Dict:
    if not title:
        return {"Response": "False"}
    url = f"http://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={OMDB_API_KEY}&plot=full"
    try:
        return requests.get(url, timeout=6).json()
    except:
        return {"Response": "False"}

def safe_poster_url(omd: Dict):
    p = omd.get("Poster", "")
    return p if p and p != "N/A" else None

def imdb_url_from_omd(omd: Dict):
    imdbid = omd.get("imdbID")
    if imdbid:
        return f"https://www.imdb.com/title/{imdbid}/"
    title = omd.get("Title", "")
    return f"https://www.imdb.com/find?q={urllib.parse.quote(title)}&s=tt"

def yt_songs_link(title: str):
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote(title + ' songs')}"

# ---------------- Movie data loader (uses project1 loader) ----------------
@st.cache_resource
def load_movie_models_wrapper():
    df = None
    vectors = None
    model = None
    try:
        if os.path.exists("df.pkl"):
            df = _cached_load("df.pkl", os.path.getmtime("df.pkl"))
        if os.path.exists("vectors.pkl"):
            vectors = _cached_load("vectors.pkl", os.path.getmtime("vectors.pkl"))
        if os.path.exists("model.pkl"):
            model = _cached_load("model.pkl", os.path.getmtime("model.pkl"))
    except Exception as e:
        print("Movie load error:", e)
    return df, vectors, model

@st.cache_resource
def load_movie_models():
    return load_movie_models_wrapper()

# ---------------- Initialize DB + Admin ----------------
try:
    init_db()
    create_admin_if_not_exists()
except Exception as e:
    st.error("DB init error: " + str(e))

# ---------------- Load models (cached) ----------------
spam_model = load_pipeline(SPAM_MODEL_PATH)
news_model = load_model_flexible(NEWS_MODEL_PATH)
if news_model is not None:
    try:
        setattr(news_model, "is_news", True)
    except Exception:
        pass

NEWS_CANONICAL_CLASSES = _extract_model_classes(news_model) if news_model is not None else DEFAULT_CANONICAL_NEWS

# ---------------- Session defaults ----------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["user"] = None
if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "spam"
if "reload_models_flag" not in st.session_state:
    st.session_state["reload_models_flag"] = False

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("<div class='card'><b>About the project</b><br>", unsafe_allow_html=True)
    st.markdown("<div class='card'><b>Admin pass** Admin@123</b><br>", unsafe_allow_html=True)
    st.markdown("<div class='card'><h3 class='title'>AI Duo Classifier</h3><div class='subtitle'>Spam + News — Admin & User dashboards</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='card'><b>We're here to make your life less difficult, not more. Your vision. Our expertise. Unstoppable results</b><br>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-muted'>Spam: {'Loaded' if spam_model is not None else 'Not loaded'}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-muted'>News: {'Loaded' if news_model is not None else 'Not loaded'}</div>", unsafe_allow_html=True)
    movie_df_check, movie_vec_check, movie_model_check = load_movie_models()
    st.markdown(f"<div class='small-muted'>Movies: {'Loaded' if movie_model_check is not None else 'Not loaded'}</div>", unsafe_allow_html=True)
    if st.button("Refresh models"):
        st.session_state["reload_models_flag"] = True
        st.session_state["last_model_refresh"] = datetime.utcnow().isoformat()
        st.success("Model refresh requested...")

# If reload flag set, reload and rerun
if st.session_state.get("reload_models_flag", False):
    try:
        spam_model = load_pipeline(SPAM_MODEL_PATH)
        news_model = load_model_flexible(NEWS_MODEL_PATH)
        if news_model is not None:
            setattr(news_model, "is_news", True)
        NEWS_CANONICAL_CLASSES = _extract_model_classes(news_model) if news_model is not None else DEFAULT_CANONICAL_NEWS
        _ = load_movie_models()
        st.session_state["reload_models_flag"] = False
        st.rerun()
    except Exception as e:
        st.error(f"Model reload failed: {e}")

# Header
st.markdown("<div class='card'><div style='display:flex;justify-content:space-between;align-items:center;'><div><div class='title'>🤖 AI Duo Classifier</div><div class='subtitle'>Spam & News classification — Real world ready</div></div></div></div>", unsafe_allow_html=True)

# Router & helpers
page = st.session_state["page"]

def login_user(email, password):
    u = get_user_by_email(email)
    if not u:
        return False, "No account found"
    if int(u.get("is_blocked") or 0) == 1:
        return False, "Account blocked"
    if verify_password(u.get("password_hash"), password):
        return True, u
    return False, "Invalid credentials"

# ---------------- PAGES ----------------
def show_landing():
    col1, col2 = st.columns([1,1], gap="large")
    with col1:
        st.markdown("<div class='card'><h3>Welcome</h3><p class='subtitle'>Sign in to continue</p></div>", unsafe_allow_html=True)
        if st.button("Admin Portal"):
            st.session_state["page"] = "admin_login"
    with col2:
        st.markdown("<div class='card'><h3>Get Started</h3><p class='subtitle'>User Portal — Signup / Login</p></div>", unsafe_allow_html=True)
        if st.button("User Portal"):
            st.session_state["page"] = "user_auth"

# HOME
if page == "home":
    show_landing()

# ADMIN LOGIN
elif page == "admin_login":
    st.markdown("<div class='card'><h3>Admin Login</h3></div>", unsafe_allow_html=True)
    em = st.text_input("Email", value=ADMIN_EMAIL, key="admin_email")
    pw = st.text_input("Password", type="password", key="admin_pw")
    c1, c2 = st.columns(2)
    if c1.button("Login as Admin"):
        ok, data = login_user(em, pw)
        if ok and int(data.get("is_admin") or 0) == 1:
            st.session_state["logged_in"] = True
            st.session_state["user"] = data
            st.session_state["page"] = "admin_dashboard"
            st.success("Welcome Admin")
        else:
            st.error("Invalid admin credentials")
    if c2.button("Back"):
        st.session_state["page"] = "home"

# USER AUTH
elif page == "user_auth":
    st.markdown("<div class='card'><h3>User — Login / Signup</h3></div>", unsafe_allow_html=True)
    tab = st.tabs(["Login","Signup"])
    with tab[0]:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            ok, info = login_user(email, password)
            if ok:
                st.session_state["logged_in"] = True
                st.session_state["user"] = info
                st.success(f"Welcome {info.get('username','User')}")
                st.session_state["page"] = "user_choice"
            else:
                st.error(info)
    with tab[1]:
        s_email = st.text_input("Email (signup)", key="su_email")
        s_username = st.text_input("Username", key="su_username")
        s_password = st.text_input("Password", type="password", key="su_pw")
        if st.button("Create Account"):
            if not s_email or not s_username or not s_password:
                st.error("All fields required")
            else:
                ok, res = create_user(s_email, s_username, s_password)
                if ok:
                    st.success("Account created — please login")
                else:
                    st.error(res)
    if st.button("Back to Home"):
        st.session_state["page"] = "home"

# ADMIN DASHBOARD
elif page == "admin_dashboard":
    if not st.session_state.get("logged_in") or not st.session_state.get("user") or int(st.session_state["user"].get("is_admin") or 0) != 1:
        st.error("Admin login required")
        if st.button("Go Home"):
            st.session_state["page"] = "home"
    else:
        admin = st.session_state["user"]
        st.markdown(f"<div class='card'><h3>Admin Dashboard</h3><div class='subtitle'>Signed in as {admin.get('email')}</div></div>", unsafe_allow_html=True)

        preds = get_all_predictions(limit=2000)
        dfp = pd.DataFrame(preds) if preds else pd.DataFrame()

        # KPIs
        try:
            total_users = pd.read_sql_query("SELECT COUNT(*) as c FROM users", get_db_conn())["c"].iloc[0]
        except Exception:
            total_users = 0
        total_preds = len(dfp)
        spam_count = int(((dfp["model"].astype(str)=="spam") & (dfp["result"].str.lower()=="spam")).sum()) if not dfp.empty else 0
        ham_count = int(((dfp["model"].astype(str)=="spam") & (dfp["result"].str.lower()=="ham")).sum()) if not dfp.empty else 0

        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f"<div class='kpi'><h3>{total_users}</h3><p>Total Users</p></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='kpi'><h3>{total_preds}</h3><p>Total Predictions</p></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='kpi'><h3>🚫 {spam_count}</h3><p>Spam</p></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='kpi'><h3>✅ {ham_count}</h3><p>Ham</p></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h4>Models</h4>", unsafe_allow_html=True)
        colA, colB = st.columns([2,3])
        with colA:
            st.write(f"Spam: {'Loaded' if spam_model is not None else 'Not loaded'}")
            st.write(f"News: {'Loaded' if news_model is not None else 'Not loaded'}")
            st.write("Last refresh: " + str(st.session_state.get("last_model_refresh", "never")))

            if st.button("Refresh models (admin)"):
                st.session_state["reload_models_flag"] = True
                st.session_state["last_model_refresh"] = datetime.utcnow().isoformat()

        with colB:
            st.markdown("<div class='card'><b>Admin Check the model performance</b></div>", unsafe_allow_html=True)
            st.info("Admin upload model option removed for security.")

        st.markdown("---")
        st.markdown("<h4>Users</h4>", unsafe_allow_html=True)
        try:
            users_df = pd.read_sql_query("SELECT id, email, username, is_admin, created_at FROM users ORDER BY created_at DESC LIMIT 1000", get_db_conn())
            st.dataframe(users_df)
        except Exception as e:
            st.error("Could not load users: " + str(e))

        st.markdown("<h4>Recent Predictions</h4>", unsafe_allow_html=True)
        if not dfp.empty:
            df_show = dfp.rename(columns={"created_at":"At","email":"User Email","message":"Message","result":"Result","model":"Model","username":"Username"})
            cols_to_show = [c for c in ["At","Username","User Email","Model","Message","Result"] if c in df_show.columns]
            st.dataframe(df_show[cols_to_show].head(MAX_LOG_DISPLAY))
            st.download_button("Download predictions CSV", data=df_to_csv_bytes(df_show), file_name="predictions.csv", mime="text/csv")
        else:
            st.info("No predictions yet")

        st.markdown("---")
        st.markdown("<h4>Contact Messages</h4>", unsafe_allow_html=True)
        contacts = get_contacts(limit=200)
        if contacts:
            for row in contacts:
                resolved = int(row.get("resolved",0) or 0)
                with st.expander(f"{row.get('sender')} — {row.get('created_at')} — {'Resolved' if resolved else 'Open'}"):
                    st.write(row.get("message"))
                    st.write("From user id:", row.get("user_id"), "email:", row.get("email"))
                    key = f"toggle_{row.get('id')}"
                    if st.button(("Mark Open" if resolved else "Mark Resolved"), key=key):
                        toggle_resolve_contact(row.get("id"), not resolved)
                        st.rerun()
        else:
            st.info("No contact messages yet")

        if st.button("Logout (admin)"):
            st.session_state["logged_in"] = False
            st.session_state["user"] = None
            st.session_state["page"] = "home"

# ---------------- USER CHOICE ----------------
elif page == "user_choice":
    if not st.session_state.get("logged_in") or not st.session_state.get("user"):
        st.error("Please login first")
        if st.button("Go to Home"):
            st.session_state["page"] = "home"
    else:
        user = st.session_state["user"]
        st.markdown(f"<div class='card'><h3>Welcome, {user.get('username','User')}</h3><p class='subtitle'>Choose a model</p></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📩 Spam - Ham Classifier"):
                st.session_state["selected_model"] = "spam"
                st.session_state["page"] = "user_model"
        with c2:
            if st.button("📰 News Classification"):
                st.session_state["selected_model"] = "news"
                st.session_state["page"] = "user_model"
        with c3:
            if st.button("🎬 Movie Recommendation"):
                st.session_state["selected_model"] = "movies"
                st.session_state["page"] = "user_model"
        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["user"] = None
            st.session_state["page"] = "home"

# ---------------- USER MODEL PAGE (with replaced Movie UI) ----------------
elif page == "user_model":
    if not st.session_state.get("logged_in") or not st.session_state.get("user"):
        st.error("Please login first")
        if st.button("Go to Home"):
            st.session_state["page"] = "home"
    else:
        user = st.session_state["user"]
        model_selected = st.session_state.get("selected_model", "spam")
        st.markdown(f"<div class='card'><h3>{'📩 Spam - Ham' if model_selected=='spam' else ('📰 News' if model_selected=='news' else '🎬 Movies')} — {user.get('username','User')}</h3><p class='subtitle'>Single & Bulk • History • Analytics</p></div>", unsafe_allow_html=True)

        # MOVIES: replaced with Project-2 Netflix-style UI (exact behavior)
        if model_selected == "movies":
            # load movie artifacts
            df_m, vecs_m, model_m = load_movie_models()
            # Minimal layout similar to Project 2
            st.markdown("<div style='display:flex;gap:12px;align-items:center;'><div style='flex:1'><h3>Movie Recommender — Premium UI</h3><div class='small-muted'>Trending, Details, Recommendations</div></div></div>", unsafe_allow_html=True)

            # Trending block (sample list)
            TRENDING = [
                "Avengers: Endgame","Inception","Interstellar","The Dark Knight","Titanic","Bajrangi Bhaijaan",
                "Avatar","Jurassic Park","Baahubali 2: The Conclusion","The Godfather","Dhoom 3","Forrest Gump","PK",
                "Harry Potter and the Deathly Hallows – Part 2","Joker","Toy Story 4"
            ]
            trending_select = st.selectbox("Select Trending Movie", ["Select"] + TRENDING, key="trending_select_user")
            if trending_select != "Select":
                om = fetch_omdb_by_title(trending_select)
                poster = safe_poster_url(om) or DEFAULT_POSTER
                title = om.get("Title", trending_select)
                year = om.get("Year", "N/A")
                genre = om.get("Genre", "N/A")
                director = om.get("Director", "N/A")
                cast = om.get("Actors", "N/A")
                plot = om.get("Plot", "N/A")
                imdb_rating = om.get("imdbRating", "N/A")
                imdb_link = imdb_url_from_omd(om)

                left, middle, right = st.columns([2,3,1])
                with left:
                    try:
                        st.image(poster, use_container_width=True)
                    except Exception:
                        if os.path.exists(DEFAULT_POSTER):
                            st.image(DEFAULT_POSTER, use_container_width=True)
                with middle:
                    st.markdown(f"### {title} ({year})")
                    st.markdown(f"**Genre:** {genre}  •  **Director:** {director}")
                    st.write(plot)
                with right:
                    st.markdown(
                        f"""
                        <div class='card'>
                            <b>IMDb:</b> {imdb_rating}<br>
                            <b>Cast:</b> {cast}<br><br>
                            <a href='{imdb_link}' target='_blank'>Open on IMDb</a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("---")
                tab_overview, tab_watch, tab_cast, tab_songs, tab_reviews = st.tabs(
                    ["Overview", "Watch Movie", "Cast", "Songs", "Reviews"]
                )
                with tab_overview:
                    st.write(plot)
                with tab_watch:
                    st.markdown(f"- [Trailer on YouTube](https://www.youtube.com/results?search_query={urllib.parse.quote(title + ' trailer')})")
                    st.markdown(f"- [Where to Watch (Google)](https://www.google.com/search?q={urllib.parse.quote(title + ' watch online')})")
                with tab_cast:
                    st.write(cast)
                with tab_songs:
                    st.markdown(f"[Search Songs on YouTube]({yt_songs_link(title)})")
                with tab_reviews:
                    st.markdown(f"[IMDb Reviews]({imdb_link})")

            st.markdown("---")

            # Main Recommender: select from df if available
            if df_m is None or vecs_m is None or model_m is None:
                st.error("Movie model files not found. Place df.pkl, vectors.pkl and model.pkl in app folder.")
            else:
                movie_list = ["Select a movie"] + df_m["name"].tolist()
                selected_movie = st.selectbox("Choose movie", movie_list, key="movie_select_user")
                if selected_movie != "Select a movie":
                    sel_row = df_m[df_m['name'].str.lower() == selected_movie.lower()]
                    if len(sel_row) and sel_row.iloc[0].get("movie_id"):
                        om = fetch_omdb_by_id(sel_row.iloc[0]["movie_id"])
                    else:
                        om = fetch_omdb_by_title(selected_movie)
                    poster = safe_poster_url(om) or DEFAULT_POSTER
                    title = om.get("Title", selected_movie)
                    year = om.get("Year", "N/A")
                    genre = om.get("Genre", "N/A")
                    director = om.get("Director", "N/A")
                    cast = om.get("Actors", "N/A")
                    plot = om.get("Plot", "N/A")
                    imdb_rating = om.get("imdbRating", "N/A")
                    imdb_link = imdb_url_from_omd(om)

                    left, middle, right = st.columns([2,3,1])
                    with left:
                        try:
                            st.image(poster, use_container_width=True)
                        except Exception:
                            if os.path.exists(DEFAULT_POSTER):
                                st.image(DEFAULT_POSTER, use_container_width=True)
                    with middle:
                        st.markdown(f"### {title} ({year})")
                        st.markdown(f"**Genre:** {genre}  •  **Director:** {director}")
                        st.write(plot)
                    with right:
                        st.markdown(
                            f"""
                            <div class='card'>
                                <b>IMDb:</b> {imdb_rating}<br>
                                <b>Cast:</b> {cast}<br><br>
                                <a href='{imdb_link}' target='_blank'>Open on IMDb</a>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("---")
                    tab_overview, tab_watch, tab_cast, tab_songs, tab_reviews = st.tabs(
                        ["Overview", "Watch Movie", "Cast", "Songs", "Reviews"]
                    )
                    with tab_overview:
                        st.write(plot)
                    with tab_watch:
                        st.markdown(f"- [Trailer on YouTube](https://www.youtube.com/results?search_query={urllib.parse.quote(title + ' trailer')})")
                        st.markdown(f"- [Where to Watch (Google)](https://www.google.com/search?q={urllib.parse.quote(title + ' watch online')})")
                    with tab_cast:
                        st.write(cast)
                    with tab_songs:
                        st.markdown(f"[Search Songs on YouTube]({yt_songs_link(title)})")
                    with tab_reviews:
                        st.markdown(f"[IMDb Reviews]({imdb_link})")

                    st.markdown("---")
                    st.header("You may also like")
                    try:
                        base_idx = int(sel_row.index[0])
                        n_nbrs = min(6, len(df_m))
                        dists, idxs = model_m.kneighbors([vecs_m[base_idx]], n_neighbors=n_nbrs)
                        rec_idxs = [i for i in idxs[0] if i != base_idx][:5]
                    except Exception:
                        rec_idxs = []

                    if not rec_idxs:
                        st.info("No recommendations available.")
                    else:
                        cols = st.columns(len(rec_idxs))
                        for i, idx in enumerate(rec_idxs):
                            row = df_m.iloc[idx]
                            om2 = fetch_omdb_by_title(row["name"])
                            poster2 = safe_poster_url(om2) or DEFAULT_POSTER
                            link2 = imdb_url_from_omd(om2)
                            title2 = om2.get("Title", row["name"])
                            year2 = om2.get("Year", "")
                            rating2 = om2.get("imdbRating", "N/A")
                            plot2 = om2.get("Plot", "No summary available.")
                            short_plot = plot2[:200] + ("..." if len(plot2) > 200 else "")
                            with cols[i]:
                                st.markdown(f"<a href='{link2}' target='_blank'><img src='{poster2}' width='100%'></a>", unsafe_allow_html=True)
                                st.markdown(f"**{title2}** ({year2}) ⭐ {rating2}")
                                st.markdown(f"<span class='small-muted'>{short_plot}</span>", unsafe_allow_html=True)

            # Bottom controls
            st.markdown("---")
            if st.button("Back to Model Choice"):
                st.session_state["page"] = "user_choice"
            if st.button("Logout"):
                st.session_state["logged_in"] = False
                st.session_state["user"] = None
                st.session_state["page"] = "home"

        else:
            # existing multi-tab UI for spam & news (unchanged)
            tabs = st.tabs(["Single Predict","Bulk Predict","History","Analytics","Contact Admin"])

            # SINGLE
            with tabs[0]:
                if model_selected == "spam":
                    st.markdown("<h4>Single Spam Predict</h4>", unsafe_allow_html=True)
                    txt = st.text_area("Enter message to classify (spam/ham)", height=160, key="single_spam_txt")
                    st.markdown("<style>.stTextArea textarea { color: #0b0b0b !important; }</style>", unsafe_allow_html=True)

                    if st.button("Predict Spam"):
                        if spam_model is None:
                            st.error("Spam model not loaded. Ask admin to upload.")
                        elif not txt.strip():
                            st.error("Enter text to predict.")
                        else:
                            try:
                                raw = _try_predict(spam_model, [txt])[0]
                                pred = normalize_spam_label(raw)
                            except Exception as e:
                                st.error(f"Model error: {e}")
                                pred = "error"
                            try:
                                log_prediction(user["id"], txt, pred, model_type="spam")
                            except Exception:
                                pass
                            if pred == "spam":
                                st.markdown("<div class='spam-bulk'>🚨 SPAM DETECTED</div>", unsafe_allow_html=True)
                            elif pred == "ham":
                                st.markdown("<div class='ham-bulk'>✔️ HAM</div>", unsafe_allow_html=True)
                            else:
                                st.info(f"Result: {pred}")
                else:
                    st.markdown("<h4>Single News Predict</h4>", unsafe_allow_html=True)
                    txt = st.text_area("Enter headline or paragraph", height=160, key="single_news_txt")
                    st.markdown("<style>.stTextArea textarea { color: #0b0b0b !important; }</style>", unsafe_allow_html=True)
                    if st.button("Predict News"):
                        if news_model is None:
                            st.error("News model not loaded.")
                        elif not txt.strip():
                            st.error("Enter text to predict.")
                        else:
                            try:
                                pred_candidates = []
                                try:
                                    pred_candidates.append(_try_predict(news_model, [txt], is_news=True)[0])
                                except Exception:
                                    pass
                                try:
                                    pred_candidates.append(_try_predict(news_model, [txt], is_news=False)[0])
                                except Exception:
                                    pass
                                try:
                                    raw_obj = news_model.raw if isinstance(news_model, ModelWrapper) else news_model
                                    if hasattr(raw_obj, "predict"):
                                        direct = raw_obj.predict([txt])[0]
                                        pred_candidates.append(direct)
                                except Exception:
                                    pass

                                chosen_raw = None
                                for c in pred_candidates:
                                    if c is None:
                                        continue
                                    cs = str(c).strip()
                                    if cs != "":
                                        chosen_raw = cs
                                        break
                                if chosen_raw is None:
                                    try:
                                        chosen_raw = _try_predict(news_model, [preprocess_news(txt)], is_news=False)[0]
                                    except Exception:
                                        chosen_raw = "NEWS"

                                pred_norm = normalize_news_output(chosen_raw, news_model, canonical_list=NEWS_CANONICAL_CLASSES)

                            except Exception as e:
                                st.error(f"Model error: {e}")
                                pred_norm = "NEWS"
                            try:
                                log_prediction(user["id"], txt, pred_norm, model_type="news")
                            except Exception:
                                pass
                            style = category_style(pred_norm)
                            st.markdown(f"<div style='{style}'>📰 Predicted: <b>{pred_norm}</b></div>", unsafe_allow_html=True)

            # BULK, HISTORY, ANALYTICS, CONTACT (unchanged)...
            with tabs[1]:
                st.markdown("<h4>Bulk Prediction</h4>", unsafe_allow_html=True)
                st.markdown("<div class='small-muted'>Upload CSV (single column) or TXT (one text per line). CSV column names can be anything; first column will be used.</div>", unsafe_allow_html=True)
                uploaded = st.file_uploader("Upload TXT or CSV", type=["txt","csv"], key=f"bulk_up_{model_selected}")

                prefilter = st.selectbox("Prefilter (optional)", ["None","Contains link (http)","Contains numbers","Custom keyword"], key=f"pref_{model_selected}")
                custom_kw = ""
                if prefilter == "Custom keyword":
                    custom_kw = st.text_input("Enter keyword", key=f"pref_kw_{model_selected}")

                if model_selected == "spam":
                    postfilter_choice = st.selectbox("Show only:", ["All","Ham","Spam"], key=f"postfilter_{model_selected}")
                else:
                    news_categories = ["All"] + NEWS_CANONICAL_CLASSES
                    postfilter_choice = st.selectbox("Show only category:", news_categories, key=f"postfilter_{model_selected}")

                if uploaded is not None:
                    try:
                        if uploaded.name.lower().endswith(".csv"):
                            try:
                                df_raw = pd.read_csv(uploaded, dtype=str)
                                if df_raw.shape[1] >= 1:
                                    df = df_raw.iloc[:,0].to_frame(name="Text")
                                else:
                                    df = pd.DataFrame(columns=["Text"])
                            except Exception:
                                uploaded.seek(0)
                                df_raw = pd.read_csv(uploaded, dtype=str, header=None)
                                df = df_raw.iloc[:,0].to_frame(name="Text")
                        else:
                            content = uploaded.read().decode("utf-8").splitlines()
                            df = pd.DataFrame(content, columns=["Text"])
                    except Exception as e:
                        st.error(f"File read error: {e}")
                        st.stop()

                    st.info(f"Loaded {len(df)} texts")

                    if prefilter != "None":
                        if prefilter == "Contains link (http)":
                            df_pref = df[df["Text"].astype(str).str.contains("http", case=False, na=False)]
                        elif prefilter == "Contains numbers":
                            df_pref = df[df["Text"].astype(str).str.contains(r"\d", regex=True, na=False)]
                        elif prefilter == "Custom keyword" and custom_kw.strip():
                            df_pref = df[df["Text"].astype(str).str.contains(custom_kw, case=False, na=False)]
                        else:
                            df_pref = df
                        st.markdown(f"<div class='card'><b>Prefilter:</b> {prefilter} — {len(df_pref)} selected</div>", unsafe_allow_html=True)
                    else:
                        df_pref = df

                    with st.form(f"bulk_form_{model_selected}"):
                        st.write("Preview (first 5 rows):")
                        st.dataframe(df_pref.head(5))
                        run_btn = st.form_submit_button("Run Bulk Predict")

                    if run_btn:
                        chosen_model = spam_model if model_selected=="spam" else news_model
                        model_type = "spam" if model_selected=="spam" else "news"
                        if chosen_model is None:
                            st.error(f"{'Spam' if model_selected=='spam' else 'News'} model not loaded.")
                        else:
                            with st.spinner("Predicting..."):
                                try:
                                    inputs = df_pref["Text"].astype(str).tolist()
                                    is_news_flag = (model_selected != "spam")
                                    preds = chunked_predict(chosen_model, inputs, chunk=2000, is_news=is_news_flag)
                                    preds = ["" if p is None else str(p) for p in preds]

                                    if model_selected=="spam":
                                        preds_norm = [normalize_spam_label(p) for p in preds]
                                    else:
                                        preds_norm = [normalize_news_output(p, news_model, canonical_list=NEWS_CANONICAL_CLASSES) for p in preds]

                                    if len(preds_norm) != len(df_pref):
                                        st.warning(f"Prediction length ({len(preds_norm)}) != rows ({len(df_pref)}). Aligning to min length.")
                                        minlen = min(len(preds_norm), len(df_pref))
                                        preds_norm = preds_norm[:minlen]
                                        preds = preds[:minlen]
                                        df_pref = df_pref.reset_index(drop=True).head(minlen)

                                    df_pref = df_pref.reset_index(drop=True)
                                    df_pref["result_raw"] = pd.Series(preds, index=df_pref.index).astype(str)
                                    if model_selected == "spam":
                                        df_pref["result"] = df_pref["result_raw"].astype(str).str.strip().str.lower()
                                    else:
                                        df_pref["result"] = pd.Series(preds_norm, index=df_pref.index).astype(str).str.strip()

                                    st.session_state["last_bulk_predictions"] = df_pref.copy()

                                    for _, r in df_pref.head(500).iterrows():
                                        try:
                                            log_prediction(user["id"], str(r["Text"]), str(r["result"]), model_type=model_type)
                                        except Exception:
                                            pass

                                    st.success(f"Prediction finished — {len(df_pref)} rows predicted.")
                                except Exception as e:
                                    st.error(f"Model prediction error: {e}")
                                    df_pref["result_raw"] = ""
                                    df_pref["result"] = ""
                                    st.session_state["last_bulk_predictions"] = df_pref.copy()

                    df_predicted = st.session_state.get("last_bulk_predictions", None)
                    if df_predicted is None:
                        st.info("No predictions available yet. Run 'Run Bulk Predict' to generate them.")
                    else:
                        shown = df_predicted.copy()
                        if model_selected == "spam":
                            if postfilter_choice == "Ham":
                                shown = shown[shown["result"].astype(str).str.lower() == "ham"]
                            elif postfilter_choice == "Spam":
                                shown = shown[shown["result"].astype(str).str.lower() == "spam"]
                        else:
                            if postfilter_choice != "All":
                                shown = shown[shown["result"].astype(str) == postfilter_choice]

                        st.write(f"Showing {len(shown)} rows (first 500):")
                        st.dataframe(shown.head(500))

                        if model_selected == "spam":
                            s_count = int((df_predicted["result"].astype(str).str.lower() == "spam").sum()) if "result" in df_predicted.columns else 0
                            h_count = int((df_predicted["result"].astype(str).str.lower() == "ham").sum()) if "result" in df_predicted.columns else 0
                            st.markdown(f"<div style='display:flex;gap:12px;margin-top:8px'><div class='kpi'><h3>🚫 {s_count}</h3><p>Spam</p></div><div class='kpi'><h3>✅ {h_count}</h3><p>Ham</p></div></div>", unsafe_allow_html=True)

                        if not shown.empty:
                            for _, r in shown.head(200).iterrows():
                                rr_raw = str(r.get("result_raw"))
                                rr = str(r.get("result"))
                                text = str(r.get("Text"))
                                if model_selected == "spam":
                                    if rr.lower() == "spam":
                                        st.markdown(f"<div class='spam-bulk'>🚨 SPAM: {text}</div>", unsafe_allow_html=True)
                                    elif rr.lower() == "ham":
                                        st.markdown(f"<div class='ham-bulk'>✔️ HAM: {text}</div>", unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"<div class='card'>{rr}: {text}</div>", unsafe_allow_html=True)
                                else:
                                    style = category_style(rr)
                                    snippet = (text[:300] + "...") if len(text) > 300 else text
                                    st.markdown(f"<div style='{style}'><b>{rr}</b>: {snippet}</div>", unsafe_allow_html=True)

                            try:
                                csv_bytes = df_to_csv_bytes(shown)
                                st.download_button("📥 Download results (CSV)", data=csv_bytes, file_name="bulk_results.csv", mime="text/csv")
                            except Exception as e:
                                st.error("Export failed: " + str(e))
                        else:
                            st.info("No rows to show after filter.")

            # HISTORY
            with tabs[2]:
                st.markdown("<h4>Your Prediction History</h4>", unsafe_allow_html=True)
                preds = get_user_predictions(user["id"], limit=2000)
                if preds:
                    dfh = pd.DataFrame(preds)
                    display_cols = [c for c in ["created_at","model","message","result"] if c in dfh.columns]
                    st.dataframe(dfh[display_cols].rename(columns={"created_at":"At","message":"Message","result":"Result","model":"Model"}))
                    if st.button("Export History"):
                        csv = dfh.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 Download history (CSV)", data=csv, file_name="history.csv", mime="text/csv")
                else:
                    st.info("No history yet — try a prediction!")

            # ANALYTICS
            with tabs[3]:
                st.markdown("<h4>Analytics</h4>", unsafe_allow_html=True)
                try:
                    preds = get_user_predictions(user["id"], limit=5000)
                    dfp = pd.DataFrame(preds) if preds else pd.DataFrame(columns=["created_at","model","message","result"])
                    if dfp.empty:
                        st.info("No activity yet.")
                    else:
                        dfp_sel = dfp[dfp["model"]==model_selected] if "model" in dfp.columns else dfp
                        total = len(dfp_sel)
                        spam_count = int((dfp_sel["result"].astype(str).str.lower()=="spam").sum()) if not dfp_sel.empty else 0
                        ham_count = int((dfp_sel["result"].astype(str).str.lower()=="ham").sum()) if not dfp_sel.empty else 0
                        error_count = int((dfp_sel["result"].astype(str).str.lower()=="error").sum()) if not dfp_sel.empty else 0
                        c1,c2,c3,c4 = st.columns([2,2,2,2])
                        c1.markdown(f"<div class='kpi'><h3>{total}</h3><p>Total Predictions</p></div>", unsafe_allow_html=True)
                        c2.markdown(f"<div class='kpi'><h3>🚫 {spam_count}</h3><p>Spam</p></div>", unsafe_allow_html=True)
                        c3.markdown(f"<div class='kpi'><h3>✅ {ham_count}</h3><p>Ham</p></div>", unsafe_allow_html=True)
                        c4.markdown(f"<div class='kpi'><h3>⚠️ {error_count}</h3><p>Errors</p></div>", unsafe_allow_html=True)

                        try:
                            dfp_sel["date_only"] = pd.to_datetime(dfp_sel["created_at"]).dt.date
                            daily = dfp_sel.groupby("date_only").size().reset_index(name="count")
                            fig, ax = plt.subplots()
                            ax.plot(daily["date_only"].astype(str), daily["count"], marker="o")
                            ax.set_title("Predictions over time")
                            ax.set_xlabel("Date")
                            ax.set_ylabel("Count")
                            plt.xticks(rotation=30)
                            st.pyplot(fig)
                        except Exception:
                            st.info("Could not build timeline chart.")

                        try:
                            if model_selected == "spam":
                                labels = []
                                counts = []
                                if ham_count>0:
                                    labels.append("Ham"); counts.append(ham_count)
                                if spam_count>0:
                                    labels.append("Spam"); counts.append(spam_count)
                                if counts:
                                    fig2, ax2 = plt.subplots()
                                    ax2.pie(counts, labels=labels, autopct="%1.0f%%", startangle=90)
                                    st.pyplot(fig2)
                                else:
                                    st.info("No results to chart yet.")
                            else:
                                vc = dfp_sel["result"].value_counts().head(10)
                                if not vc.empty:
                                    fig3, ax3 = plt.subplots()
                                    ax3.bar(vc.index.astype(str), vc.values)
                                    ax3.set_xticklabels(vc.index.astype(str), rotation=30, ha="right")
                                    ax3.set_title("Top predicted categories")
                                    st.pyplot(fig3)
                                else:
                                    st.info("No categories to chart yet.")
                        except Exception:
                            st.info("Could not build distribution chart.")
                except Exception as e:
                    st.error(f"Analytics failed: {e}")

            # CONTACT
            with tabs[4]:
                st.markdown("<h4>Contact Admin</h4>", unsafe_allow_html=True)
                subject = st.text_input("Subject", key=f"contact_subject_{user['id']}")
                message = st.text_area("Describe your problem or feedback", key=f"contact_message_{user['id']}", height=140)
                if st.button("Send Message"):
                    if not subject.strip() and not message.strip():
                        st.error("Please add a message")
                    else:
                        send_contact(user["id"], subject or user.get("username","user"), user.get("email",""), message)
                        st.success("Message sent to admin")
                        try:
                            st.balloons()
                        except Exception:
                            pass

            if st.button("Back to Model Choice"):
                st.session_state["page"] = "user_choice"
            if st.button("Logout"):
                st.session_state["logged_in"] = False
                st.session_state["user"] = None
                st.session_state["page"] = "home"

# FALLBACK
else:
    st.write("Unknown page — returning home")
    st.session_state["page"] = "home"

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:rgba(255,255,255,0.75)'>Developed ❤️ by SM Organization</div>", unsafe_allow_html=True)

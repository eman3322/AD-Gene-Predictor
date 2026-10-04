import base64
import html
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go_              # ADDED
from streamlit_searchbox import st_searchbox    # ADDED
st.set_page_config(page_title="AD Gene Predictor", page_icon=":dna:", layout="wide")

ROOT = Path(__file__).parent
TABLE = ROOT / "results" / "app_gene_table.csv"
LIT = ROOT / "results" / "novel_literature_counts.csv"
PINK, TEAL, NAVY = "#E0457B", "#0B8FB0", "#12395F"


# ----------------------------------------------------------------- data
import utils

df = utils.load_table()
GENES = sorted(df["gene"].astype(str).unique())
GENESET = set(GENES)
N_TOTAL, N_POS, N_NEG = len(df), int((df.label == 1).sum()), int((df.label == 0).sum())


def neighbours_of(row):
    return [x.strip() for x in str(row["known_AD_neighbours"]).split(",") if x.strip()]


def open_gene(g):
    st.session_state["gene"] = g
    st.switch_page(P_GENE)


# ----------------------------------------------------------------- pictures
def network_svg(w, h, n, seed, edge, node, hub, cand, edge_op, node_op):
    rng = random.Random(seed)
    pts = [(rng.uniform(0, w), rng.uniform(0, h)) for _ in range(n)]
    edges, near = set(), {}
    for i, (x, y) in enumerate(pts):
        order = sorted(range(n), key=lambda j: (pts[j][0] - x) ** 2 + (pts[j][1] - y) ** 2)[1:4]
        near[i] = order
        for j in order[:2]:
            edges.add((min(i, j), max(i, j)))
    hubs = rng.sample(range(n), 7)
    cands = {near[hb][0] for hb in hubs} | {near[hb][1] for hb in hubs[:4]}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid slice">']
    for i, j in edges:
        strong = (i in hubs and j in cands) or (j in hubs and i in cands)
        col, op, sw = (cand, 0.8, 1.8) if strong else (edge, edge_op, 1)
        out.append(f'<line x1="{pts[i][0]:.0f}" y1="{pts[i][1]:.0f}" x2="{pts[j][0]:.0f}" '
                   f'y2="{pts[j][1]:.0f}" stroke="{col}" stroke-opacity="{op}" stroke-width="{sw}"/>')
    for i, (x, y) in enumerate(pts):
        if i in hubs:
            out.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="16" fill="{hub}" fill-opacity="0.18"/>'
                       f'<circle cx="{x:.0f}" cy="{y:.0f}" r="7" fill="{hub}"/>')
        elif i in cands:
            out.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="5.5" fill="{cand}"/>')
        else:
            out.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="3" fill="{node}" fill-opacity="{node_op}"/>')
    out.append("</svg>")
    return "data:image/svg+xml;base64," + base64.b64encode("".join(out).encode()).decode()


def image_uri(path):
    ext = path.suffix.lower().lstrip(".").replace("jpg", "jpeg")
    return f"data:image/{ext};base64," + base64.b64encode(path.read_bytes()).decode()


def star_svg(center, neigh, center_known):
    w, h, cx, cy = 640, 330, 320, 165
    n = len(neigh)
    pos = []
    for i in range(n):
        a = 2 * math.pi * i / n - math.pi / 2
        pos.append((cx + 215 * math.cos(a), cy + 112 * math.sin(a)))
    s = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:660px;height:auto">']
    for x, y in pos:
        s.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.0f}" y2="{y:.0f}" stroke="{TEAL}" stroke-opacity="0.55" stroke-width="2"/>')
    for (x, y), g in zip(pos, neigh):
        ly = y - 20 if y < cy else y + 28
        s.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="20" fill="{PINK}" fill-opacity="0.16"/>'
                 f'<circle cx="{x:.0f}" cy="{y:.0f}" r="11" fill="{PINK}"/>'
                 f'<text x="{x:.0f}" y="{ly:.0f}" font-size="14" text-anchor="middle" fill="{NAVY}">{g}</text>')
    c = PINK if center_known else TEAL
    s.append(f'<circle cx="{cx}" cy="{cy}" r="38" fill="{c}" fill-opacity="0.16"/>'
             f'<circle cx="{cx}" cy="{cy}" r="24" fill="{c}"/>'
             f'<text x="{cx}" y="{cy + 50}" font-size="16" font-weight="600" text-anchor="middle" fill="{NAVY}">{center}</text>')
    s.append("</svg>")
    return "".join(s)


def dna_svg(w=280, h=210):
    n = 60
    p1, p2, rungs = [], [], []
    for i in range(n + 1):
        t = i / n
        y = 10 + t * (h - 20)
        dx = 72 * math.sin(t * 4 * math.pi)
        p1.append(f"{w / 2 + dx:.1f},{y:.1f}")
        p2.append(f"{w / 2 - dx:.1f},{y:.1f}")
        if i % 3 == 0:
            col = PINK if i % 15 == 0 else "#9CC0DE"
            rungs.append(f'<line x1="{w / 2 + dx:.1f}" y1="{y:.1f}" x2="{w / 2 - dx:.1f}" y2="{y:.1f}" '
                         f'stroke="{col}" stroke-width="2"/>')
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">'
            + "".join(rungs)
            + f'<polyline points="{" ".join(p1)}" fill="none" stroke="#0B6BCB" stroke-width="4" stroke-linecap="round"/>'
            + f'<polyline points="{" ".join(p2)}" fill="none" stroke="{TEAL}" stroke-width="4" stroke-linecap="round"/></svg>')


def icon_network():
    pts = [(20, 18), (60, 10), (100, 22), (35, 52), (85, 55), (60, 34)]
    ed = [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (0, 1), (2, 4), (3, 0)]
    out = ['<svg viewBox="0 0 120 70" xmlns="http://www.w3.org/2000/svg" style="width:96px;height:auto">']
    out += [f'<line x1="{pts[i][0]}" y1="{pts[i][1]}" x2="{pts[j][0]}" y2="{pts[j][1]}" stroke="{TEAL}" stroke-opacity=".6" stroke-width="2"/>' for i, j in ed]
    out += [f'<circle cx="{x}" cy="{y}" r="5" fill="{TEAL}"/>' for x, y in pts[:5]]
    out.append(f'<circle cx="60" cy="34" r="9" fill="{PINK}"/></svg>')
    return "".join(out)


def icon_heat():
    rng = random.Random(3)
    out = ['<svg viewBox="0 0 120 70" xmlns="http://www.w3.org/2000/svg" style="width:96px;height:auto">']
    for r_ in range(5):
        for c_ in range(10):
            col = PINK if r_ == 2 and c_ > 5 else "#0B6BCB"
            out.append(f'<rect x="{6 + c_ * 11}" y="{6 + r_ * 12}" width="9" height="10" rx="2" fill="{col}" fill-opacity="{rng.uniform(.15, .95):.2f}"/>')
    out.append("</svg>")
    return "".join(out)


def icon_roc():
    return ('<svg viewBox="0 0 120 70" xmlns="http://www.w3.org/2000/svg" style="width:96px;height:auto">'
            '<line x1="10" y1="62" x2="110" y2="62" stroke="#7FA6C9" stroke-width="2"/>'
            '<line x1="10" y1="62" x2="10" y2="6" stroke="#7FA6C9" stroke-width="2"/>'
            '<line x1="10" y1="62" x2="110" y2="8" stroke="#9CB4CC" stroke-width="1.5" stroke-dasharray="4 4"/>'
            f'<path d="M10 62 C 22 36, 48 22, 110 8" fill="none" stroke="{PINK}" stroke-width="3.5" stroke-linecap="round"/></svg>')


custom = next((p for p in (ROOT / "assets").glob("hero.*")), None) if (ROOT / "assets").exists() else None
HERO_IMG = image_uri(custom) if custom else network_svg(
    1400, 300, 85, 11, "#5B8DB8", "#5B8DB8", PINK, TEAL, 0.38, 0.6)
PAGE_IMG = network_svg(1600, 900, 70, 5, "#2A6FA8", "#2A6FA8", "#7FA6C9", "#7FA6C9", 0.09, 0.12)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600&family=Source+Sans+3:wght@400;600&display=swap');
.stApp, .stApp p, .stApp label, .stApp li, .stApp td, .stApp th, .stApp button, .stApp input, .stApp textarea {
  font-family: 'Source Sans 3', 'Segoe UI', sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons {
  font-family: 'Material Symbols Rounded', 'Material Icons' !important; }
.stApp { background: url("%%PAGE%%") center / cover fixed no-repeat, #E8F1F9; }
footer { visibility: hidden; }
[data-testid="stAppDeployButton"], .stDeployButton { display: none; }
.block-container { padding-top: 4.2rem; max-width: 1180px; }
h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif; color: #12395F; }

.st-key-searchbar {
  background: linear-gradient(100deg, #0F3257 0%, #1A5A94 100%);
  border-radius: 12px; padding: 20px 26px 12px; margin-bottom: 20px;
}
.brand { color: #FFFFFF; font-family: 'Source Serif 4', Georgia, serif; font-size: 1.7rem; line-height: 1.2; }
.brand small { display: block; color: #B8D4EA; font-family: 'Source Sans 3', sans-serif; font-size: .95rem; margin: 2px 0 12px; }

.hero {
  border-radius: 12px; padding: 30px 34px 24px; margin-bottom: 20px; border: 1px solid #C9DCEB;
  background:
    linear-gradient(100deg, rgba(255,255,255,.97) 0%, rgba(255,255,255,.84) 55%, rgba(255,255,255,.30) 100%),
    url("%%HERO%%") center / cover no-repeat, #FFFFFF;
}
.hero h1 { color: #12395F !important; font-size: 2.1rem; line-height: 1.2; margin: 0 0 8px; padding: 0; }
.hero p { color: #3F5A75; font-size: 1.05rem; max-width: 640px; margin: 0 0 4px; }
.chip {
  display: inline-block; border: 1px solid #B7D0E4; color: #2A5A86; background: rgba(255,255,255,.75);
  border-radius: 999px; padding: 3px 13px; font-size: .84rem; margin: 12px 8px 0 0;
}
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 7px; }

[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; border-color: #C9DCEB !important; border-radius: 10px; }
.sect { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.15rem; color: #12395F;
        border-bottom: 2px solid #0B6BCB; padding-bottom: 4px; margin: 0 0 12px; }
.kv { width: 100%; border-collapse: collapse; }
.kv th { text-align: left; width: 230px; color: #4A6580; font-weight: 600; padding: 7px 8px; border-bottom: 1px solid #E1ECF5; }
.kv td { padding: 7px 8px; border-bottom: 1px solid #E1ECF5; color: #12395F; }
.badge { display: inline-block; border-radius: 6px; padding: 2px 11px; font-size: .9rem; font-weight: 600; margin-right: 8px; }
.b-pos { background: #FBE3EC; color: #A3234F; }
.b-cand { background: #DDF1F5; color: #0B6A82; }
.b-strong { background: #DDF1E4; color: #1D6B3A; }
.b-mid { background: #FFF1D6; color: #8A5A00; }
.b-weak { background: #ECEFF3; color: #4A5D78; }
.evid li { margin-bottom: 6px; }

[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #C9DCEB; border-left: 4px solid #0B6BCB;
                           border-radius: 10px; padding: 14px 18px; }
[data-testid="stMetricLabel"] { color: #4A6580; }
[data-testid="stDataFrame"] { border: 1px solid #C9DCEB; border-radius: 10px; background: #FFFFFF; }
.stButton button, .stLinkButton a { border-radius: 8px; border: 1px solid #0B6BCB; color: #0B5CAD; background: #FFFFFF; }
.stButton button:hover, .stLinkButton a:hover { background: #E3EFFB; border-color: #0B5CAD; color: #084A8C; }

.footer { text-align: center; color: #4A6580; font-size: .85rem; margin: 28px 0 6px; padding-top: 12px; border-top: 1px solid #C9DCEB; }
.qsum { display: flex; flex-wrap: wrap; gap: 8px 34px; background: #FFFFFF; border: 1px solid #C9DCEB;
        border-radius: 10px; padding: 12px 20px; margin: 4px 0 16px; }
.qsum span { display: block; color: #4A6580; font-size: .8rem; }
.qsum b { color: #12395F; font-size: 1.02rem; }
button[data-baseweb="tab"] { font-weight: 600; color: #4A6580; }
button[data-baseweb="tab"][aria-selected="true"] { color: #0B5CAD; }
[data-baseweb="tab-highlight"] { background-color: #0B6BCB !important; }
.hero { display: flex; align-items: center; justify-content: space-between; gap: 20px; }
.hero-art { flex: 0 0 280px; }
@media (max-width: 900px) { .hero-art { display: none; } }
.st-key-searchbar [data-testid="stExpander"] { background: #FFFFFF; border: 0; border-radius: 8px; }
.st-key-searchbar button { background: #0B6BCB; border: 0; color: #FFFFFF; font-weight: 600; width: 100%; }
.st-key-searchbar button:hover { background: #084A8C; color: #FFFFFF; }
.st-key-searchbar button p { color: #FFFFFF; }
.flow { display: flex; align-items: stretch; gap: 8px; flex-wrap: wrap; }
.step { flex: 1 1 140px; border: 1px solid #C9DCEB; border-top: 4px solid #0B6BCB; border-radius: 8px;
        padding: 10px 12px; background: #F7FBFE; }
.step b { display: block; color: #12395F; }
.step span { color: #4A6580; font-size: .9rem; }
.arrow { align-self: center; color: #0B6BCB; font-size: 1.4rem; }
.cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
@media (max-width: 800px) { .cards { grid-template-columns: 1fr; } }
.card { border: 1px solid #C9DCEB; border-radius: 10px; padding: 14px 16px; background: #F7FBFE; }
.card h4 { margin: 8px 0 4px; color: #12395F; font-family: 'Source Serif 4', Georgia, serif; }
.card p { margin: 0; color: #3F5A75; font-size: .95rem; }
.gname { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.35rem; color: #12395F; }
.gsub { color: #4A6580; font-size: .9rem; margin-bottom: 6px; }
.bar { height: 6px; background: #E1ECF5; border-radius: 3px; margin-bottom: 10px; }
.bar i { display: block; height: 6px; background: #0B6BCB; border-radius: 3px; }
</style>
""".replace("%%PAGE%%", PAGE_IMG).replace("%%HERO%%", HERO_IMG)
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------- shared search bar
# CHANGED: NCBI-style autocomplete search bar
@st.cache_data
def _upper_genes():
    return [(g, g.upper()) for g in GENES]


def suggest(term: str):
    t = term.strip().upper()
    if not t:
        ex = [g for g in ["APOE", "MS4A8", "SCIMP", "NFAM1", "ZCWPW1", "ITPKB"] if g in GENESET]
        return ex + [g for g in GENES if g not in ex][:44]
    pairs = _upper_genes()
    starts = [g for g, u in pairs if u.startswith(t)]
    has = [g for g, u in pairs if t in u and not u.startswith(t)]
    out = (starts + has)[:10]
    if t.isdigit():
        out = [term.strip()] + out[:9]
    return out


def search_bar():
    cur = st.session_state.get("gene")
    with st.container(key="searchbar"):
        st.markdown('<div class="brand">AD Gene Predictor<small>Alzheimer\'s gene prioritisation from the '
                    'protein interaction network</small></div>', unsafe_allow_html=True)
        c1, c2 = st.columns([6, 1])
        with c1:
            choice = st_searchbox(
                suggest, key="gene_search",
                placeholder="Type a gene symbol or NCBI Gene ID (e.g. APOE, MS4A8, 7157)",
                default=cur if cur in GENESET else None,
                edit_after_submit="option",      # selected gene stays in the bar
                style_overrides={"searchbox": {
                    "control": {"backgroundColor": "#FFFFFF"},
                    "singleValue": {"color": "#12395F"},
                    "input": {"color": "#12395F"},
                    "placeholder": {"color": "#6B8399"},
                    "option": {"color": "#12395F"}}})
        go = c2.button("Predict", key="predict_btn")
        with st.expander("Advanced parameters"):
            c3, c4 = st.columns(2)
            c3.slider("Genes shown in ranking context (top-N)", 5, 50, 15, key="top_n")
            c4.slider("Minimum score in ranking context", 0.0, 1.0, 0.0, 0.05, key="min_score")
            st.radio("Compare the score against", ["All scored genes", "Candidate genes only"], horizontal=True, key="ref_set")
            st.radio("Chart axis", ["Model score", "Percentile"], horizontal=True, key="axis_mode")
    if go:
        if not choice:
            st.session_state["search_err"] = ("Please type a gene and pick one from the suggestions.", [], 0)
        else:
            with st.spinner("Looking up the gene and reading its scores..."):
                sym, err, sugg = utils.resolve_query(choice, GENES)
            if sym:
                st.session_state.pop("search_err", None)
                open_gene(sym)
            else:
                st.session_state["search_err"] = (err, sugg, 0)
    err = st.session_state.get("search_err")
    if err:
        st.error(err[0])
        if err[1]:
            st.write("Did you mean:")
            for col, sg in zip(st.columns(len(err[1])), err[1]):
                if col.button(sg, key=f"sug_{sg}"):
                    st.session_state.pop("search_err", None)
                    open_gene(sg)
        if err[2] >= 1:
            st.session_state.pop("search_err", None)
        else:
            st.session_state["search_err"] = (err[0], err[1], 1)


# ----------------------------------------------------------------- pages
# ===================== ADDED: plotly charts =====================
def _layout(fig, h=300):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=NAVY))
    return fig


def gauge_chart(score, pct):
    fig = go_.Figure(go_.Indicator(
        mode="gauge+number", value=float(score),
        number={"valueformat": ".3f"},
        title={"text": f"AD score (higher than {pct:.0f}% of genes)"},
        gauge={"axis": {"range": [0, 1]}, "bar": {"color": PINK},
               "steps": [{"range": [0, .33], "color": "#E8F1F9"},
                         {"range": [.33, .66], "color": "#CFE2F3"},
                         {"range": [.66, 1], "color": "#B7D0E4"}]}))
    return _layout(fig, 280)


def hist_chart(df_, gene, score):
    fig = go_.Figure(go_.Histogram(x=df_["score"], nbinsx=50, marker_color=TEAL, opacity=0.85))
    fig.add_vline(x=float(score), line_color=PINK, line_width=3,
                  annotation_text=gene, annotation_font_color=PINK)
    fig.update_layout(title="Score distribution of all genes",
                      xaxis_title="Model score", yaxis_title="Number of genes")
    return _layout(fig, 340)


def partners_chart(df_, nb):
    d = df_[df_["gene"].isin(nb)].sort_values("score", ascending=False).head(10)
    d = d.iloc[::-1]
    fig = go_.Figure(go_.Bar(x=d["score"], y=d["gene"], orientation="h",
                             marker_color=[PINK if l == 1 else TEAL for l in d["label"]]))
    fig.update_layout(title="Top interacting partners (red = known AD, blue = candidate)",
                      xaxis_title="Model score")
    return _layout(fig, 340)


def network_chart(gene, nb, known, df_):
    nb = nb[:12]
    n = max(len(nb), 1)
    xs = [math.cos(2 * math.pi * i / n) for i in range(n)]
    ys = [math.sin(2 * math.pi * i / n) for i in range(n)]
    lab = df_.set_index("gene")["label"].to_dict()
    fig = go_.Figure()
    for x, y in zip(xs, ys):
        fig.add_trace(go_.Scatter(x=[0, x], y=[0, y], mode="lines",
                                  line=dict(color="#9CC0DE", width=2),
                                  hoverinfo="skip", showlegend=False))
    fig.add_trace(go_.Scatter(
        x=xs, y=ys, mode="markers+text", text=nb, textposition="top center",
        marker=dict(size=20, color=[PINK if lab.get(x_) == 1 else TEAL for x_ in nb]),
        showlegend=False))
    fig.add_trace(go_.Scatter(
        x=[0], y=[0], mode="markers+text", text=[gene], textposition="bottom center",
        marker=dict(size=34, color=PINK if known else TEAL), showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title="Interaction neighbourhood")
    return _layout(fig, 420)


def ranked_chart(d):
    d = d.sort_values("score", ascending=False).head(20).iloc[::-1]
    fig = go_.Figure(go_.Bar(x=d["score"], y=d["gene"], orientation="h",
                             marker_color=[PINK if l == 1 else TEAL for l in d["label"]]))
    fig.update_layout(title="Top 20 genes by score", xaxis_title="Model score")
    return _layout(fig, 560)
# ===================== end of ADDED =====================

# ===================== ADDED: PubMed, brief, extra charts =====================
@st.cache_data(show_spinner=False, ttl=86400)
def _pubmed_fetch(symbol, n=8):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    term = f"{symbol}[Title/Abstract] AND Alzheimer*[Title/Abstract]"
    res = requests.get(base + "esearch.fcgi", timeout=8, params={
        "db": "pubmed", "term": term, "retmax": n, "sort": "relevance", "retmode": "json"}).json()["esearchresult"]
    ids, total = res["idlist"], int(res["count"])
    if not ids:
        return total, []
    summ = requests.get(base + "esummary.fcgi", timeout=8, params={
        "db": "pubmed", "id": ",".join(ids), "retmode": "json"}).json()["result"]
    out = []
    for i in ids:
        d = summ[i]
        out.append({"pmid": i,
                    "title": re.sub(r"<[^>]+>", "", d.get("title", "")),
                    "authors": ", ".join(a["name"] for a in d.get("authors", [])[:3]),
                    "year": d.get("pubdate", "")[:4], "journal": d.get("source", "")})
    return total, out


def pubmed_articles(symbol, n=8):
    try:
        return _pubmed_fetch(symbol, n)
    except Exception:
        return None, []


def make_brief(g, r, nb, known, conf):
    kind = "a known Alzheimer's gene" if known else "a candidate gene"
    s = (f"**{g}** is {kind} with a model score of {r['score']:.3f}. It ranks {int(r['rank']):,} of "
         f"{N_TOTAL:,} genes (percentile {r['score_pct']:.1f}, confidence: {conf}). "
         f"It has {int(r['degree'])} interaction partners in STRING and {r['nb_frac']:.0%} of them are known AD genes")
    s += f" (for example {', '.join(nb[:5])}). " if nb else ". "
    if r["p_adj"] < 0.05:
        s += f"In brain expression data it differs between AD and control (log2FC {r['log2fc']:+.2f}, adjusted p {r['p_adj']:.1e}). "
    else:
        s += "Brain expression shows no significant change. "
    if not known:
        s += "This is a network-based hypothesis, not proof of association."
    return s


def answer_question(q, g, r, nb, known, conf):
    q = q.lower()
    if any(k in q for k in ["rank", "score", "high", "low"]):
        return (f"{g} scores {r['score']:.3f}, rank {int(r['rank']):,} of {N_TOTAL:,}. The score mainly reflects "
                f"how close it sits to known AD genes in the STRING network ({r['nb_frac']:.0%} of its "
                f"{int(r['degree'])} neighbours are known AD genes).")
    if any(k in q for k in ["neighbour", "neighbor", "partner", "interact", "network"]):
        return f"Known AD neighbours of {g}: " + (", ".join(nb) if nb else "none listed.")
    if any(k in q for k in ["express", "log2", "fold", "p-value", "pvalue", "brain"]):
        return f"log2FC {r['log2fc']:+.2f}, adjusted p {r['p_adj']:.2e} across the brain expression datasets."
    if any(k in q for k in ["confiden", "reliable", "trust"]):
        return f"Confidence band is {conf}. It is rank-based (percentile {r['score_pct']:.1f}), not a probability."
    if any(k in q for k in ["known", "candidate", "status"]):
        return f"{g} is {'a known AD gene' if known else 'a candidate gene, not yet labelled as an AD gene'}."
    return "I can answer about this gene's rank and score, neighbours, expression, confidence and status."


def volcano_chart(df_, gene):
    d = df_.dropna(subset=["log2fc", "p_adj"]).copy()
    d["nl"] = -np.log10(d["p_adj"].clip(lower=1e-300))
    fig = go_.Figure(go_.Scattergl(x=d["log2fc"], y=d["nl"], mode="markers", text=d["gene"],
                                   marker=dict(size=4, color="#9CC0DE")))
    s = d[d["gene"] == gene]
    if len(s):
        fig.add_trace(go_.Scatter(x=s["log2fc"], y=s["nl"], mode="markers+text", text=[gene],
                                  textposition="top center", marker=dict(size=14, color=PINK)))
    fig.add_hline(y=-math.log10(0.05), line_dash="dash", line_color="#9CB4CC")
    fig.update_layout(title="Volcano plot: AD vs control expression", showlegend=False,
                      xaxis_title="log2 fold change", yaxis_title="-log10 adjusted p")
    return _layout(fig, 380)


def scatter_chart(df_, gene):
    fig = go_.Figure(go_.Scattergl(x=df_["nb_frac"], y=df_["score"], mode="markers", text=df_["gene"],
                                   marker=dict(size=4, color=[PINK if l == 1 else TEAL for l in df_["label"]],
                                               opacity=0.6)))
    s = df_[df_["gene"] == gene]
    if len(s):
        fig.add_trace(go_.Scatter(x=s["nb_frac"], y=s["score"], mode="markers+text", text=[gene],
                                  textposition="top center",
                                  marker=dict(size=15, color="#12395F", symbol="diamond")))
    fig.update_layout(title="Score vs known-AD neighbour fraction", showlegend=False,
                      xaxis_title="Neighbour evidence", yaxis_title="Model score")
    return _layout(fig, 380)
# ===================== end of ADDED =====================


def render_pubmed(g):
    with st.container(border=True):
        st.markdown('<div class="sect">PubMed: Alzheimer\'s literature</div>', unsafe_allow_html=True)
        if st.button("Load PubMed papers", key=f"pm_btn_{g}"):
            with st.spinner("Searching PubMed..."):
                st.session_state[f"pm_{g}"] = pubmed_articles(g)
        res = st.session_state.get(f"pm_{g}")
        if res:
            total, arts = res
            if total is None:
                st.warning("Could not reach PubMed. Check the internet connection and try again.")
            elif not arts:
                st.info(f"No PubMed papers found linking {g} with Alzheimer's disease.")
            else:
                st.caption(f"{total:,} papers mention {g} with Alzheimer's. Top {len(arts)} by relevance:")
                for a in arts:
                    st.markdown(f"**{a['title']}**  \n{a['authors']} · {a['journal']} · {a['year']} · PMID {a['pmid']}  \n"
                                f"[View on PubMed](https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/)")
                    st.divider()


def render_brief(g, r, nb, known, conf):
    with st.container(border=True):
        st.markdown('<div class="sect">Gene brief</div>', unsafe_allow_html=True)
        st.write(make_brief(g, r, nb, known, conf))
        st.caption("Generated from this app's own data, not from a chatbot.")
        q = st.text_input("Ask about this gene", placeholder="For example: why is the rank high?", key="ask_q")
        if q:
            st.info(answer_question(q, g, r, nb, known, conf))


def home():
    st.markdown(
        f"""
<div class="hero">
  <div class="hero-text">
    <h1>Find where a gene sits in the Alzheimer's network</h1>
    <p>Search a gene above to open its report: model score, rank, known AD neighbours and
    expression evidence.</p>
    <span class="chip"><span class="dot" style="background:{PINK}"></span>Known AD gene</span>
    <span class="chip"><span class="dot" style="background:{TEAL}"></span>Candidate gene</span>
    <span class="chip">{N_TOTAL:,} genes scored</span>
    <span class="chip">STRING v12.5 + Open Targets</span>
  </div>
  <div class="hero-art">{dna_svg()}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    with st.expander("How it works"):
        st.markdown(
            "1. Type a gene symbol or NCBI Gene ID in the search bar and press **Predict**.\n"
            "2. The tool reads the gene's model score, rank and known Alzheimer's neighbours, and fetches "
            "its description, chromosome, aliases and links from NCBI Gene.\n"
            "3. Open the tabs for the neighbourhood picture, the genes ranked around it, and the model features. "
            "Download the result as CSV or PDF.")
    ex = [g for g in ["MS4A8", "SCIMP", "NFAM1", "ZCWPW1", "ITPKB"] if g in GENESET]
    cols = st.columns(len(ex) + 3)
    cols[0].markdown("Try an example:")
    for col, g in zip(cols[1:], ex):
        if col.button(g, key=f"ex_{g}"):
            open_gene(g)

    st.write("")
    a, b, c = st.columns(3)
    a.metric("Genes scored", f"{N_TOTAL:,}")
    b.metric("Known AD genes", f"{N_POS:,}")
    c.metric("Candidate genes", f"{N_NEG:,}")

    st.write("")
    with st.container(border=True):
        st.markdown('<div class="sect">From data to gene report</div>', unsafe_allow_html=True)
        steps = [("Brain expression", "9 GEO and Kaggle cohorts"), ("Interaction network", "STRING v12.5"),
                 ("Gene labels", "Open Targets"), ("Model", "Network propagation and logistic regression"),
                 ("Gene report", "Score, rank, neighbours")]
        st.markdown('<div class="flow">' + '<div class="arrow">&#8594;</div>'.join(
            f'<div class="step"><b>{t}</b><span>{d}</span></div>' for t, d in steps) + "</div>", unsafe_allow_html=True)

    st.write("")
    with st.container(border=True):
        st.markdown('<div class="sect">How a gene is scored</div>', unsafe_allow_html=True)
        cards = [(icon_network(), "Network proximity",
                  "A random walk starts from known AD genes and spreads through the STRING network. "
                  "Genes it reaches often score higher."),
                 (icon_heat(), "Brain expression",
                  "Differential expression across 9 brain datasets, corrected for study and brain region, "
                  "adds a smaller signal."),
                 (icon_roc(), "Held-out check",
                  "The model was scored once on genes it never saw (AUROC about 0.68). "
                  "Expression alone stayed near chance.")]
        st.markdown('<div class="cards">' + "".join(
            f'<div class="card">{ic}<h4>{t}</h4><p>{d}</p></div>' for ic, t, d in cards) + "</div>",
                    unsafe_allow_html=True)

    st.write("")
    st.markdown("### Top-ranked candidates")
    top = df[df.label == 0].sort_values("score", ascending=False).head(4)
    for col, (_, t) in zip(st.columns(4), top.iterrows()):
        with col.container(border=True):
            st.markdown(f'<div class="gname">{t["gene"]}</div>'
                        f'<div class="gsub">score {t["score"]:.3f}, rank {int(t["novel_rank"])}</div>'
                        f'<div class="bar"><i style="width:{t["score"] * 100:.0f}%"></i></div>', unsafe_allow_html=True)
            if st.button("Open report", key=f"top_{t['gene']}"):
                open_gene(t["gene"])
    st.caption("High rank means close to known AD genes in the network. It is a hypothesis, not a finding.")


def gene_page():
    g = st.session_state.get("gene")
    if not g or g not in GENESET:
        st.info("Use the search bar above to open a gene report.")
        return

    r = utils.predict_gene(df, g)
    known = r["label"] == 1
    nb = neighbours_of(r)
    conf = utils.confidence(r["score_pct"])
    top_n = st.session_state.get("top_n", 15)
    min_score = st.session_state.get("min_score", 0.0)
    only_cand = st.session_state.get("ref_set") == "Candidate genes only"
    use_pct = st.session_state.get("axis_mode") == "Percentile"

    if known:
        status = '<span class="badge b-pos">Known AD gene</span>'
    else:
        lvl = ("b-strong", "Strong network evidence") if r["nb_frac"] >= 0.3 else (
            ("b-mid", "Moderate network evidence") if r["nb_frac"] >= 0.1 else ("b-weak", "Weak evidence, treat as hypothesis"))
        status = f'<span class="badge b-cand">Candidate</span><span class="badge {lvl[0]}">{lvl[1]}</span>'

    lines = [f"{r['nb_frac']:.0%} of its {int(r['degree'])} network neighbours are known AD genes."]
    if r["p_adj"] < 0.05:
        lines.append(f"Expression differs between AD and control brain (log2FC {r['log2fc']:+.2f}, adjusted p {r['p_adj']:.1e}).")
    else:
        lines.append(f"No significant expression change (adjusted p {r['p_adj']:.2f}).")
    if not known:
        lines.append("The score ranks network proximity. It is not proof of association.")

    with st.spinner("Fetching gene details from NCBI..."):
        info = utils.ncbi_gene_info(symbol=g)
    ctx = utils.ranking_context(df, g, top_n, min_score)

    st.markdown(f"## Gene report: {g}")
    st.markdown(
        f"""<div class="qsum">
  <div><span>Query</span><b>{g}</b></div>
  <div><span>Database</span><b>{N_TOTAL:,} scored genes</b></div>
  <div><span>Method</span><b>Network propagation + logistic regression</b></div>
  <div><span>Held-out AUROC</span><b>0.68</b></div>
</div>""",
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Query gene", g)
    m2.metric("Model score", f"{r['score']:.3f}")
    m3.metric("Overall rank", f"{int(r['rank']):,} / {N_TOTAL:,}")
    m4.metric("Confidence", conf, help="Rank-based band from the score percentile: High 95 and above, "
                                       "Medium 80 to 95, Low below 80. Not a probability.")
    st.write("")

    main, side = st.columns([3, 1])
    with main:
        t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(
    ["Summary", "Gene details", "Network neighbourhood", "Ranking context", "Score position",
     "Model features", "PubMed", "Brief and Ask"])

        with t1:
            with st.container(border=True):
                st.markdown('<div class="sect">Summary</div>', unsafe_allow_html=True)
                rows = [("Gene symbol", g), ("Status", status), ("Model score", f"{r['score']:.3f}"),
                        ("Overall rank", f"{int(r['rank']):,} of {N_TOTAL:,}"),
                        ("Rank among candidates", "not applicable" if known else f"{int(r['novel_rank']):,} of {N_NEG:,}"),
                        ("Score percentile", f"{r['score_pct']:.1f}"),
                        ("Confidence band", conf),
                        ("Network degree", f"{int(r['degree'])}"),
                        ("Neighbours that are known AD genes", f"{r['nb_frac']:.0%}")]
                st.markdown('<table class="kv">' + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
                            + "</table>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown('<div class="sect">Evidence</div>', unsafe_allow_html=True)
                st.markdown('<ul class="evid">' + "".join(f"<li>{x}</li>" for x in lines) + "</ul>", unsafe_allow_html=True)

        with t2:
            with st.container(border=True):
                st.markdown('<div class="sect">Gene details (NCBI Gene)</div>', unsafe_allow_html=True)
                if info:
                    esc = html.escape
                    mim = ", ".join(f'<a href="https://www.omim.org/entry/{esc(m)}" target="_blank">{esc(m)}</a>'
                                    for m in info["mim"]) or "none listed"
                    where = f"{info['chromosome']}  {info['maplocation']}".strip()
                    rows = [("Official symbol", esc(info["symbol"])), ("Full name", esc(info["name"])),
                            ("Entrez Gene ID", esc(info["gene_id"])), ("Chromosome and location", esc(where) or "not listed"),
                            ("Also known as", esc(", ".join(info["aliases"])) or "none listed"), ("OMIM", mim)]
                    st.markdown('<table class="kv">' + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
                                + "</table>", unsafe_allow_html=True)
                    if info["summary"]:
                        st.markdown("**Summary from NCBI**")
                        st.write(info["summary"])
                else:
                    st.warning("Could not load gene details from NCBI. Check the internet connection and search again.")
            with st.container(border=True):
                st.markdown('<div class="sect">Known Alzheimer\'s neighbours in the network</div>', unsafe_allow_html=True)
                st.write(", ".join(nb) if nb else "None listed.")

        with t3:
            with st.container(border=True):
                st.markdown('<div class="sect">Network neighbourhood</div>', unsafe_allow_html=True)
                if nb:
                    st.markdown(star_svg(g, nb, known), unsafe_allow_html=True)
                    st.caption("Pink: known AD genes directly connected to this gene in STRING.")
                else:
                    st.write("No known AD neighbours listed for this gene.")

            # ADDED: network + partner charts
            with st.container(border=True):
                st.markdown('<div class="sect">Interactive network</div>', unsafe_allow_html=True)
                if nb:
                    st.plotly_chart(network_chart(g, nb, known, df), width="stretch")
                    st.plotly_chart(partners_chart(df, nb), width="stretch")
                else:
                    st.info("No known AD neighbours to chart for this gene.")

        with t4:
            with st.container(border=True):
                st.markdown('<div class="sect">Ranking context</div>', unsafe_allow_html=True)
                st.write(f"The {len(ctx)} genes ranked closest to {g}. Select a row to open that gene's report. "
                         "Change the size and the score cut-off under Advanced parameters.")
                ev = st.dataframe(utils.style_table(ctx, g), hide_index=True, on_select="rerun",
                                  selection_mode="single-row", key="ctx_table")
                if ev.selection.rows:
                    pick = ctx.iloc[ev.selection.rows[0]]["Gene"]
                    if pick != g:
                        open_gene(pick)
                try:
                    st.plotly_chart(utils.bar_fig(ctx, g), width="stretch")
                except ImportError:
                    st.warning("Charts need the plotly package: pip install plotly")

        with t5:
            with st.container(border=True):
                st.markdown('<div class="sect">Position among scored genes</div>', unsafe_allow_html=True)
                ref = df[df.label == 0] if only_cand else df
                col_, title_ = ("score_pct", "Score percentile") if use_pct else ("score", "Model score")
                try:
                    st.plotly_chart(utils.hist_fig(ref[col_], r[col_], title_), width="stretch")
                except ImportError:
                    st.warning("Charts need the plotly package: pip install plotly")
                st.caption(f"Pink line: this gene, against {'candidate genes only' if only_cand else 'all scored genes'}. "
                           "Change this under Advanced parameters.")

            # ADDED: score charts
            with st.container(border=True):
                st.markdown('<div class="sect">Score charts</div>', unsafe_allow_html=True)
                st.plotly_chart(gauge_chart(r["score"], float(r["score_pct"])), width="stretch")
                st.plotly_chart(hist_chart(df, g, r["score"]), width="stretch")
                st.plotly_chart(scatter_chart(df, g), width="stretch")
                st.plotly_chart(volcano_chart(df, g), width="stretch")

        with t6:
            with st.container(border=True):
                st.markdown('<div class="sect">Features used by the model</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame({
                    "feature": ["rwr (random walk with restart)", "nb_frac (share of neighbours that are known AD genes)",
                                "degree", "t_stat", "log2fc", "p_adj"],
                    "value": [f"{r['rwr']:.2e}", f"{r['nb_frac']:.3f}", f"{int(r['degree'])}",
                              f"{r['t_stat']:.3f}", f"{r['log2fc']:.3f}", f"{r['p_adj']:.2e}"]}), hide_index=True)

        # ADDED: PubMed tab
        with t7:
            render_pubmed(g)

        # ADDED: brief + ask tab
        with t8:
            render_brief(g, r, nb, known, conf)

    with side:
        with st.container(border=True):
            st.markdown('<div class="sect">Related information</div>', unsafe_allow_html=True)
            gid = info.get("gene_id") if info else None
            st.link_button("NCBI Gene", f"https://www.ncbi.nlm.nih.gov/gene/{gid}" if gid
                           else f"https://www.ncbi.nlm.nih.gov/gene/?term={g}%5Bsym%5D+AND+human%5Borgn%5D")
            if info and info["mim"]:
                st.link_button("OMIM", f"https://www.omim.org/entry/{info['mim'][0]}")
            st.link_button("PubMed", f"https://pubmed.ncbi.nlm.nih.gov/?term={g}+AND+Alzheimer")
            st.link_button("Open Targets", f"https://platform.opentargets.org/search?q={g}")
        with st.container(border=True):
            st.markdown('<div class="sect">Download</div>', unsafe_allow_html=True)
            st.download_button("Ranking context (CSV)", ctx.to_csv(index=False).encode("utf-8"),
                               file_name=f"{g}_ranking_context.csv", mime="text/csv", on_click="ignore")
            try:
                pdf = utils.build_pdf(r, info, lines, ctx, nb)
                st.download_button("Gene report (PDF)", pdf, file_name=f"{g}_report.pdf",
                                   mime="application/pdf", on_click="ignore")
            except ImportError:
                st.caption("PDF export needs the fpdf2 package: pip install fpdf2")
            except Exception as e:
                st.caption(f"PDF could not be built: {e}")
        inside = [n for n in nb if n in GENESET and n != g]
        if inside:
            with st.container(border=True):
                st.markdown('<div class="sect">Neighbouring genes</div>', unsafe_allow_html=True)
                for n in inside:
                    if st.button(n, key=f"nb_{n}"):
                        open_gene(n)


def browse():
    st.markdown("## Ranked predictions")
    st.write("Genes not yet labelled as Alzheimer's genes, ordered by model score. "
             "Select a row to open its gene report.")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        top_n = c1.slider("Show top", 10, 300, 50, step=10)
        min_nb = c2.slider("Minimum neighbour evidence", 0.0, 1.0, 0.0, step=0.05)
        q = c3.text_input("Filter by name", "").strip().upper()
    cand = df[(df.label == 0) & (df.nb_frac >= min_nb)]
    if q:
        cand = cand[cand["gene"].astype(str).str.upper().str.contains(q, regex=False)]
    cand = cand.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    cols = ["novel_rank", "gene", "score", "nb_frac", "degree", "log2fc", "p_adj", "known_AD_neighbours"]
    ev = st.dataframe(
        cand[cols], hide_index=True, on_select="rerun", selection_mode="single-row", key="cand_table",
        column_config={"novel_rank": "Rank", "gene": "Gene",
                       "score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.3f"),
                       "nb_frac": st.column_config.ProgressColumn("Neighbour evidence", min_value=0.0, max_value=1.0, format="%.2f"),
                       "degree": st.column_config.NumberColumn("Degree", format="%d"),
                       "log2fc": st.column_config.NumberColumn("log2FC", format="%.3f"),
                       "p_adj": st.column_config.NumberColumn("Adj. p", format="%.2e"),
                       "known_AD_neighbours": "Known AD neighbours"})
    if ev.selection.rows:
        open_gene(cand.iloc[ev.selection.rows[0]]["gene"])

    # ADDED: top-20 chart + CSV download
    st.plotly_chart(ranked_chart(cand), width="stretch")
    st.download_button("Download this table (CSV)",
                       cand[cols].to_csv(index=False).encode(),
                       file_name="ad_ranked_predictions.csv", mime="text/csv",
                       key="dl_rank_csv")
    st.caption("Genes with low neighbour evidence rank high through indirect proximity and are weaker hypotheses.")


def methods():
    st.markdown("## Results and method")
    with st.container(border=True):
        st.markdown('<div class="sect">Held-out test set (1,587 genes, 66 positives; random AUPRC about 0.042)</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            ["LogReg, all features", 0.681, "0.609 - 0.750", 0.123],
            ["LogReg, propagation + extra seeds", 0.670, "0.597 - 0.739", 0.111],
            ["RF (regularised), propagation", 0.661, "-", 0.190],
            ["RF, all features", 0.665, "-", 0.097],
            ["LogReg, expression only", 0.552, "includes 0.5", np.nan],
            ["RF, expression only", 0.496, "includes 0.5", np.nan],
            ["LogReg, topology only", 0.540, "includes 0.5", np.nan],
            ["RF, topology only", 0.532, "includes 0.5", np.nan]],
            columns=["Model", "AUROC", "95% CI (bootstrap)", "AUPRC"]), hide_index=True)
    with st.container(border=True):
        st.markdown('<div class="sect">Grouped permutation importance (LogReg, all features)</div>', unsafe_allow_html=True)
        st.bar_chart(pd.DataFrame({"AUROC drop": [0.193, 0.040, 0.020, 0.004]},
                                  index=["Propagation", "Topology (degree, pagerank)", "Expression", "Other topology"]))
    with st.container(border=True):
        st.markdown('<div class="sect">Independent checks</div>', unsafe_allow_html=True)
        st.markdown(
            "- Weak-evidence genes (Open Targets 0 < score < 0.3, n = 261) scored above negatives: AUROC 0.55, "
            "95% CI 0.51 - 0.59, Mann-Whitney p = 0.005. Modest effect; single features such as degree (0.527) "
            "were closer to chance.\n"
            "- PubMed: top-50 candidates against 50 random label-0 genes showed no clear enrichment "
            "(median hits 0 against 0; mean 6.74 against 1.54 came from a few outliers).")
        if LIT.exists():
            with st.expander("PubMed literature counts"):
                st.dataframe(pd.read_csv(LIT), hide_index=True)
    with st.container(border=True):
        st.markdown('<div class="sect">Pipeline and limitations</div>', unsafe_allow_html=True)
        st.markdown(
            "Nine expression datasets merged, batch-aware differential expression, STRING v12.5 network "
            "(combined score >= 400), Open Targets labels (positive: genetic association >= 0.3, negative: 0, "
            "middle dropped), propagation features with cross-fitted seeds, stratified 80/20 split with the test "
            "set evaluated once.\n\n"
            "- Scores are ranking scores from a class-balanced model, not probabilities.\n"
            "- Labels reflect genetic association only; label 0 genes are not proven unrelated.\n"
            "- Top candidates often rank high through proximity to a few loci such as the MS4A cluster.\n"
            "- Literature validation did not show clear enrichment.\n"
            "- Studies cluster apart in expression space; batch effects are only partly corrected.")


P_HOME = st.Page(home, title="Search", icon=":material/search:", url_path="search", default=True)
P_GENE = st.Page(gene_page, title="Gene report", icon=":material/biotech:", url_path="gene")
P_BROWSE = st.Page(browse, title="Ranked predictions", icon=":material/table_rows:", url_path="ranked")
P_METHODS = st.Page(methods, title="Results and method", icon=":material/science:", url_path="method")

nav = st.navigation([P_HOME, P_GENE, P_BROWSE, P_METHODS], position="top")
with st.sidebar:
    st.markdown("### About")
    st.write("AD Gene Predictor ranks human genes by how close they sit to known Alzheimer's disease genes in the "
             "STRING protein interaction network, combined with brain expression changes.")
    st.markdown("### Help")
    st.markdown("- Type a gene symbol (MS4A8) or an NCBI Gene ID (7157), or pick from the list, then press **Predict**.\n"
                "- Use **Advanced parameters** to change how many ranked genes are shown and the score cut-off.\n"
                "- Scores rank network proximity. They are not probabilities.\n"
                "- Gene descriptions are fetched live from NCBI Gene, so they need an internet connection.")
search_bar()
nav.run()
st.markdown('<div class="footer">For research use only, not for clinical diagnosis.</div>', unsafe_allow_html=True)

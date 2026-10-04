"""Helper functions for the AD Gene Atlas Streamlit app.

The "backend" of this project is the precomputed scored-gene table
(results/app_gene_table.csv) produced by the modelling notebook, so
`predict_gene` simply looks a gene up in that table.
"""
import difflib
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).parent
TABLE = ROOT / "results" / "app_gene_table.csv"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PINK, BLUE, NAVY = "#E0457B", "#5B8DB8", "#12395F"


# ----------------------------------------------------------------- backend
@st.cache_data
def load_table():
    d = pd.read_csv(TABLE)
    d["known_AD_neighbours"] = d["known_AD_neighbours"].fillna("")
    d["novel_rank"] = np.nan
    neg = d["label"] == 0
    d.loc[neg, "novel_rank"] = d.loc[neg, "score"].rank(ascending=False, method="min")
    return d


def confidence(pct):
    """Rank-based confidence band from the score percentile (not a probability)."""
    if pct >= 95:
        return "High"
    if pct >= 80:
        return "Medium"
    return "Low"


def predict_gene(df, gene):
    """Return the scored row for one gene symbol."""
    return df[df["gene"] == gene].iloc[0]


def ranking_context(df, gene, top_n=15, min_score=0.0):
    """The `top_n` genes ranked closest to `gene`, keeping genes with score >= min_score."""
    d = df.sort_values("score", ascending=False).reset_index(drop=True)
    pos = int(d.index[d["gene"] == gene][0])
    lo = max(0, pos - top_n // 2)
    hi = min(len(d), lo + top_n)
    lo = max(0, hi - top_n)
    w = d.iloc[lo:hi]
    w = w[(w["score"] >= min_score) | (w["gene"] == gene)]
    return pd.DataFrame({
        "Rank": w["rank"].astype(int),
        "Gene": w["gene"],
        "Score": w["score"],
        "Confidence": w["score_pct"].map(confidence),
        "Status": np.where(w["label"] == 1, "Known AD gene", "Candidate"),
        "Neighbour evidence": w["nb_frac"],
    }).reset_index(drop=True)


# ----------------------------------------------------------------- input handling
def suggest(query, genes, n=6):
    q = str(query).strip().upper()
    pref = [g for g in genes if g.upper().startswith(q)][:n]
    close = difflib.get_close_matches(q, genes, n=n, cutoff=0.6)
    out = []
    for g in pref + close:
        if g not in out:
            out.append(g)
    return out[:n]


def resolve_query(q, genes):
    """Turn what the user typed (symbol or Entrez ID) into a scored gene symbol.

    Returns (symbol or None, error message, suggestions)."""
    q = (q or "").strip()
    if not q:
        return None, "Enter a gene symbol or an NCBI Gene ID.", []
    by_upper = {g.upper(): g for g in genes}
    if q.isdigit():
        info = ncbi_gene_info(gene_id=q)
        if not info:
            return None, f"Gene ID {q} was not found at NCBI, or NCBI could not be reached.", []
        sym = info["symbol"]
        if sym.upper() in by_upper:
            return by_upper[sym.upper()], "", []
        return None, f"Gene ID {q} is {sym}, which is not among the {len(genes):,} genes scored by this tool.", suggest(sym, genes)
    if q.upper() in by_upper:
        return by_upper[q.upper()], "", []
    return None, f"'{q}' is not among the {len(genes):,} genes scored by this tool.", suggest(q, genes)


# ----------------------------------------------------------------- NCBI Gene details
def _eutils(path, **params):
    r = requests.get(f"{EUTILS}/{path}", params={**params, "retmode": "json"}, timeout=12)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False, ttl=86400)
def _ncbi_gene_info(symbol, gene_id):
    # raises on network errors, so failures are never cached
    if not gene_id:
        ids = _eutils("esearch.fcgi", db="gene", term=f"{symbol}[sym] AND human[orgn]")["esearchresult"]["idlist"]
        if not ids:
            return {}
        gene_id = ids[0]
    doc = _eutils("esummary.fcgi", db="gene", id=str(gene_id))["result"][str(gene_id)]
    return {
        "gene_id": str(gene_id),
        "symbol": doc.get("name", ""),
        "name": doc.get("description", ""),
        "chromosome": doc.get("chromosome", ""),
        "maplocation": doc.get("maplocation", ""),
        "aliases": [a.strip() for a in str(doc.get("otheraliases", "")).split(",") if a.strip()],
        "summary": doc.get("summary", ""),
        "mim": [str(m) for m in (doc.get("mim") or [])],
    }


def ncbi_gene_info(symbol="", gene_id=""):
    try:
        return _ncbi_gene_info(symbol, gene_id)
    except Exception:
        return {}


# ----------------------------------------------------------------- tables and charts
def style_table(ctx, gene):
    def shade(v):
        return f"background-color: rgba(11,107,203,{0.08 + 0.55 * float(v):.2f})"

    def mark(row):
        return ["background-color: #FDE3EC; font-weight: 700" if (row["Gene"] == gene and c in ("Rank", "Gene")) else ""
                for c in row.index]

    return (ctx.style.map(shade, subset=["Score"]).apply(mark, axis=1)
            .format({"Score": "{:.3f}", "Neighbour evidence": "{:.2f}"}))


def bar_fig(ctx, gene):
    import plotly.graph_objects as go
    d = ctx.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["Score"], y=d["Gene"], orientation="h",
        marker_color=[PINK if g == gene else BLUE for g in d["Gene"]],
        text=[f"{v:.3f}" for v in d["Score"]], textposition="outside"))
    fig.update_layout(height=max(280, 28 * len(d) + 90), margin=dict(l=10, r=40, t=10, b=10),
                      xaxis_title="Model score", yaxis=dict(type="category"),
                      plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(range=[0, min(1.1, float(d["Score"].max()) * 1.18)], gridcolor="#E1ECF5")
    return fig


def hist_fig(values, mark, xtitle):
    import plotly.graph_objects as go
    fig = go.Figure(go.Histogram(x=values, nbinsx=40, marker_color="#8FB8CC"))
    fig.add_vline(x=float(mark), line_width=3, line_color=PINK, annotation_text="query gene",
                  annotation_position="top")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), xaxis_title=xtitle,
                      yaxis_title="Genes", plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", bargap=0.05)
    fig.update_yaxes(gridcolor="#E1ECF5")
    return fig


# ----------------------------------------------------------------- PDF report
def _s(t):
    return str(t).encode("latin-1", "replace").decode("latin-1")


def build_pdf(row, info, evidence, ctx, neighbours):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin

    def heading(txt):
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(18, 57, 95)
        pdf.cell(0, 8, _s(txt), new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(11, 107, 203)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + width, pdf.get_y())
        pdf.ln(2)

    def para(txt, size=10):
        pdf.set_font("Helvetica", "", size)
        pdf.set_text_color(40, 50, 65)
        pdf.multi_cell(0, 5.2, _s(txt), new_x="LMARGIN", new_y="NEXT")

    def kv(k, v):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(74, 101, 128)
        pdf.cell(52, 6, _s(k), new_x="RIGHT", new_y="TOP")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(18, 57, 95)
        pdf.multi_cell(0, 6, _s(v), new_x="LMARGIN", new_y="NEXT")

    gene = row["gene"]
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(18, 57, 95)
    pdf.cell(0, 10, "AD Gene Atlas - gene report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(74, 101, 128)
    pdf.cell(0, 6, _s(f"Query gene: {gene}"), new_x="LMARGIN", new_y="NEXT")

    heading("Model summary")
    kv("Status", "Known AD gene" if row["label"] == 1 else "Candidate gene")
    kv("Model score", f"{row['score']:.3f}")
    kv("Overall rank", f"{int(row['rank']):,} of the scored genes")
    kv("Score percentile", f"{row['score_pct']:.1f}")
    kv("Confidence band", confidence(row["score_pct"]) + " (rank based, not a probability)")
    kv("Network degree", f"{int(row['degree'])}")
    kv("Known AD neighbours", ", ".join(neighbours) if neighbours else "none listed")

    heading("Evidence")
    for line in evidence:
        para("- " + line)

    if info:
        heading("Gene details (NCBI Gene)")
        kv("Full name", info.get("name", ""))
        kv("Entrez Gene ID", info.get("gene_id", ""))
        kv("Chromosome", f"{info.get('chromosome', '')} {info.get('maplocation', '')}".strip())
        kv("Also known as", ", ".join(info.get("aliases", [])) or "none listed")
        kv("OMIM", ", ".join(info.get("mim", [])) or "none listed")
        if info.get("summary"):
            pdf.ln(1)
            para(info["summary"], 9)

    heading("Ranking context")
    cols = [("Rank", 16), ("Gene", 34), ("Score", 22), ("Confidence", 26)]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(74, 101, 128)
    for name, w in cols:
        pdf.cell(w, 6, name, new_x="RIGHT", new_y="TOP")
    pdf.cell(0, 6, "Score bar", new_x="LMARGIN", new_y="NEXT")
    bar_x = pdf.l_margin + sum(w for _, w in cols)
    bar_w = width - sum(w for _, w in cols)
    for _, r in ctx.iterrows():
        me = r["Gene"] == gene
        pdf.set_font("Helvetica", "B" if me else "", 9)
        pdf.set_text_color(*((224, 69, 123) if me else (40, 50, 65)))
        y = pdf.get_y()
        vals = [str(int(r["Rank"])), str(r["Gene"]), f"{r['Score']:.3f}", str(r["Confidence"])]
        for (name, w), v in zip(cols, vals):
            pdf.cell(w, 6, _s(v), new_x="RIGHT", new_y="TOP")
        pdf.set_fill_color(*((224, 69, 123) if me else (91, 141, 184)))
        pdf.rect(bar_x, y + 1.3, max(0.5, bar_w * float(r["Score"])), 3.4, style="F")
        pdf.set_xy(pdf.l_margin, y + 6)

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(110, 120, 135)
    pdf.multi_cell(0, 4.5, "For research use only, not for clinical diagnosis. Scores rank network proximity "
                           "to known Alzheimer's genes and are not probabilities.",
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())

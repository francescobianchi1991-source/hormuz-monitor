"""
hormuz_app.py
=============
Hormuz Disruption Monitor — Web App Streamlit
Importa hormuz_engine.py

Avvio: streamlit run hormuz_app.py

Pagine:
    🗺️  Mappa Live        — navi in tempo reale + contatore traffico
    📊  Disruption Index  — indice 0-100 + componenti + gauge
    📈  Analisi Storica   — serie storica indice + eventi noti
    🔁  Correlazione      — forward return Brent dopo spike
    📖  Metodologia
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import time
from datetime import datetime, timedelta

from hormuz_engine import (
    fetch_ais_snapshot,
    fetch_ais_progressive,
    fetch_financial_proxies,
    compute_disruption_index,
    build_historical_index,
    compute_brent_correlation,
    classify_disruption,
    get_disruption_color,
    get_disruption_emoji,
    get_hormuz_bbox,
    KNOWN_EVENTS,
    HORMUZ_BBOX,
    GULF_BBOX,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hormuz Disruption Monitor",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Syne:wght@400;600;700;800&display=swap');
:root{
    --bg-primary:#0a0e17;--bg-secondary:#111827;--bg-card:#151d2e;
    --border:#1e2d47;--text-primary:#e8edf5;--text-secondary:#8a9ab5;--text-dim:#4a5a75;
    --accent-cyan:#06b6d4;--accent-green:#10b981;--accent-red:#ef4444;
    --accent-amber:#f59e0b;--accent-purple:#8b5cf6;--accent-blue:#3b82f6;
}
.stApp{background:var(--bg-primary);color:var(--text-primary);}
.main .block-container{padding:1.5rem 2rem;max-width:1600px;}
#MainMenu,footer,header{visibility:hidden;}.stDeployButton{display:none;}
[data-testid="stSidebar"]{background:var(--bg-secondary)!important;border-right:1px solid var(--border);}
[data-testid="stSidebar"] .stRadio label{color:var(--text-secondary)!important;font-family:'JetBrains Mono',monospace;font-size:0.85rem;}
h1,h2,h3{font-family:'Syne',sans-serif!important;color:var(--text-primary)!important;letter-spacing:-0.02em;}
p,div,span,label{font-family:'JetBrains Mono',monospace;}
.kpi-card{background:var(--bg-card);border:1px solid var(--border);border-radius:8px;padding:1.1rem 1.3rem;position:relative;overflow:hidden;}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--accent-purple),var(--accent-cyan));}
.kpi-label{font-size:0.66rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.4rem;}
.kpi-value{font-family:'Syne',sans-serif;font-size:1.7rem;font-weight:700;line-height:1;margin-bottom:0.2rem;}
.kpi-sub{font-size:0.7rem;color:var(--text-secondary);}
.section-header{font-size:0.68rem;color:var(--accent-cyan);text-transform:uppercase;letter-spacing:0.2em;border-bottom:1px solid var(--border);padding-bottom:0.4rem;margin:1.4rem 0 0.8rem 0;}
.top-bar{display:flex;align-items:center;justify-content:space-between;padding:0.7rem 0;border-bottom:1px solid var(--border);margin-bottom:1.3rem;}
.top-bar-title{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:var(--text-primary);letter-spacing:-0.03em;}
.top-bar-meta{font-size:0.7rem;color:var(--text-dim);}
.level-badge-NORMAL{display:inline-block;padding:0.3rem 1rem;border-radius:4px;font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;background:rgba(16,185,129,0.15);color:#10b981;border:1px solid #10b981;}
.level-badge-ELEVATED{display:inline-block;padding:0.3rem 1rem;border-radius:4px;font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;background:rgba(245,158,11,0.15);color:#f59e0b;border:1px solid #f59e0b;}
.level-badge-CRITICAL{display:inline-block;padding:0.3rem 1rem;border-radius:4px;font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;background:rgba(239,68,68,0.15);color:#ef4444;border:1px solid #ef4444;animation:pulse-border 1.5s infinite;}
@keyframes pulse-border{0%,100%{opacity:1;}50%{opacity:0.5;}}
.narrative-box{background:var(--bg-card);border:1px solid var(--border);border-left:3px solid var(--accent-cyan);border-radius:6px;padding:0.8rem 1.2rem;font-size:0.78rem;color:var(--text-secondary);margin:0.8rem 0;}
.vessel-row{background:var(--bg-card);border:1px solid var(--border);border-radius:6px;padding:0.6rem 1rem;margin-bottom:0.4rem;font-size:0.75rem;}
.dot-live{display:inline-block;width:7px;height:7px;background:var(--accent-green);border-radius:50%;margin-right:6px;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.3;}}
.disclaimer{background:rgba(245,158,11,0.05);border:1px solid rgba(245,158,11,0.2);border-left:3px solid #f59e0b;border-radius:4px;padding:0.7rem 1rem;font-size:0.7rem;color:var(--text-secondary);margin-top:0.8rem;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# COSTANTI
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_API_KEY = "13a6cc4808b9045138b6c983e7a2c136ea79f8e7"

PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(21,29,46,0.6)",
    font=dict(family="JetBrains Mono", color="#8a9ab5", size=11),
    xaxis=dict(gridcolor="#1e2d47", linecolor="#1e2d47", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1e2d47", linecolor="#1e2d47", tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2d47", borderwidth=1),
    margin=dict(l=50, r=20, t=40, b=40),
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in {
    "vessels_df":    None,
    "fin_df":        None,
    "hist_index":    None,
    "disruption":    None,
    "correlation":   None,
    "last_ais_time": None,
    "last_fin_time": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# HELPER UI
# ─────────────────────────────────────────────────────────────────────────────
def kpi(label, val, sub, color):
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>{label}</div>"
                f"<div class='kpi-value' style='color:{color};'>{val}</div>"
                f"<div class='kpi-sub'>{sub}</div></div>", unsafe_allow_html=True)

def section(t):
    st.markdown(f"<div class='section-header'>{t}</div>", unsafe_allow_html=True)

def top_bar(t, m):
    st.markdown(f"<div class='top-bar'><div class='top-bar-title'>{t}</div>"
                f"<div class='top-bar-meta'>{m}</div></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CARICAMENTO DATI FINANZIARI (cachato 1 ora)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_financial_data():
    return fetch_financial_proxies()

@st.cache_data(ttl=3600, show_spinner=False)
def load_historical_index(fin_hash: str):
    fin_df = load_financial_data()
    if fin_df.empty:
        return pd.DataFrame()
    return build_historical_index(fin_df)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='padding:1rem 0 1.2rem 0;'>
        <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:#e8edf5;'>
            🛢️ Hormuz Monitor
        </div>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#4a5a75;
                    margin-top:4px;text-transform:uppercase;letter-spacing:0.1em;'>
            Disruption Intelligence
        </div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("nav", [
        "🗺️  Mappa Live",
        "📊  Disruption Index",
        "📈  Analisi Storica",
        "🔁  Correlazione Brent",
        "📖  Metodologia",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1e2d47;margin:0.8rem 0;'>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.62rem;color:#4a5a75;text-transform:uppercase;"
                "letter-spacing:0.1em;margin-bottom:0.4rem;'>API Key AISStream</div>",
                unsafe_allow_html=True)
    api_key = st.text_input("API Key", value=DEFAULT_API_KEY,
                             type="password", label_visibility="collapsed")

    st.markdown("<div style='font-size:0.62rem;color:#4a5a75;text-transform:uppercase;"
                "letter-spacing:0.1em;margin-top:0.8rem;margin-bottom:0.4rem;'>Raccolta AIS</div>",
                unsafe_allow_html=True)
    ais_seconds = st.slider("Secondi raccolta", 20, 180, 90, 10)
    only_tankers = st.checkbox("Solo tanker/cargo", value=False)

    bbox_choice = st.radio("Area geografica", ["Golfo Persico completo", "Stretto Hormuz"],
                           label_visibility="collapsed")
    bbox = GULF_BBOX if bbox_choice == "Golfo Persico completo" else HORMUZ_BBOX

    st.markdown("<hr style='border-color:#1e2d47;margin:0.8rem 0;'>", unsafe_allow_html=True)

    # Status
    if st.session_state.last_ais_time:
        st.markdown(f"<div style='font-size:0.62rem;color:#4a5a75;'>"
                    f"<span class='dot-live'></span>Ultimo AIS: "
                    f"<span style='color:#8a9ab5;'>{st.session_state.last_ais_time}</span><br>"
                    f"Navi rilevate: <span style='color:#8a9ab5;'>"
                    f"{len(st.session_state.vessels_df) if st.session_state.vessels_df is not None else 0}"
                    f"</span></div>", unsafe_allow_html=True)

    if st.session_state.disruption:
        d = st.session_state.disruption
        color = get_disruption_color(d["level"])
        st.markdown(f"<div style='margin-top:0.8rem;font-size:0.62rem;color:#4a5a75;'>"
                    f"Disruption Index: <span style='color:{color};font-weight:700;font-size:0.9rem;'>"
                    f"{d['score']}</span> "
                    f"<span style='color:{color};'>{get_disruption_emoji(d['level'])} {d['level']}</span>"
                    f"</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MAPPA LIVE
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MAPPA LIVE
if "Mappa" in page:
    top_bar("🗺️ Mappa Live — Stretto di Hormuz",
            "<span class='dot-live'></span>Dati AIS in tempo reale via AISStream.io")

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        total_minutes = st.slider("Durata raccolta (minuti)", 1, 30, 5, 1)
        total_sec_live = total_minutes * 60
    with col_cfg2:
        chunk_sec = st.slider("Aggiorna mappa ogni (secondi)", 10, 60, 20, 5)
    with col_cfg3:
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        run_live = st.button("📡 Avvia raccolta live", use_container_width=True)

    def _build_map_fig(df_v):
        """Costruisce la figura Plotly della mappa."""
        def vc(row):
            if row.get("is_tanker", False):
                s = row.get("speed", 0) or 0
                return "#f59e0b" if s < 1 else "#ef4444" if s < 8 else "#10b981"
            return "#3b82f6"

        df_v = df_v.copy()
        df_v["color"] = df_v.apply(vc, axis=1)
        df_v["size"]  = df_v["speed"].fillna(5).clip(2, 25) + 8

        def ht(row):
            name = str(row.get("name", "") or "Unknown")
            spd  = f"{row['speed']:.1f} kn" if pd.notna(row.get("speed")) else "n/a"
            dest = str(row.get("destination", "") or "n/a")
            typ  = str(row.get("ship_type_label", "") or "Unknown")
            mmsi = str(row.get("mmsi", ""))
            return f"<b>{name}</b><br>MMSI: {mmsi}<br>Tipo: {typ}<br>Velocità: {spd}<br>Dest: {dest}"

        df_v["hover"] = df_v.apply(ht, axis=1)

        fig = go.Figure()
        fig.add_scattermapbox(
            lat=[26.4, 26.4, 26.8, 26.8, 26.4],
            lon=[55.8, 57.0, 57.0, 55.8, 55.8],
            mode="lines", line=dict(color="rgba(6,182,212,0.4)", width=2),
            name="Stretto Hormuz", hoverinfo="skip",
        )
        fig.add_scattermapbox(
            lat=df_v["lat"].tolist(), lon=df_v["lon"].tolist(),
            mode="markers",
            marker=dict(size=df_v["size"].tolist(), color=df_v["color"].tolist(), opacity=0.85),
            text=df_v["hover"].tolist(),
            hovertemplate="%{text}<extra></extra>",
            name="Navi",
        )
        fig.update_layout(
            mapbox=dict(style="carto-darkmatter", center=dict(lat=25.5, lon=57.0), zoom=5.5),
            paper_bgcolor="rgba(0,0,0,0)", height=500,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True,
            legend=dict(bgcolor="rgba(21,29,46,0.8)", bordercolor="#1e2d47",
                        font=dict(color="#8a9ab5", size=10), x=0.01, y=0.99),
        )
        return fig

    def _render_kpi(df_v):
        n_tot    = len(df_v)
        n_tanker = int(df_v["is_tanker"].sum()) if "is_tanker" in df_v.columns else n_tot
        avg_spd  = df_v["speed"].dropna().mean() if "speed" in df_v.columns else 0
        n_mov    = int((df_v["speed"].fillna(0) > 1).sum()) if "speed" in df_v.columns else 0
        k1,k2,k3,k4 = st.columns(4)
        for col,(lbl,val,sub,c) in zip([k1,k2,k3,k4],[
            ("Navi rilevate", str(n_tot),  "accumulate",       "#3b82f6"),
            ("Tanker/Cargo",  str(n_tanker),"tipo confermato", "#8b5cf6"),
            ("In movimento",  str(n_mov),  "velocità > 1 kn",  "#10b981"),
            ("Velocità media",f"{avg_spd:.1f} kn","SOG",       "#06b6d4"),
        ]):
            with col:
                kpi(lbl, val, sub, c)

    if run_live:
        status_ph = st.empty()
        kpi_ph    = st.empty()
        map_ph    = st.empty()
        table_ph  = st.empty()

        final_holder = {"df": pd.DataFrame()}

        def on_chunk(df_acc, elapsed, total):
            final_holder["df"] = df_acc
            pct = min(100, int(elapsed / total * 100))
            rem = total - elapsed
            status_ph.markdown(
                f"<div style='background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.2);"
                f"border-radius:6px;padding:0.7rem 1rem;font-family:JetBrains Mono,monospace;"
                f"font-size:0.75rem;color:#06b6d4;'>"
                f"<span class='dot-live'></span> Raccolta in corso — "
                f"<b>{elapsed}s / {total}s</b> ({pct}%) — "
                f"Rimanente: <b>{rem//60:02d}:{rem%60:02d}</b> — "
                f"Navi: <b style='color:#e8edf5;'>{len(df_acc)}</b></div>",
                unsafe_allow_html=True
            )
            if df_acc.empty:
                return
            with kpi_ph.container():
                _render_kpi(df_acc)
            map_ph.plotly_chart(_build_map_fig(df_acc), use_container_width=True)
            # Tabella
            sc = ["name","mmsi","ship_type_label","speed","nav_status_label","destination"]
            sc = [c for c in sc if c in df_acc.columns]
            df_s = df_acc[sc].copy()
            df_s.columns = [c.replace("_label","").replace("_"," ").title() for c in sc]
            if "Speed" in df_s.columns:
                df_s["Speed"] = df_s["Speed"].apply(
                    lambda x: f"{x:.1f} kn" if pd.notna(x) else "n/a")
            table_ph.dataframe(df_s, use_container_width=True, hide_index=True, height=250)

        fetch_ais_progressive(
            api_key=api_key,
            total_seconds=total_sec_live,
            chunk_seconds=chunk_sec,
            bbox=bbox,
            on_chunk_ready=on_chunk,
        )

        final_df = final_holder["df"]
        st.session_state.vessels_df    = final_df
        st.session_state.last_ais_time = datetime.now().strftime("%H:%M:%S")

        status_ph.markdown(
            f"<div style='background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);"
            f"border-radius:6px;padding:0.7rem 1rem;font-family:JetBrains Mono,monospace;"
            f"font-size:0.75rem;color:#10b981;'>"
            f"✅ Raccolta completata — {len(final_df)} navi in {total_minutes} min."
            f"</div>", unsafe_allow_html=True
        )

        if not final_df.empty:
            n_tk  = int(final_df["is_tanker"].sum()) if "is_tanker" in final_df.columns else len(final_df)
            dlt   = n_tk - 18
            dlt_p = dlt / 18 * 100
            lv    = "CRITICAL" if n_tk<10 else "ELEVATED" if n_tk<14 else "NORMAL"
            cv    = get_disruption_color(lv)
            av_sp = final_df["speed"].dropna().mean() if "speed" in final_df.columns else 0

            section("TRAFFICO VS BASELINE STORICA")
            cg, ct = st.columns([1, 2])
            with cg:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=n_tk,
                    delta={"reference": 18, "valueformat": ".0f"},
                    title={"text": "Tanker in zona<br><span style='font-size:0.8em'>baseline: 18</span>"},
                    gauge={
                        "axis": {"range":[0,40],"tickcolor":"#4a5a75"},
                        "bar":  {"color": cv},
                        "bgcolor": "rgba(21,29,46,0.6)",
                        "steps": [
                            {"range":[0,10],  "color":"rgba(239,68,68,0.15)"},
                            {"range":[10,14], "color":"rgba(245,158,11,0.15)"},
                            {"range":[14,25], "color":"rgba(16,185,129,0.15)"},
                            {"range":[25,40], "color":"rgba(245,158,11,0.15)"},
                        ],
                        "threshold": {"line":{"color":"#06b6d4","width":3},"value":18},
                    },
                    number={"font":{"color":"#e8edf5","size":36}},
                ))
                fig_g.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#8a9ab5", family="JetBrains Mono"),
                    height=250, margin=dict(l=20,r=20,t=40,b=10),
                )
                st.plotly_chart(fig_g, use_container_width=True)
            with ct:
                st.markdown(f"""
                <div style='padding:1rem;'>
                    <div style='font-size:0.68rem;color:#4a5a75;text-transform:uppercase;
                                letter-spacing:0.1em;margin-bottom:0.5rem;'>Lettura traffico finale</div>
                    <div class='level-badge-{lv}'>{get_disruption_emoji(lv)} {lv}</div>
                    <div style='margin-top:1rem;font-size:0.8rem;color:#8a9ab5;line-height:1.8;'>
                        Tanker rilevati: <b style='color:#e8edf5;'>{n_tk}</b><br>
                        Baseline: <b style='color:#e8edf5;'>18 navi</b><br>
                        Delta: <b style='color:{cv};'>{dlt:+d} ({dlt_p:+.1f}%)</b><br>
                        Velocità media: <b style='color:#e8edf5;'>{av_sp:.1f} kn</b>
                    </div>
                </div>""", unsafe_allow_html=True)

    elif st.session_state.vessels_df is not None and not st.session_state.vessels_df.empty:
        df_v = st.session_state.vessels_df
        st.markdown(
            f"<div style='font-size:0.72rem;color:#4a5a75;margin-bottom:0.8rem;'>"
            f"<span class='dot-live'></span> Ultima raccolta: "
            f"<b style='color:#8a9ab5;'>{st.session_state.last_ais_time}</b> — "
            f"{len(df_v)} navi. Avvia una nuova raccolta per aggiornare.</div>",
            unsafe_allow_html=True
        )
        _render_kpi(df_v)
        st.plotly_chart(_build_map_fig(df_v), use_container_width=True)
        st.markdown("""<div style='font-size:0.68rem;color:#4a5a75;display:flex;gap:1.5rem;margin-top:0.3rem;'>
            <span>🟢 In transito (&gt;8 kn)</span><span>🔴 Lenta (1-8 kn)</span>
            <span>🟡 Ferma (&lt;1 kn)</span><span>🔵 Cargo/Altro</span>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='background:rgba(6,182,212,0.05);border:1px solid rgba(6,182,212,0.2);
                    border-radius:8px;padding:2rem;text-align:center;'>
            <div style='font-size:1rem;color:#06b6d4;margin-bottom:0.5rem;'>📡 Pronto per la raccolta</div>
            <div style='font-size:0.78rem;color:#4a5a75;'>
                Imposta la durata (1-30 minuti), scegli l'aggiornamento mappa e clicca Avvia.<br>
                La mappa si aggiorna automaticamente ogni chunk durante la raccolta.
            </div>
        </div>""", unsafe_allow_html=True)
        fig_e = go.Figure()
        fig_e.add_scattermapbox(
            lat=[21.0,21.0,27.0,27.0,21.0], lon=[54.0,60.0,60.0,54.0,54.0],
            mode="lines", line=dict(color="rgba(6,182,212,0.5)", width=2),
            name="Area monitorata",
        )
        fig_e.update_layout(
            mapbox=dict(style="carto-darkmatter", center=dict(lat=25.5,lon=57.0), zoom=5),
            paper_bgcolor="rgba(0,0,0,0)", height=420, margin=dict(l=0,r=0,t=0,b=0),
        )
        st.plotly_chart(fig_e, use_container_width=True)


elif "Disruption Index" in page:
    top_bar("📊 Disruption Index", "Indice 0-100 · Componenti finanziarie + AIS")

    # Carica dati finanziari
    with st.spinner("Caricamento proxy finanziari..."):
        fin_df = load_financial_data()

    if fin_df.empty:
        st.error("Impossibile caricare i dati finanziari. Verifica la connessione.")
        st.stop()

    # Calcola indice
    n_vessels  = len(st.session_state.vessels_df) if st.session_state.vessels_df is not None else 0
    avg_speed  = (st.session_state.vessels_df["speed"].dropna().mean()
                  if st.session_state.vessels_df is not None and not st.session_state.vessels_df.empty
                  else 0)

    with st.spinner("Calcolo Disruption Index..."):
        disruption = compute_disruption_index(n_vessels, avg_speed, fin_df)

    st.session_state.disruption = disruption
    st.session_state.fin_df     = fin_df

    score = disruption["score"]
    level = disruption["level"]
    color = get_disruption_color(level)
    emoji = get_disruption_emoji(level)

    # ── Header con livello ────────────────────────────────────────────────────
    col_score, col_level = st.columns([1,2])
    with col_score:
        # Gauge principale
        fig_main = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "HORMUZ DISRUPTION INDEX",
                   "font": {"size":13, "color":"#8a9ab5"}},
            gauge={
                "axis":  {"range":[0,100], "tickcolor":"#4a5a75",
                          "tickvals":[0,25,55,70,100],
                          "ticktext":["0","25","55","70","100"]},
                "bar":   {"color": color, "thickness": 0.25},
                "bgcolor": "rgba(21,29,46,0.8)",
                "steps": [
                    {"range":[0,55],  "color":"rgba(16,185,129,0.12)"},
                    {"range":[55,70], "color":"rgba(245,158,11,0.12)"},
                    {"range":[70,100],"color":"rgba(239,68,68,0.12)"},
                ],
                "threshold": {"line":{"color":"#06b6d4","width":3},"value":score},
            },
            number={"font":{"color":color,"size":52,"family":"Syne"},"suffix":""},
        ))
        fig_main.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8a9ab5", family="JetBrains Mono"),
            height=300, margin=dict(l=20,r=20,t=60,b=10),
        )
        st.plotly_chart(fig_main, use_container_width=True)

    with col_level:
        st.markdown(f"""
        <div style='padding:1.5rem 0;'>
            <div class='level-badge-{level}'>{emoji} {level}</div>
            <div class='narrative-box' style='margin-top:1rem;'>
                {disruption['narrative']}
            </div>
            <div style='font-size:0.68rem;color:#4a5a75;margin-top:0.5rem;'>
                Aggiornato: {disruption['timestamp']}
                {f"<br>Dati AIS: {n_vessels} navi in zona" if n_vessels>0 else "<br>Dati AIS: non disponibili (vai a Mappa Live)"}
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Componenti ────────────────────────────────────────────────────────────
    section("COMPONENTI DELL'INDICE")
    comps = disruption.get("components", {})

    if comps:
        comp_labels = []
        comp_scores = []
        comp_colors = []
        comp_hover  = []

        comp_order = ["spread_brent_wti","brent_volatility","xle_anomaly",
                      "vessel_density","avg_speed"]

        for key in comp_order:
            if key not in comps:
                continue
            c = comps[key]
            comp_labels.append(c["label"])
            comp_scores.append(c["score"])
            comp_colors.append(
                "#ef4444" if c["score"]>70 else
                "#f59e0b" if c["score"]>55 else "#10b981"
            )
            comp_hover.append(
                f"Score: {c['score']:.1f}<br>"
                f"Z-score: {c['z_score']:.2f}σ<br>"
                f"Valore: {c['value']}"
            )

        fig_comp = go.Figure()
        fig_comp.add_bar(
            x=comp_scores,
            y=comp_labels,
            orientation="h",
            marker_color=comp_colors,
            opacity=0.85,
            text=[f"{s:.1f}" for s in comp_scores],
            textposition="outside",
            textfont=dict(size=11, color="#8a9ab5"),
            hovertemplate="%{y}<br>%{customdata}<extra></extra>",
            customdata=comp_hover,
        )
        fig_comp.add_vline(x=55, line=dict(color="#f59e0b", dash="dash", width=1),
                           annotation_text="ELEVATED", annotation_font_color="#f59e0b",
                           annotation_font_size=10)
        fig_comp.add_vline(x=70, line=dict(color="#ef4444", dash="dash", width=1),
                           annotation_text="CRITICAL", annotation_font_color="#ef4444",
                           annotation_font_size=10)
        fig_comp.update_layout(
            **PLOTLY_DARK, height=280,
            xaxis=dict(range=[0,110], **PLOTLY_DARK["xaxis"]),
            yaxis=dict(autorange="reversed", **PLOTLY_DARK["yaxis"]),
            title=dict(text=""),
            showlegend=False,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # ── Dati proxy finanziari ─────────────────────────────────────────────────
    section("PROXY FINANZIARI — ULTIMI 30 GIORNI")

    fin_30 = fin_df.tail(30).copy()
    plot_cols = [c for c in ["brent","wti","brent_wti_spread","xle"] if c in fin_30.columns]

    if plot_cols:
        fig_fin = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Brent vs WTI (USD)","Spread Brent-WTI","XLE Energy ETF","Brent Return 1d (%)"],
            vertical_spacing=0.15, horizontal_spacing=0.1,
        )

        if "brent" in fin_30.columns:
            fig_fin.add_scatter(x=fin_30.index, y=fin_30["brent"],
                                line=dict(color="#06b6d4",width=2), name="Brent", row=1, col=1)
        if "wti" in fin_30.columns:
            fig_fin.add_scatter(x=fin_30.index, y=fin_30["wti"],
                                line=dict(color="#8b5cf6",width=1.5,dash="dash"),
                                name="WTI", row=1, col=1)
        if "brent_wti_spread" in fin_30.columns:
            fig_fin.add_scatter(x=fin_30.index, y=fin_30["brent_wti_spread"],
                                line=dict(color="#f59e0b",width=2), name="Spread", row=1, col=2)
            fig_fin.add_hline(y=fin_30["brent_wti_spread"].mean(),
                              line=dict(color="#4a5a75",dash="dot",width=1), row=1, col=2)
        if "xle" in fin_30.columns:
            fig_fin.add_scatter(x=fin_30.index, y=fin_30["xle"],
                                line=dict(color="#10b981",width=2), name="XLE", row=2, col=1)
        if "brent_ret_1d" in fin_30.columns:
            ret = fin_30["brent_ret_1d"] * 100
            fig_fin.add_bar(x=fin_30.index, y=ret,
                            marker_color=["#10b981" if v>0 else "#ef4444" for v in ret],
                            name="Brent ret", row=2, col=2)
            fig_fin.add_hline(y=0, line=dict(color="#4a5a75",width=0.8), row=2, col=2)

        fig_fin.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(21,29,46,0.6)",
            font=dict(family="JetBrains Mono",color="#8a9ab5",size=10),
            height=400, showlegend=False,
            margin=dict(l=40,r=20,t=40,b=30),
        )
        for i in range(1,3):
            for j in range(1,3):
                fig_fin.update_xaxes(gridcolor="#1e2d47",linecolor="#1e2d47",row=i,col=j)
                fig_fin.update_yaxes(gridcolor="#1e2d47",linecolor="#1e2d47",row=i,col=j)

        st.plotly_chart(fig_fin, use_container_width=True)

    st.markdown("""<div class='disclaimer'>
        ⚠️ Il Disruption Index è un indicatore composito a scopo analitico.
        Combina segnali di mercato pubblici e dati AIS in tempo reale.
        Non costituisce raccomandazione di investimento.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANALISI STORICA
# ═══════════════════════════════════════════════════════════════════════════════
elif "Analisi Storica" in page:
    top_bar("📈 Analisi Storica", "Serie storica Disruption Index + eventi geopolitici noti")

    with st.spinner("Caricamento dati storici..."):
        fin_df   = load_financial_data()
        hist_idx = load_historical_index(str(len(fin_df)))

    if hist_idx.empty:
        st.error("Dati storici non disponibili."); st.stop()

    st.session_state.hist_index = hist_idx
    st.session_state.fin_df     = fin_df

    # KPI storici
    pct_normal   = (hist_idx["level"]=="NORMAL").mean()*100
    pct_elevated = (hist_idx["level"]=="ELEVATED").mean()*100
    pct_critical = (hist_idx["level"]=="CRITICAL").mean()*100
    score_now    = hist_idx["score"].iloc[-1]

    k1,k2,k3,k4 = st.columns(4)
    for col,(lbl,val,sub,c) in zip([k1,k2,k3,k4],[
        ("Score attuale",   f"{score_now:.1f}", classify_disruption(score_now), get_disruption_color(classify_disruption(score_now))),
        ("% giorni NORMAL", f"{pct_normal:.1f}%", f"su {len(hist_idx)} giorni","#10b981"),
        ("% giorni ELEVATED",f"{pct_elevated:.1f}%","","#f59e0b"),
        ("% giorni CRITICAL",f"{pct_critical:.1f}%","","#ef4444"),
    ]): col.markdown(f"<div class='kpi-card'><div class='kpi-label'>{lbl}</div><div class='kpi-value' style='color:{c};font-size:1.5rem;'>{val}</div><div class='kpi-sub'>{sub}</div></div>",unsafe_allow_html=True)

    # ── Serie storica con eventi ──────────────────────────────────────────────
    section("DISRUPTION INDEX — SERIE STORICA CON EVENTI GEOPOLITICI")

    fig_hist = go.Figure()

    # Area colorata per livello
    idx_dates = hist_idx.index

    # Banda CRITICAL
    critical_mask = hist_idx["score"] >= 70
    fig_hist.add_scatter(
        x=idx_dates, y=hist_idx["score"].where(critical_mask),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        line=dict(width=0), name="CRITICAL", showlegend=True,
    )
    # Banda ELEVATED
    elevated_mask = (hist_idx["score"] >= 55) & (hist_idx["score"] < 70)
    fig_hist.add_scatter(
        x=idx_dates, y=hist_idx["score"].where(elevated_mask),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.10)",
        line=dict(width=0), name="ELEVATED", showlegend=True,
    )
    # Linea principale
    fig_hist.add_scatter(
        x=idx_dates, y=hist_idx["score"],
        line=dict(color="#06b6d4", width=1.8),
        name="Disruption Index",
    )

    # Soglie
    fig_hist.add_hline(y=55, line=dict(color="#f59e0b",dash="dash",width=1),
                       annotation_text="ELEVATED threshold",
                       annotation_font_color="#f59e0b",annotation_font_size=9)
    fig_hist.add_hline(y=70, line=dict(color="#ef4444",dash="dash",width=1),
                       annotation_text="CRITICAL threshold",
                       annotation_font_color="#ef4444",annotation_font_size=9)

    # Annotazioni eventi geopolitici
    for ev in KNOWN_EVENTS:
        ev_date = pd.Timestamp(ev["date"])
        if ev_date < idx_dates[0] or ev_date > idx_dates[-1]:
            continue
        color_ev = "#ef4444" if ev["severity"]=="CRITICAL" else "#f59e0b"
        fig_hist.add_vline(
            x=ev_date, line=dict(color=color_ev, dash="dot", width=1.5),
        )
        # Trova lo score in quella data
        try:
            score_ev = hist_idx.loc[ev_date, "score"] if ev_date in hist_idx.index else 50
        except:
            score_ev = 50
        fig_hist.add_annotation(
            x=ev_date, y=float(score_ev)+8,
            text=ev["label"],
            showarrow=True,
            arrowhead=2, arrowsize=0.8,
            arrowcolor=color_ev,
            font=dict(size=8, color=color_ev),
            bgcolor="rgba(21,29,46,0.8)",
            bordercolor=color_ev,
            borderwidth=1,
        )

    fig_hist.update_layout(
        **PLOTLY_DARK, height=420,
        yaxis=dict(range=[0,105], title="Disruption Index", **PLOTLY_DARK["yaxis"]),
        title=dict(text=""),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Componenti storiche ───────────────────────────────────────────────────
    section("COMPONENTI STORICHE — SPREAD / VOLATILITÀ / XLE")

    fig_comp = make_subplots(rows=3, cols=1,
                              subplot_titles=["Spread Brent-WTI Score","Volatilità Brent Score","XLE Anomalia Score"],
                              vertical_spacing=0.08, shared_xaxes=True)

    pairs = [("spread_score","#f59e0b"),("vol_score","#ef4444"),("xle_score","#10b981")]
    for i, (col_name, color) in enumerate(pairs,1):
        if col_name in hist_idx.columns:
            fig_comp.add_scatter(x=hist_idx.index, y=hist_idx[col_name],
                                 line=dict(color=color,width=1.5), name=col_name,
                                 showlegend=False, row=i, col=1)
            fig_comp.add_hline(y=55, line=dict(color="#4a5a75",dash="dash",width=0.8), row=i, col=1)
            fig_comp.add_hline(y=70, line=dict(color="#ef4444",dash="dash",width=0.8), row=i, col=1)

    fig_comp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(21,29,46,0.6)",
        font=dict(family="JetBrains Mono",color="#8a9ab5",size=10),
        height=400, showlegend=False, margin=dict(l=50,r=20,t=40,b=30),
    )
    for i in range(1,4):
        fig_comp.update_xaxes(gridcolor="#1e2d47",linecolor="#1e2d47",row=i,col=1)
        fig_comp.update_yaxes(gridcolor="#1e2d47",linecolor="#1e2d47",range=[0,105],row=i,col=1)

    st.plotly_chart(fig_comp, use_container_width=True)

    # Tabella eventi
    section("EVENTI GEOPOLITICI DI RIFERIMENTO")
    ev_df = pd.DataFrame(KNOWN_EVENTS)
    ev_df.columns = ["Data","Evento","Severità"]
    st.dataframe(ev_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CORRELAZIONE BRENT
# ═══════════════════════════════════════════════════════════════════════════════
elif "Correlazione" in page:
    top_bar("🔁 Correlazione Brent", "Forward return Brent dopo spike Disruption Index (score ≥ 70)")

    with st.spinner("Calcolo correlazione..."):
        fin_df   = load_financial_data()
        hist_idx = load_historical_index(str(len(fin_df)))

    if hist_idx.empty or fin_df.empty:
        st.error("Dati insufficienti per l'analisi."); st.stop()

    threshold = st.slider("Soglia spike (score ≥)", 55, 85, 70, 5)
    corr      = compute_brent_correlation(hist_idx, fin_df,
                                          horizons=[1,3,5,10,15], threshold=threshold)

    if not corr:
        st.warning(f"Nessun episodio con score ≥ {threshold}. Abbassa la soglia.")
        st.stop()

    st.session_state.correlation = corr

    H_list = sorted(corr.keys())
    n_ep   = corr[H_list[0]]["n_episodes"]

    section(f"EPISODI SPIKE RILEVATI (score ≥ {threshold}): {n_ep}")

    # KPI per orizzonte
    cols_kpi = st.columns(len(H_list))
    for col, H in zip(cols_kpi, H_list):
        m = corr[H]
        c = "#10b981" if m["avg_ret"]>0 else "#ef4444"
        col.markdown(f"<div class='kpi-card'><div class='kpi-label'>H={H} giorni</div>"
                     f"<div class='kpi-value' style='color:{c};font-size:1.3rem;'>{m['avg_ret']:+.2f}%</div>"
                     f"<div class='kpi-sub'>Hit Rate: {m['hit_rate']:.1%} | p={m['p_val']:.3f} "
                     f"{'✅' if m['sig'] else '⚠️'}</div></div>",
                     unsafe_allow_html=True)

    # Distribuzione forward return
    section("DISTRIBUZIONE FORWARD RETURN BRENT — SPIKE vs BASELINE")
    H_sel = st.select_slider("Orizzonte", options=H_list, value=H_list[2] if len(H_list)>2 else H_list[0])
    m     = corr[H_sel]

    col_dist, col_prof = st.columns(2)
    with col_dist:
        fig_d = go.Figure()
        fig_d.add_histogram(x=m["y_all"],  nbinsx=60, marker_color="#4a5a75", opacity=0.45,
                            histnorm="probability density", name="Baseline")
        fig_d.add_histogram(x=m["y_spike"],nbinsx=40, marker_color="#ef4444", opacity=0.85,
                            histnorm="probability density", name=f"Spike (n={m['n_episodes']})")
        fig_d.add_vline(x=m["avg_ret"], line=dict(color="#ef4444",dash="dash",width=2),
                        annotation_text=f"μ spike={m['avg_ret']:+.2f}%",
                        annotation_font_color="#ef4444",annotation_font_size=10)
        fig_d.add_vline(x=m["avg_baseline"], line=dict(color="#4a5a75",dash="dot",width=1.5),
                        annotation_text=f"μ base={m['avg_baseline']:+.2f}%",
                        annotation_font_color="#4a5a75",annotation_font_size=10)
        fig_d.add_vline(x=0, line=dict(color="#8a9ab5",width=0.8))
        fig_d.update_layout(**PLOTLY_DARK, height=280, barmode="overlay",
                            xaxis_title=f"Brent Return H={H_sel}gg (%)",
                            yaxis_title="Densità", title=dict(text=""))
        st.plotly_chart(fig_d, use_container_width=True)

    with col_prof:
        section("PROFILO H=1→MAX")
        avgs  = [corr[H]["avg_ret"]      for H in H_list]
        bases = [corr[H]["avg_baseline"] for H in H_list]
        hrs   = [corr[H]["hit_rate"]*100 for H in H_list]

        fig_p = make_subplots(specs=[[{"secondary_y":True}]])
        fig_p.add_scatter(x=H_list, y=bases, mode="lines+markers",
                          line=dict(color="#4a5a75",dash="dash",width=1.5),
                          marker=dict(size=5), name="Baseline", secondary_y=False)
        fig_p.add_scatter(x=H_list, y=avgs, mode="lines+markers",
                          line=dict(color="#ef4444",width=2.5),
                          marker=dict(size=8), fill="tonexty",
                          fillcolor="rgba(239,68,68,0.08)", name="Post-spike", secondary_y=False)
        fig_p.add_scatter(x=H_list, y=hrs, mode="lines+markers",
                          line=dict(color="#f59e0b",width=1.5,dash="dot"),
                          marker=dict(size=6), name="Hit Rate (%)", secondary_y=True)
        fig_p.add_hline(y=0, line=dict(color="#4a5a75",width=0.8), secondary_y=False)
        fig_p.update_layout(**PLOTLY_DARK, height=280, title=dict(text=""))
        fig_p.update_yaxes(title_text="Avg Ret Brent (%)", gridcolor="#1e2d47", secondary_y=False)
        fig_p.update_yaxes(title_text="Hit Rate (%)", gridcolor="rgba(0,0,0,0)", secondary_y=True)
        st.plotly_chart(fig_p, use_container_width=True)

    # Tabella metriche
    section("TABELLA METRICHE COMPLETE")
    rows = [{"H":H,"N episodi":c["n_episodes"],"Avg Ret Brent":f"{c['avg_ret']:+.3f}%",
             "Baseline":f"{c['avg_baseline']:+.3f}%","Hit Rate":f"{c['hit_rate']:.1%}",
             "T-stat":f"{c['t_stat']:.2f}","P-value":f"{c['p_val']:.4f}",
             "Sig.":"✅" if c["sig"] else "⚠️"} for H,c in corr.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("""<div class='disclaimer'>
        ⚠️ L'analisi di correlazione è basata su dati storici finanziari.
        Non implica causalità diretta. Strumento di supporto alla ricerca macro.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — METODOLOGIA
# ═══════════════════════════════════════════════════════════════════════════════
elif "Metodologia" in page:
    top_bar("📖 Metodologia", "Come funziona il Hormuz Disruption Monitor")

    st.markdown("""
    <div style='background:rgba(6,182,212,0.05);border:1px solid rgba(6,182,212,0.2);
                border-radius:8px;padding:1.2rem 1.6rem;margin-bottom:1.5rem;
                font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#06b6d4;
                text-align:center;letter-spacing:0.05em;'>
        Dati AIS live (AISStream.io)
        &nbsp;→&nbsp; Posizioni tanker in Hormuz
        &nbsp;→&nbsp; Densità + velocità media
        &nbsp;+&nbsp; Proxy finanziari (Brent, WTI, XLE)
        &nbsp;→&nbsp; Z-score rolling
        &nbsp;→&nbsp; Disruption Index 0-100
        &nbsp;→&nbsp; NORMAL / ELEVATED / CRITICAL
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        for title, body in [
            ("📡 Fonte dati AIS",
             "AISStream.io fornisce un feed WebSocket gratuito di posizioni AIS in tempo reale. "
             "Il sistema si connette all'endpoint <b>wss://stream.aisstream.io/v0/stream</b> "
             "con un bounding box sul Golfo di Oman e Stretto di Hormuz "
             "([21°N-27°N, 54°E-60°E]). Raccoglie PositionReport e ShipStaticData "
             "per 45 secondi, filtrando per navi tipo 80-89 (tanker) e 70-79 (cargo)."),
            ("💹 Proxy finanziari",
             "Quattro serie giornaliere da Yahoo Finance: "
             "<b>Brent futures (BZ=F)</b> — prezzo petrolio di riferimento europeo; "
             "<b>WTI futures (CL=F)</b> — petrolio USA; "
             "<b>Spread Brent-WTI</b> — si allarga in caso di tensione geopolitica mediorientale; "
             "<b>XLE Energy ETF</b> — reazione del settore energy nel suo complesso."),
        ]:
            st.markdown(f"<div class='kpi-card' style='margin-bottom:1rem;'>"
                        f"<div style='font-family:Syne,sans-serif;color:#06b6d4;font-size:0.95rem;"
                        f"margin-bottom:0.5rem;'>{title}</div>"
                        f"<div style='font-size:0.76rem;color:#8a9ab5;line-height:1.7;'>{body}</div>"
                        f"</div>", unsafe_allow_html=True)

    with col2:
        for title, body in [
            ("📐 Costruzione dell'indice",
             "Ogni componente viene trasformata in z-score rolling (finestra 60gg) "
             "e poi normalizzata in un punteggio 0-100 con la formula: "
             "<b>score = clip(50 + z × 15, 0, 100)</b>. "
             "Le cinque componenti vengono aggregate con pesi fissi: "
             "Spread Brent-WTI 30%, Volatilità Brent 25%, XLE anomalia 20%, "
             "Densità traffico AIS 15%, Velocità media navi 10%."),
            ("🔒 Soglie e livelli",
             "<b>NORMAL</b> (score < 55): traffico e mercati nella norma. "
             "<b>ELEVATED</b> (55-70): segnali di attenzione su uno o più proxy. "
             "<b>CRITICAL</b> (≥ 70): anomalie significative, lettura potenzialmente disruptiva. "
             "Le soglie sono calibrate empiricamente sul periodo 2018-2024 "
             "confrontando i picchi dell'indice con eventi geopolitici noti."),
        ]:
            st.markdown(f"<div class='kpi-card' style='margin-bottom:1rem;'>"
                        f"<div style='font-family:Syne,sans-serif;color:#06b6d4;font-size:0.95rem;"
                        f"margin-bottom:0.5rem;'>{title}</div>"
                        f"<div style='font-size:0.76rem;color:#8a9ab5;line-height:1.7;'>{body}</div>"
                        f"</div>", unsafe_allow_html=True)

    section("PARAMETRI CORRENTI")
    params_df = pd.DataFrame([
        {"Parametro":"Bounding box","Valore":"[21°N-27°N, 54°E-60°E]","Descrizione":"Stretto Hormuz + Golfo di Oman"},
        {"Parametro":"Tipi nave","Valore":"80-89 (tanker), 70-79 (cargo)","Descrizione":"Codici AIS tipo nave"},
        {"Parametro":"Secondi raccolta AIS","Valore":"45","Descrizione":"Durata snapshot WebSocket"},
        {"Parametro":"Baseline finestra","Valore":"60 giorni","Descrizione":"Rolling window per z-score"},
        {"Parametro":"Peso spread Brent-WTI","Valore":"30%","Descrizione":"Componente principale"},
        {"Parametro":"Peso volatilità Brent","Valore":"25%","Descrizione":""},
        {"Parametro":"Peso XLE anomalia","Valore":"20%","Descrizione":""},
        {"Parametro":"Peso densità AIS","Valore":"15%","Descrizione":"Richiede snapshot live"},
        {"Parametro":"Peso velocità media","Valore":"10%","Descrizione":"Richiede snapshot live"},
        {"Parametro":"Soglia ELEVATED","Valore":"55","Descrizione":"Score ≥ 55"},
        {"Parametro":"Soglia CRITICAL","Valore":"70","Descrizione":"Score ≥ 70"},
        {"Parametro":"Fonte AIS","Valore":"AISStream.io","Descrizione":"Gratuito, copertura terrestre"},
        {"Parametro":"Upgrade istituzionale","Valore":"Spire / ExactEarth","Descrizione":"Dati satellitari, copertura globale"},
    ])
    st.dataframe(params_df, use_container_width=True, hide_index=True)

    st.markdown("""<div class='disclaimer'>
        ⚠️ Il Hormuz Disruption Monitor è uno strumento di analisi e supporto alla ricerca.
        I dati AIS gratuiti (AISStream.io) hanno copertura limitata nelle acque internazionali
        lontano dalla costa — per uso istituzionale si raccomanda l'upgrade a dati satellitari
        (Spire Maritime, ExactEarth) con copertura globale completa.
        Non costituisce consulenza di investimento ai sensi di MiFID II.
    </div>""", unsafe_allow_html=True)

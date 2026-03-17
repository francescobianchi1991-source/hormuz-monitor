"""
hormuz_engine.py
================
Modulo Python puro — Hormuz Disruption Monitor.
Nessuna UI. Importato dalla web app Streamlit.

Funzioni esportate:
    fetch_ais_snapshot(api_key, seconds, bbox, only_tankers) -> DataFrame navi
    fetch_financial_proxies(start, end)                      -> DataFrame proxy
    compute_disruption_index(n_vessels, avg_speed, df_fin)   -> dict indice
    build_historical_index(df_fin, baseline_window)          -> DataFrame serie storica
    compute_brent_correlation(df_index, df_fin, horizons)    -> dict correlazione
    classify_disruption(score)                               -> str livello
    get_vessel_type_label(code)                              -> str tipo nave
    get_disruption_color(level)                              -> str hex color
"""

import asyncio
import json
import warnings
import websockets
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone
from scipy import stats as spstats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# COSTANTI
# ─────────────────────────────────────────────────────────────────────────────

HORMUZ_BBOX    = [[21.0, 54.0], [27.0, 60.0]]   # Stretto di Hormuz + Golfo di Oman
GULF_BBOX      = [[22.0, 48.0], [30.0, 60.0]]   # Golfo Persico completo
TANKER_TYPES   = set(range(80, 90))
CARGO_TYPES    = set(range(70, 80))
TARGET_TYPES   = TANKER_TYPES | CARGO_TYPES
BASELINE_VESSELS    = 18
BASELINE_VESSELS_STD= 5
BASELINE_SPEED      = 12.0
BASELINE_SPEED_STD  = 3.0

FINANCIAL_PROXIES = {
    "brent": "BZ=F",
    "wti":   "CL=F",
    "xle":   "XLE",
    "xop":   "XOP",
    "uup":   "UUP",
}

KNOWN_EVENTS = [
    {"date":"2019-05-12","label":"Attacchi tanker UAE",        "severity":"HIGH"},
    {"date":"2019-06-13","label":"Attacchi tanker Golfo Oman", "severity":"HIGH"},
    {"date":"2019-09-14","label":"Attacco Abqaiq-Khurais",     "severity":"CRITICAL"},
    {"date":"2020-01-03","label":"Uccisione Soleimani",        "severity":"CRITICAL"},
    {"date":"2020-01-08","label":"Risposta missilistica Iran",  "severity":"HIGH"},
    {"date":"2021-07-29","label":"Attacco Mercer Street",      "severity":"HIGH"},
    {"date":"2022-02-24","label":"Invasione Ucraina",          "severity":"HIGH"},
    {"date":"2023-10-07","label":"Attacco Hamas",              "severity":"HIGH"},
    {"date":"2023-11-19","label":"Sequestro Galaxy Leader",    "severity":"HIGH"},
    {"date":"2024-01-09","label":"Escalation Houthi",          "severity":"CRITICAL"},
]

# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO AIS — RACCOLTA DATI WEBSOCKET
# ─────────────────────────────────────────────────────────────────────────────

async def _stream_ais(api_key: str, seconds: int, bbox: list) -> list:
    url      = "wss://stream.aisstream.io/v0/stream"
    vessels  = {}
    deadline = asyncio.get_event_loop().time() + seconds

    subscribe_msg = {
        "APIKey":             api_key,
        "BoundingBoxes":      [bbox],
        "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
    }

    try:
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            await ws.send(json.dumps(subscribe_msg))
            while asyncio.get_event_loop().time() < deadline:
                try:
                    raw  = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    msg  = json.loads(raw)
                    meta = msg.get("MetaData", {})
                    mmsi = str(meta.get("MMSI", ""))
                    if not mmsi:
                        continue
                    if mmsi not in vessels:
                        vessels[mmsi] = {
                            "mmsi":        mmsi,
                            "name":        meta.get("ShipName","").strip(),
                            "lat":         meta.get("latitude",  None),
                            "lon":         meta.get("longitude", None),
                            "speed":       None,
                            "heading":     None,
                            "ship_type":   None,
                            "destination": "",
                            "nav_status":  None,
                            "timestamp":   meta.get("time_utc",""),
                        }
                    msg_type = msg.get("MessageType","")
                    if msg_type == "PositionReport":
                        pr = msg.get("Message",{}).get("PositionReport",{})
                        vessels[mmsi].update({
                            "lat":       pr.get("Latitude",          vessels[mmsi]["lat"]),
                            "lon":       pr.get("Longitude",         vessels[mmsi]["lon"]),
                            "speed":     pr.get("Sog",               None),
                            "heading":   pr.get("TrueHeading",       None),
                            "nav_status":pr.get("NavigationalStatus",None),
                            "timestamp": meta.get("time_utc",        vessels[mmsi]["timestamp"]),
                        })
                    elif msg_type == "ShipStaticData":
                        sd = msg.get("Message",{}).get("ShipStaticData",{})
                        vessels[mmsi].update({
                            "name":        sd.get("Name",       vessels[mmsi]["name"]).strip(),
                            "ship_type":   sd.get("Type",       None),
                            "destination": sd.get("Destination","").strip(),
                        })
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    continue
    except Exception:
        pass

    return list(vessels.values())


def fetch_ais_snapshot(
    api_key:      str,
    seconds:      int  = 45,
    bbox:         list = None,
    only_tankers: bool = True,
) -> pd.DataFrame:
    """
    Raccoglie uno snapshot delle navi nell'area Hormuz
    dal WebSocket AISStream.io per `seconds` secondi.
    """
    if bbox is None:
        bbox = HORMUZ_BBOX
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        vessels = loop.run_until_complete(_stream_ais(api_key, seconds, bbox))
        loop.close()
    except Exception:
        return pd.DataFrame()

    if not vessels:
        return pd.DataFrame()

    df = pd.DataFrame(vessels)
    df = df.dropna(subset=["lat","lon"]).copy()
    df = df[(df["lat"] != 0) | (df["lon"] != 0)]

    # Filtra coordinate fuori dal range marino plausibile per il Golfo
    # Rimuove errori GPS che posizionano navi sulla terraferma
    df = df[(df["lat"] >= 21.0) & (df["lat"] <= 31.0)].copy()
    df = df[(df["lon"] >= 47.0) & (df["lon"] <= 62.0)].copy()

    df["is_tanker"] = df["ship_type"].apply(
        lambda x: int(x) in TARGET_TYPES if pd.notna(x) else False
    )
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    df.loc[df["speed"] > 50, "speed"] = np.nan
    df["ship_type_label"] = df["ship_type"].apply(get_vessel_type_label)
    df["nav_status_label"] = df["nav_status"].apply(format_nav_status)

    if only_tankers:
        df = df[df["is_tanker"]].copy()

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATI FINANZIARI
# ─────────────────────────────────────────────────────────────────────────────

def fetch_financial_proxies(start: str = None, end: str = None) -> pd.DataFrame:
    """Scarica i proxy finanziari da yfinance."""
    if start is None:
        start = (datetime.today() - timedelta(days=365*3)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    frames = {}
    for name, ticker in FINANCIAL_PROXIES.items():
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            frames[name] = df["Close"].rename(name)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames.values(), axis=1)
    result.index.name = "date"
    result = result.dropna(how="all")

    if "brent" in result.columns and "wti" in result.columns:
        result["brent_wti_spread"] = result["brent"] - result["wti"]
    if "brent" in result.columns:
        result["brent_ret_1d"] = result["brent"].pct_change()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# DISRUPTION INDEX — CALCOLO PUNTUALE
# ─────────────────────────────────────────────────────────────────────────────

def compute_disruption_index(
    n_vessels_now:   int,
    avg_speed_now:   float,
    df_fin:          pd.DataFrame,
    baseline_window: int = 60,
) -> dict:
    """
    Calcola il Disruption Index corrente (0-100).

    Componenti e pesi:
        spread_brent_wti  30% — proxy tensione geografica
        brent_volatility  25% — volatilità mercato
        xle_anomaly       20% — reazione settore energy
        vessel_density    15% — traffico AIS in zona
        avg_speed         10% — comportamento navi
    """
    if df_fin.empty:
        return {"score":50,"level":"NORMAL","components":{},"narrative":"Dati insufficienti","timestamp":""}

    df = df_fin.copy().sort_index()
    components = {}

    def z_to_score(z): return min(100, max(0, 50 + float(z) * 15))

    # Spread Brent-WTI
    if "brent_wti_spread" in df.columns:
        s = df["brent_wti_spread"].dropna()
        if len(s) >= baseline_window:
            mu = s.rolling(baseline_window).mean().iloc[-1]
            sg = s.rolling(baseline_window).std().iloc[-1]
            z  = (s.iloc[-1] - mu) / sg if sg > 0 else 0
            components["spread_brent_wti"] = {"z_score":round(float(z),3),"value":round(float(s.iloc[-1]),3),"score":z_to_score(z),"label":"Spread Brent-WTI"}

    # Volatilità Brent
    if "brent_ret_1d" in df.columns:
        r = df["brent_ret_1d"].abs().dropna()
        if len(r) >= baseline_window:
            mu = r.rolling(baseline_window).mean().iloc[-1]
            sg = r.rolling(baseline_window).std().iloc[-1]
            z  = (r.iloc[-1] - mu) / sg if sg > 0 else 0
            components["brent_volatility"] = {"z_score":round(float(z),3),"value":round(float(r.iloc[-1]*100),3),"score":z_to_score(z),"label":"Volatilità Brent"}

    # XLE anomalia
    if "xle" in df.columns:
        x = df["xle"].pct_change().dropna()
        if len(x) >= baseline_window:
            mu = x.rolling(baseline_window).mean().iloc[-1]
            sg = x.rolling(baseline_window).std().iloc[-1]
            z  = -(x.iloc[-1] - mu) / sg if sg > 0 else 0  # drop XLE = disruption
            components["xle_anomaly"] = {"z_score":round(float(z),3),"value":round(float(x.iloc[-1]*100),3),"score":z_to_score(z),"label":"XLE Anomalia"}

    # Densità traffico AIS
    if n_vessels_now > 0:
        z = -(n_vessels_now - BASELINE_VESSELS) / BASELINE_VESSELS_STD
        components["vessel_density"] = {"z_score":round(float(z),3),"value":n_vessels_now,"score":z_to_score(z),"label":f"Densità traffico ({n_vessels_now} navi)"}

    # Velocità media
    if avg_speed_now > 0:
        z = -(avg_speed_now - BASELINE_SPEED) / BASELINE_SPEED_STD
        components["avg_speed"] = {"z_score":round(float(z),3),"value":round(float(avg_speed_now),1),"score":z_to_score(z),"label":f"Velocità media ({avg_speed_now:.1f} kn)"}

    # Aggregazione
    weights = {"spread_brent_wti":0.30,"brent_volatility":0.25,"xle_anomaly":0.20,"vessel_density":0.15,"avg_speed":0.10}
    tw = sum(weights[k] for k in components if k in weights)
    ws = sum(components[k]["score"] * weights[k] for k in components if k in weights)
    score = round(ws / tw if tw > 0 else 50.0, 1)
    level = classify_disruption(score)

    return {
        "score":      score,
        "level":      level,
        "components": components,
        "narrative":  _build_narrative(score, level, components),
        "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def _build_narrative(score, level, components):
    parts = []
    if "spread_brent_wti" in components:
        c = components["spread_brent_wti"]
        if abs(c["z_score"]) > 1.5:
            parts.append(f"Spread Brent-WTI {'elevato' if c['z_score']>0 else 'compresso'} (z={c['z_score']:.1f}σ)")
    if "brent_volatility" in components:
        c = components["brent_volatility"]
        if c["z_score"] > 1.5:
            parts.append(f"Volatilità Brent anomala ({c['value']:.2f}%, z={c['z_score']:.1f}σ)")
    if "vessel_density" in components:
        c = components["vessel_density"]
        if abs(c["z_score"]) > 1.0:
            parts.append(f"Traffico tanker {'sotto' if c['z_score']>0 else 'sopra'} baseline ({c['value']} navi)")
    if "avg_speed" in components:
        c = components["avg_speed"]
        if c["z_score"] > 1.5:
            parts.append(f"Velocità media bassa ({c['value']:.1f} kn)")
    if not parts:
        return "Traffico regolare. Nessuna anomalia rilevata." if level=="NORMAL" else "Segnali di tensione. Monitorare."
    intro = {"NORMAL":"Situazione nella norma.","ELEVATED":"⚠️ Segnali di attenzione:","CRITICAL":"🔴 Anomalie significative:"}.get(level,"")
    return f"{intro} {' | '.join(parts)}."


# ─────────────────────────────────────────────────────────────────────────────
# SERIE STORICA
# ─────────────────────────────────────────────────────────────────────────────

def build_historical_index(df_fin: pd.DataFrame, baseline_window: int = 60) -> pd.DataFrame:
    """Costruisce la serie storica del Disruption Index dai proxy finanziari."""
    if df_fin.empty:
        return pd.DataFrame()

    df = df_fin.copy().sort_index()
    rows = []

    for i in range(baseline_window, len(df)):
        w = df.iloc[:i+1]
        scores = {}

        if "brent_wti_spread" in w.columns:
            s = w["brent_wti_spread"].dropna()
            mu = s.rolling(baseline_window).mean().iloc[-1]
            sg = s.rolling(baseline_window).std().iloc[-1]
            z  = (s.iloc[-1]-mu)/sg if sg>0 else 0
            scores["spread"] = min(100, max(0, 50+float(z)*15))

        if "brent_ret_1d" in w.columns:
            r = w["brent_ret_1d"].abs().dropna()
            mu = r.rolling(baseline_window).mean().iloc[-1]
            sg = r.rolling(baseline_window).std().iloc[-1]
            z  = (r.iloc[-1]-mu)/sg if sg>0 else 0
            scores["vol"] = min(100, max(0, 50+float(z)*15))

        if "xle" in w.columns:
            x = w["xle"].pct_change().dropna()
            mu = x.rolling(baseline_window).mean().iloc[-1]
            sg = x.rolling(baseline_window).std().iloc[-1]
            z  = -(x.iloc[-1]-mu)/sg if sg>0 else 0
            scores["xle"] = min(100, max(0, 50+float(z)*15))

        if not scores:
            continue

        ws_ = {"spread":0.40,"vol":0.35,"xle":0.25}
        tw  = sum(ws_[k] for k in scores)
        agg = sum(scores[k]*ws_[k] for k in scores)/tw if tw>0 else 50

        rows.append({
            "date":         df.index[i],
            "score":        round(agg,1),
            "level":        classify_disruption(agg),
            "spread_score": scores.get("spread",np.nan),
            "vol_score":    scores.get("vol",   np.nan),
            "xle_score":    scores.get("xle",   np.nan),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("date")


# ─────────────────────────────────────────────────────────────────────────────
# CORRELAZIONE BRENT
# ─────────────────────────────────────────────────────────────────────────────

def compute_brent_correlation(
    df_index:  pd.DataFrame,
    df_fin:    pd.DataFrame,
    horizons:  list = [1,3,5,10],
    threshold: float = 70.0,
) -> dict:
    """
    Analizza il forward return del Brent dopo spike del Disruption Index.
    """
    if df_index.empty or df_fin.empty:
        return {}

    df = df_index.join(df_fin[["brent"]], how="inner").dropna(subset=["score","brent"])
    if len(df) < 20:
        return {}

    for H in horizons:
        df[f"brent_fwd_t{H}"] = df["brent"].pct_change(H).shift(-H) * 100

    spike_mask = df["score"] >= threshold
    df_spike   = df[spike_mask].copy()
    if len(df_spike) < 3:
        return {}

    out = {}
    for H in horizons:
        col = f"brent_fwd_t{H}"
        if col not in df.columns:
            continue
        y_spike = df_spike[col].dropna()
        y_all   = df[col].dropna()
        if len(y_spike) < 3:
            continue
        t, p = spstats.ttest_1samp(y_spike, 0)
        out[H] = {
            "n_episodes":   len(y_spike),
            "avg_ret":      round(float(y_spike.mean()),3),
            "avg_baseline": round(float(y_all.mean()),  3),
            "hit_rate":     round(float((y_spike>0).mean()),3),
            "t_stat":       round(float(t),2),
            "p_val":        round(float(p),4),
            "sig":          float(p)<0.10,
            "y_spike":      y_spike,
            "y_all":        y_all,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def classify_disruption(score: float) -> str:
    if score >= 70:   return "CRITICAL"
    elif score >= 55: return "ELEVATED"
    else:             return "NORMAL"

def get_disruption_color(level: str) -> str:
    return {"NORMAL":"#10b981","ELEVATED":"#f59e0b","CRITICAL":"#ef4444"}.get(level,"#8a9ab5")

def get_disruption_emoji(level: str) -> str:
    return {"NORMAL":"🟢","ELEVATED":"🟡","CRITICAL":"🔴"}.get(level,"⚪")

def get_hormuz_bbox() -> list:
    return HORMUZ_BBOX

def get_vessel_type_label(code) -> str:
    if pd.isna(code): return "Unknown"
    try: code = int(code)
    except: return "Unknown"
    if 80<=code<=89: return "Tanker"
    elif 70<=code<=79: return "Cargo"
    elif 60<=code<=69: return "Passenger"
    elif 30<=code<=39: return "Fishing"
    else: return f"Type {code}"

def format_nav_status(code) -> str:
    if pd.isna(code): return "Unknown"
    STATUS = {0:"Under way",1:"At anchor",2:"Not under command",
              3:"Restricted manoeuvrability",5:"Moored",6:"Aground",15:"Undefined"}
    return STATUS.get(int(code), f"Status {int(code)}")


# ─────────────────────────────────────────────────────────────────────────────
# RACCOLTA PROGRESSIVA — per aggiornamento live in Streamlit
# ─────────────────────────────────────────────────────────────────────────────

async def _stream_ais_progressive(
    api_key:          str,
    total_seconds:    int,
    chunk_seconds:    int,
    bbox:             list,
    on_chunk_ready,   # callback(df_chunk, elapsed, total) chiamata ogni chunk
):
    """
    Raccoglie dati AIS in chunk successivi e chiama on_chunk_ready
    ad ogni chunk con il DataFrame aggiornato.
    """
    url = "wss://stream.aisstream.io/v0/stream"
    all_vessels = {}
    start_time  = asyncio.get_event_loop().time()
    deadline    = start_time + total_seconds
    next_chunk  = start_time + chunk_seconds

    subscribe_msg = {
        "APIKey":             api_key,
        "BoundingBoxes":      [bbox],
        "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
    }

    try:
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            await ws.send(json.dumps(subscribe_msg))

            while asyncio.get_event_loop().time() < deadline:
                try:
                    raw  = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    msg  = json.loads(raw)
                    meta = msg.get("MetaData", {})
                    mmsi = str(meta.get("MMSI", ""))
                    if not mmsi:
                        continue

                    if mmsi not in all_vessels:
                        all_vessels[mmsi] = {
                            "mmsi":        mmsi,
                            "name":        meta.get("ShipName","").strip(),
                            "lat":         meta.get("latitude",  None),
                            "lon":         meta.get("longitude", None),
                            "speed":       None,
                            "heading":     None,
                            "ship_type":   None,
                            "destination": "",
                            "nav_status":  None,
                            "timestamp":   meta.get("time_utc",""),
                        }

                    msg_type = msg.get("MessageType","")
                    if msg_type == "PositionReport":
                        pr = msg.get("Message",{}).get("PositionReport",{})
                        all_vessels[mmsi].update({
                            "lat":       pr.get("Latitude",          all_vessels[mmsi]["lat"]),
                            "lon":       pr.get("Longitude",         all_vessels[mmsi]["lon"]),
                            "speed":     pr.get("Sog",               None),
                            "heading":   pr.get("TrueHeading",       None),
                            "nav_status":pr.get("NavigationalStatus",None),
                            "timestamp": meta.get("time_utc",        all_vessels[mmsi]["timestamp"]),
                        })
                    elif msg_type == "ShipStaticData":
                        sd = msg.get("Message",{}).get("ShipStaticData",{})
                        all_vessels[mmsi].update({
                            "name":        sd.get("Name",       all_vessels[mmsi]["name"]).strip(),
                            "ship_type":   sd.get("Type",       None),
                            "destination": sd.get("Destination","").strip(),
                        })

                except asyncio.TimeoutError:
                    pass
                except Exception:
                    pass

                # Ogni chunk_seconds chiama il callback con i dati accumulati
                now = asyncio.get_event_loop().time()
                if now >= next_chunk:
                    elapsed = int(now - start_time)
                    df_acc  = _postprocess_vessels(list(all_vessels.values()), bbox)
                    on_chunk_ready(df_acc, elapsed, total_seconds)
                    next_chunk = now + chunk_seconds

    except Exception:
        pass

    # Chunk finale
    df_final = _postprocess_vessels(list(all_vessels.values()), bbox)
    on_chunk_ready(df_final, total_seconds, total_seconds)
    return df_final


def _postprocess_vessels(vessels: list, bbox: list) -> pd.DataFrame:
    """Pulizia e normalizzazione lista vessels -> DataFrame."""
    if not vessels:
        return pd.DataFrame()

    df = pd.DataFrame(vessels)
    df = df.dropna(subset=["lat","lon"]).copy()
    df = df[(df["lat"] != 0) | (df["lon"] != 0)]

    # Filtro coordinate marine plausibili per il Golfo
    lat_min = min(bbox[0][0], bbox[1][0]) - 1
    lat_max = max(bbox[0][0], bbox[1][0]) + 1
    lon_min = min(bbox[0][1], bbox[1][1]) - 1
    lon_max = max(bbox[0][1], bbox[1][1]) + 1
    df = df[
        (df["lat"] >= lat_min) & (df["lat"] <= lat_max) &
        (df["lon"] >= lon_min) & (df["lon"] <= lon_max)
    ].copy()

    if df.empty:
        return df

    df["is_tanker"] = df["ship_type"].apply(
        lambda x: int(x) in TARGET_TYPES if pd.notna(x) else False
    )
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    df.loc[df["speed"] > 50, "speed"] = np.nan
    df["ship_type_label"]  = df["ship_type"].apply(get_vessel_type_label)
    df["nav_status_label"] = df["nav_status"].apply(format_nav_status)

    return df.reset_index(drop=True)


def fetch_ais_progressive(
    api_key:       str,
    total_seconds: int  = 300,
    chunk_seconds: int  = 20,
    bbox:          list = None,
    on_chunk_ready       = None,
) -> pd.DataFrame:
    """
    Raccolta progressiva: rimane connesso per total_seconds,
    chiama on_chunk_ready ogni chunk_seconds con il DataFrame
    aggiornato di tutte le navi viste finora.

    on_chunk_ready(df, elapsed_seconds, total_seconds)
    """
    if bbox is None:
        bbox = GULF_BBOX

    def _cb(df, elapsed, total):
        if on_chunk_ready:
            on_chunk_ready(df, elapsed, total)

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            _stream_ais_progressive(api_key, total_seconds, chunk_seconds, bbox, _cb)
        )
        loop.close()
        return result
    except Exception:
        return pd.DataFrame()

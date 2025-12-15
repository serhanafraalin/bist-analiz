import math
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ================================
# App Config
# ================================
st.set_page_config(page_title="BIST 50 Tarayıcı", layout="wide")
st.title("📋 BIST 50 Tarayıcı — Kart Kart Liste")
st.caption(
    "Bu sistem yatırım tavsiyesi değildir. "
    "'Ben olsam' bölümü yalnızca **örnek plan şablonudur**; karar %100 sende."
)

# ================================
# BIST 50 (yaklaşık, dönemsel değişebilir)
# ================================
BIST50 = [
    "AKBNK.IS","ALARK.IS","ARCLK.IS","ASELS.IS","ASTOR.IS",
    "BIMAS.IS","BRISA.IS","CCOLA.IS","DOAS.IS","EKGYO.IS",
    "ENJSA.IS","ENKAI.IS","EREGL.IS","FROTO.IS","GARAN.IS",
    "GUBRF.IS","HEKTS.IS","ISCTR.IS","KCHOL.IS","KOZAA.IS",
    "KOZAL.IS","KRDMD.IS","MAVI.IS","ODAS.IS","OTKAR.IS",
    "PETKM.IS","PGSUS.IS","SAHOL.IS","SASA.IS","SISE.IS",
    "SKBNK.IS","SMRTG.IS","SOKM.IS","TCELL.IS","THYAO.IS",
    "TKFEN.IS","TOASO.IS","TSKB.IS","TTKOM.IS","TUPRS.IS",
    "TTRAK.IS","VAKBN.IS","VESBE.IS","VESTL.IS","YKBNK.IS",
    "ZOREN.IS","HALKB.IS","KONTR.IS","ULKER.IS","CIMSA.IS"
]

# ================================
# Helper Functions
# ================================
def fmt(x, n=2):
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return "-"
        return f"{x:.{n}f}"
    except:
        return "-"

def short(t):
    return t.replace(".IS", "")

def normalize_yf(df):
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str)
    need = ["Open","High","Low","Close","Volume"]
    for c in need:
        if c not in df.columns:
            df[c] = np.nan

    df = df[need]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Close"])
    return df

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def build_features(df):
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI14"] = rsi(df["Close"])
    df["ATR14"] = atr(df)

    high120 = df["High"].rolling(120).max()
    df["DROP120"] = (high120 - df["Close"]) / high120 * 100

    low252 = df["Low"].rolling(252).min()
    high252 = df["High"].rolling(252).max()
    rng = high252 - low252
    df["RANGE1Y"] = np.where(rng > 0, (df["Close"] - low252) / rng * 100, np.nan)

    return df

def classify(df):
    last = df.iloc[-1]
    close = last["Close"]
    rsi14 = last["RSI14"]
    ma20 = last["MA20"]
    drop = last["DROP120"]
    rp = last["RANGE1Y"]
    atr14 = last["ATR14"]

    buy = (
        (rp <= 45 if not math.isnan(rp) else False)
        and (30 <= rsi14 <= 60 if not math.isnan(rsi14) else False)
        and (close >= ma20 * 0.98 if not math.isnan(ma20) else False)
    )

    stop = close - 1.5 * atr14
    tp1 = close + 2 * atr14
    tp2 = close + 3.5 * atr14

    reasons = [
        f"RSI: {fmt(rsi14,0)}",
        f"1Y Konum: {fmt(rp,0)}/100",
        f"Zirveden düşüş: %{fmt(drop,0)}"
    ]

    plan = []
    if buy:
        plan.append("✅ **Ben olsam ALIM listesine koyardım.**")
    else:
        plan.append("🟡 **Ben olsam izlerdim, acele etmezdim.**")

    plan.append(f"• Stop: **{fmt(stop,2)}**")
    plan.append(f"• Hedef 1: **{fmt(tp1,2)}**")
    plan.append(f"• Hedef 2: **{fmt(tp2,2)}**")

    return buy, reasons, plan, stop, tp1, tp2

# ================================
# Data Fetch (Cache)
# ================================
@st.cache_data(ttl=3600)
def fetch(ticker):
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    return normalize_yf(df)

# ================================
# Sidebar
# ================================
with st.sidebar:
    st.header("Ayarlar")
    only_buy = st.checkbox("Sadece 'Ben olsam alırdım'", value=True)
    max_cards = st.slider("Maks kart", 10, 50, 25)
    slow = st.checkbox("Yavaş tarama", value=True)

# ================================
# Scan
# ================================
results, errors = [], []

prog = st.progress(0)
status = st.empty()

total = len(BIST50)

for i, t in enumerate(BIST50, start=1):
    try:
        df = fetch(t)
        df = build_features(df)
        df = df.dropna()

        buy, reasons, plan, stop, tp1, tp2 = classify(df)

        results.append({
            "ticker": t,
            "name": short(t),
            "buy": buy,
            "close": df.iloc[-1]["Close"],
            "reasons": reasons,
            "plan": plan,
            "stop": stop,
            "tp1": tp1,
            "tp2": tp2
        })
    except Exception as e:
        errors.append((t, str(e)))

    prog.progress(i / total)
    status.info(f"Taranıyor: {i}/{total}")

    if slow:
        time.sleep(0.2)

prog.empty()
status.empty()

df_res = pd.DataFrame(results)
if only_buy:
    df_res = df_res[df_res["buy"] == True]

st.subheader("🧾 Kart Kart Liste — BIST 50")

shown = 0
for _, r in df_res.iterrows():
    if shown >= max_cards:
        break
    shown += 1

    st.markdown(f"### {r['name']}  \n`{r['ticker']}`")
    st.metric("Kapanış", fmt(r["close"],2))

    st.markdown("**🧠 Sistem Durumu**")
    for x in r["reasons"]:
        st.write(f"• {x}")

    st.markdown("**🧭 Ben olsam**")
    for p in r["plan"]:
        st.write(p)

    st.divider()

if errors:
    with st.expander("⚠️ Hatalı hisseler"):
        st.write(pd.DataFrame(errors, columns=["Ticker","Hata"]))
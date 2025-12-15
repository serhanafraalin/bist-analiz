import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="BIST 50 Tarayıcı", layout="wide")

st.title("📋 BIST 50 Tarayıcı — Kart Kart Liste")
st.caption("Bu sistem yatırım tavsiyesi değildir. 'Ben olsam' bakış açısıyla bilgi sunar.")

# BIST 50 (manuel, stabil)
BIST50 = [
    "ASELS.IS","THYAO.IS","KCHOL.IS","SISE.IS","BIMAS.IS","AKBNK.IS",
    "EREGL.IS","TUPRS.IS","FROTO.IS","ISCTR.IS","SAHOL.IS","PETKM.IS"
]

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def analyze(symbol):
    df = yf.download(symbol, period="1y", interval="1d", auto_adjust=True, progress=False)
    if df.empty or len(df) < 120:
        return None

    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["VOL20"] = df["Volume"].rolling(20).mean()

    last = df.iloc[-1]
    high_120 = df["Close"].rolling(120).max().iloc[-1]
    drop_pct = (high_120 - last["Close"]) / high_120 * 100

    ben_olsam_alirdim = (
        30 <= last["RSI"] <= 55 and
        last["Close"] >= last["MA20"] * 0.98 and
        drop_pct >= 10
    )

    sat_fiyat = max(last["MA50"], df["Close"].rolling(60).max().iloc[-1])
    stop_fiyat = last["MA20"]

    return {
        "symbol": symbol,
        "price": round(last["Close"],2),
        "rsi": round(last["RSI"],1),
        "drop": round(drop_pct,1),
        "vol_ratio": round(last["Volume"]/last["VOL20"],2) if last["VOL20"]>0 else None,
        "alirdim": ben_olsam_alirdim,
        "sat": round(sat_fiyat,2),
        "stop": round(stop_fiyat,2),
        "df": df
    }

results = []
for s in BIST50:
    res = analyze(s)
    if res:
        results.append(res)

for r in results:
    st.markdown("---")
    col1, col2 = st.columns([2,1])

    with col1:
        st.subheader(r["symbol"])
        st.write(f"**Kapanış:** {r['price']}")
        st.write(f"**RSI:** {r['rsi']} | **Zirveden düşüş:** %{r['drop']}")
        st.write(f"**Hacim / 20g:** {r['vol_ratio']}x")

        if r["alirdim"]:
            st.success("🟢 Ben olsam **ALIRDIM**")
        else:
            st.warning("🟡 Ben olsam **BEKLERDİM**")

        st.write(
            f"""
            **Ben olsam planım:**
            - Alım referansı: {r['price']}
            - Satardım: **{r['sat']}**
            - Koruma (stop): **{r['stop']}**
            """
        )

    with col2:
        fig, ax = plt.subplots(figsize=(4,2))
        ax.plot(r["df"]["Close"], label="Fiyat")
        ax.plot(r["df"]["MA20"], label="MA20")
        ax.plot(r["df"]["MA50"], label="MA50")
        ax.legend(fontsize=6)
        st.pyplot(fig)
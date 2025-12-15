import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="BIST Analiz", layout="wide")

st.title("📊 BIST Bilgi Amaçlı Analiz Sistemi")
st.caption("Bu sistem al/sat önerisi vermez. Sadece teknik durumu yorumlar.")

hisse = st.text_input("Hisse kodu (Örn: THYAO.IS)", "THYAO.IS")

if hisse:
    df = yf.download(hisse, period="6mo", interval="1d", group_by="column")

    if df.empty:
        st.error("Veri çekilemedi.")
        st.stop()

    # MultiIndex varsa düzelt
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()

    # İndikatörler
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    st.subheader("📈 Fiyat & Ortalamalar")
    st.line_chart(df[["Close", "EMA20", "EMA50"]])

    st.subheader("📉 RSI")
    st.line_chart(df["RSI"])

    # ANALİZ YORUMLARI
    st.subheader("🧠 Sistem Yorumu (Bilgi Amaçlı)")

    yorumlar = []

    if df["RSI"].iloc[-1] < 30:
        yorumlar.append("• RSI 30 altı → aşırı satım, tepki ihtimali artar.")
    elif df["RSI"].iloc[-1] > 70:
        yorumlar.append("• RSI 70 üstü → aşırı alım, yorulma riski.")
    else:
        yorumlar.append("• RSI dengeli bölgede.")

    if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]:
        yorumlar.append("• Kısa vadeli trend yukarı (EMA20 > EMA50).")
    else:
        yorumlar.append("• Kısa vadeli trend zayıf / aşağı.")

    if df["Close"].iloc[-1] > df["EMA20"].iloc[-1]:
        yorumlar.append("• Fiyat kısa vadeli ortalamanın üzerinde.")
    else:
        yorumlar.append("• Fiyat kısa vadeli ortalamanın altında.")

    for y in yorumlar:
        st.write(y)
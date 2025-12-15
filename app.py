import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="BIST Analiz", layout="wide")

st.title("📊 BIST Bilgi Amaçlı Analiz Sistemi")

st.markdown("""
Bu sistem **al/sat önerisi vermez**.  
Sadece düşüş, tepki ve teknik bölgeleri **bilgi amaçlı** gösterir.
""")

hisse = st.text_input("Hisse kodu gir (Örn: THYAO.IS)", "THYAO.IS")

if hisse:
    data = yf.download(hisse, period="6mo", interval="1d")

    if len(data) > 0:
        data["RSI"] = 100 - (100 / (1 + (data["Close"].diff().clip(lower=0).rolling(14).mean() /
                                         data["Close"].diff().clip(upper=0).abs().rolling(14).mean())))

        data["EMA20"] = data["Close"].ewm(span=20).mean()
        data["EMA50"] = data["Close"].ewm(span=50).mean()

        st.subheader("📈 Fiyat Grafiği")
        st.line_chart(data[["Close", "EMA20", "EMA50"]])

        st.subheader("📉 RSI")
        st.line_chart(data["RSI"])

        son_rsi = data["RSI"].iloc[-1]

        st.subheader("📌 Teknik Durum")
        if son_rsi < 30:
            st.info("RSI düşük → düşüş sonrası tepki ihtimali (bilgi amaçlı)")
        elif son_rsi > 70:
            st.warning("RSI yüksek → aşırı alım bölgesi (bilgi amaçlı)")
        else:
            st.success("RSI dengeli bölgede")

    else:
        st.error("Veri bulunamadı")
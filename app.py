import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="BIST Analiz", layout="wide")

st.title("📊 BIST Bilgi Amaçlı Analiz Sistemi")

st.markdown("""
Bu sistem **al/sat önerisi vermez**.  
Sadece teknik durumları **bilgi amaçlı** listeler.
""")

hisse = st.text_input("Hisse kodu gir (Örn: THYAO.IS)", "THYAO.IS")

if hisse:
    data = yf.download(hisse, period="6mo", interval="1d", auto_adjust=True)

    if data.empty:
        st.error("Veri bulunamadı.")
    else:
        data = data.reset_index()

        data["EMA20"] = data["Close"].ewm(span=20).mean()
        data["EMA50"] = data["Close"].ewm(span=50).mean()

        delta = data["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean()
        data["RSI"] = 100 - (100 / (1 + rs))

        st.subheader("📈 Fiyat & Ortalamalar")
        st.line_chart(data.set_index("Date")[["Close", "EMA20", "EMA50"]])

        st.subheader("📉 RSI")
        st.line_chart(data.set_index("Date")["RSI"])

        son_fiyat = data["Close"].iloc[-1]
        rsi = data["RSI"].iloc[-1]
        ema20 = data["EMA20"].iloc[-1]
        ema50 = data["EMA50"].iloc[-1]

        st.subheader("🧠 TEKNİK DURUM ÖZETİ")

        analiz = []

        if rsi < 30:
            analiz.append("🔵 RSI 30 altı → Sert düşüş sonrası **tepki ihtimali**")
        elif rsi > 70:
            analiz.append("🔴 RSI 70 üstü → **Aşırı alım**, kâr satışı gelebilir")
        else:
            analiz.append("🟡 RSI dengeli bölgede")

        if son_fiyat > ema20 > ema50:
            analiz.append("🟢 Fiyat ortalamaların üstünde → **Pozitif trend**")
        elif son_fiyat < ema20 < ema50:
            analiz.append("🔴 Fiyat ortalamaların altında → **Negatif trend**")
        else:
            analiz.append("🟠 Fiyat sıkışma bölgesinde")

        destek = data["Close"].rolling(20).min().iloc[-1]
        direnç = data["Close"].rolling(20).max().iloc[-1]

        analiz.append(f"📉 Yakın destek: {destek:.2f}")
        analiz.append(f"📈 Yakın direnç: {direnç:.2f}")

        for madde in analiz:
            st.write(madde)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="BIST Bilgi Sistemi", layout="wide")

st.title("📊 BIST Bilgi Amaçlı Analiz Sistemi")

st.markdown("""
Bu sistem **kesinlikle al / sat demez**.  
Hiç bilmeyen biri için **ne oluyor, neden oluyor** onu anlatır.
""")

hisse = st.text_input("Hisse kodu (Örn: THYAO.IS)", "THYAO.IS")

if hisse:
    data = yf.download(hisse, period="6mo", interval="1d", auto_adjust=True)

    if not data.empty:

        # MultiIndex temizleme
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Ortalamalar
        data["EMA20"] = data["Close"].ewm(span=20).mean()
        data["EMA50"] = data["Close"].ewm(span=50).mean()

        # RSI
        delta = data["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean()
        data["RSI"] = 100 - (100 / (1 + rs))

        # Ortalama hacim
        avg_volume = data["Volume"].rolling(20).mean()

        st.subheader("📈 Fiyat ve Ortalamalar")
        st.line_chart(data[["Close", "EMA20", "EMA50"]])

        st.subheader("📊 Hacim")
        st.line_chart(data["Volume"])

        st.subheader("🧠 Sistem Yorumu (Bilgi Amaçlı)")

        yorumlar = []

        son_fiyat = data["Close"].iloc[-1]
        rsi = data["RSI"].iloc[-1]

        if rsi < 30:
            yorumlar.append(
                "RSI 30’un altında. Bu genelde sert düşüş sonrası görülür. "
                "Bazı yatırımcılar bu bölgelerde tepki gelip gelmediğini izler."
            )
        elif rsi > 70:
            yorumlar.append(
                "RSI 70’in üstünde. Fiyat kısa sürede çok yükselmiş olabilir. "
                "Bu bölgelerde genelde temkinli olunur."
            )
        else:
            yorumlar.append(
                "RSI dengeli bölgede. Ne aşırı alım ne aşırı satım var."
            )

        if son_fiyat < data["EMA20"].iloc[-1]:
            yorumlar.append(
                "Fiyat kısa vadeli ortalamanın altında. "
                "Bu genelde kısa vadede zayıflık anlamına gelir."
            )
        else:
            yorumlar.append(
                "Fiyat kısa vadeli ortalamanın üzerinde. "
                "Bu kısa vadede olumlu kabul edilir."
            )

        if data["Volume"].iloc[-1] > avg_volume.iloc[-1]:
            yorumlar.append(
                "Bugünkü hacim son 20 gün ortalamasının üzerinde. "
                "Bu, hareketin daha dikkat çekici olduğu anlamına gelir."
            )
        else:
            yorumlar.append(
                "Hacim ortalama seviyede. Büyük oyuncular henüz belirgin değil."
            )

        for y in yorumlar:
            st.write("•", y)

    else:
        st.error("Veri alınamadı.")
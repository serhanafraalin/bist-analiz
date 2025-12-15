import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="BIST Bilgi Sistemi", layout="wide")

st.title("📊 BIST Bilgi Amaçlı Analiz Sistemi")

st.markdown("""
⚠️ **Bu sistem al/sat önerisi vermez.**  
Ama **ne olduğunu, neden olduğunu ve şu an piyasanın ne anlattığını** sade Türkçe ile açıklar.
""")

hisse = st.text_input("Hisse kodu gir (Örn: THYAO.IS)", "THYAO.IS")

if hisse:
    data = yf.download(hisse, period="6mo", interval="1d")

    if data.empty:
        st.error("Veri bulunamadı.")
        st.stop()

    # Göstergeler
    data["EMA20"] = data["Close"].ewm(span=20).mean()
    data["EMA50"] = data["Close"].ewm(span=50).mean()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # Grafik
    st.subheader("📈 Fiyat ve Ortalamalar")
    st.line_chart(data[["Close", "EMA20", "EMA50"]])

    st.subheader("📊 Hacim")
    st.bar_chart(data["Volume"])

    st.subheader("📉 RSI (Momentum)")
    st.line_chart(data["RSI"])

    # Son değerler
    son_fiyat = data["Close"].iloc[-1]
    rsi = data["RSI"].iloc[-1]
    ema20 = data["EMA20"].iloc[-1]
    ema50 = data["EMA50"].iloc[-1]
    hacim = data["Volume"].iloc[-1]
    ort_hacim = data["Volume"].rolling(20).mean().iloc[-1]

    st.subheader("🧠 Sistem Yorumu (Bilgi Amaçlı)")

    # RSI Yorumu
    if rsi < 30:
        st.write("• RSI **çok düşük**. Bu, hissede son dönemde **sert satışlar olduğunu** gösterir.")
        st.write("• Bu tür bölgelerde bazen **kısa vadeli toparlanmalar** görülebilir.")
    elif rsi > 70:
        st.write("• RSI **çok yüksek**. Hisse son günlerde **fazla hızlı yükselmiş** olabilir.")
        st.write("• Böyle dönemlerde **dinlenme / geri çekilme** yaşanabilir.")
    else:
        st.write("• RSI **dengeli**. Ne aşırı alım ne aşırı satım var.")

    # Trend Yorumu
    if son_fiyat < ema20 < ema50:
        st.write("• Fiyat, hem kısa hem orta vadeli ortalamanın **altında**.")
        st.write("• Bu durum **zayıf trend / baskılı piyasa** anlamına gelir.")
    elif son_fiyat > ema20 > ema50:
        st.write("• Fiyat, ortalamaların **üzerinde**.")
        st.write("• Bu genelde **güçlü trend** olarak yorumlanır.")
    else:
        st.write("• Fiyat ve ortalamalar **kararsız bölgede**.")
        st.write("• Piyasa yön arıyor olabilir.")

    # Hacim Yorumu
    if hacim > ort_hacim:
        st.write("• Bugünkü hacim **ortalamanın üzerinde**.")
        st.write("• Bu, yapılan hareketin **daha dikkat çekici** olduğunu gösterir.")
    else:
        st.write("• Hacim **düşük**.")
        st.write("• Hareketler şu an **çok güçlü katılımla yapılmıyor**.")

    st.markdown("""
---
📌 **Özet:**  
Bu ekran sana **“şu an piyasada ne oluyor?”** sorusunun cevabını verir.  
Kararı **sen verirsin**.
""")
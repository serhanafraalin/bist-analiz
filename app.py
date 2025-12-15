import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="BIST Analiz Sistemi", layout="wide")

st.title("📊 BIST Bilgi Amaçlı Analiz Sistemi")
st.caption("Bu sistem yatırım tavsiyesi vermez. Ben olsam ne yapardım mantığıyla bilgi sunar.")

# -------------------------
# HİSSE GİRİŞİ
# -------------------------
hisse = st.text_input("Hisse Kodu (Örn: THYAO.IS)", "THYAO.IS")

if hisse:
    data = yf.download(hisse, period="6mo", interval="1d", auto_adjust=True)

    if data.empty:
        st.error("Veri çekilemedi.")
        st.stop()

    # -------------------------
    # İNDİKATÖRLER
    # -------------------------
    data["EMA20"] = data["Close"].ewm(span=20).mean()
    data["EMA50"] = data["Close"].ewm(span=50).mean()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    data["RSI"] = 100 - (100 / (1 + rs))

    data["Hacim_Ort"] = data["Volume"].rolling(20).mean()

    son = data.iloc[-1]

    # -------------------------
    # GRAFİK
    # -------------------------
    st.subheader("📈 Fiyat ve Ortalamalar")
    st.line_chart(data[["Close", "EMA20", "EMA50"]])

    st.subheader("📉 RSI")
    st.line_chart(data["RSI"])

    st.subheader("📊 Hacim")
    st.bar_chart(data[["Volume", "Hacim_Ort"]])

    # -------------------------
    # YORUM MOTORU
    # -------------------------
    st.subheader("🧠 Sistem Yorumu (Bilgi Amaçlı)")

    yorumlar = []

    # RSI
    if son["RSI"] < 30:
        yorumlar.append(
            "RSI 30 altı. Piyasa aşırı satımda. "
            "Ben olsam satış yapmaz, tepki gelir mi diye izlerdim."
        )
    elif son["RSI"] > 70:
        yorumlar.append(
            "RSI 70 üstü. Aşırı alım bölgesi. "
            "Ben olsam yeni alım yapmaz, kârı korumayı düşünürdüm."
        )
    else:
        yorumlar.append(
            "RSI dengeli. Ne aşırı alım ne aşırı satım var."
        )

    # Trend
    if son["Close"] > son["EMA20"] > son["EMA50"]:
        yorumlar.append(
            "Fiyat kısa ve orta vadeli ortalamaların üzerinde. "
            "Ben olsam trend yukarı diye düşünür, geri çekilmeleri kollardım."
        )
    elif son["Close"] < son["EMA20"] < son["EMA50"]:
        yorumlar.append(
            "Fiyat ortalamaların altında. Trend zayıf. "
            "Ben olsam acele etmezdim."
        )
    else:
        yorumlar.append(
            "Fiyat ortalamalar arasında. Kararsız bir yapı var."
        )

    # Hacim
    if son["Volume"] > son["Hacim_Ort"]:
        yorumlar.append(
            "Bugünkü hacim son 20 gün ortalamasının üzerinde. "
            "Hareket ciddiye alınmalı."
        )
    else:
        yorumlar.append(
            "Hacim düşük. Hareket çok ikna edici değil."
        )

    # -------------------------
    # BEN OLSAM NE YAPARDIM?
    # -------------------------
    st.subheader("🧩 Ben Olsam Ne Yapardım?")

    if son["RSI"] < 35 and son["Close"] > son["EMA20"]:
        st.success(
            "Ben olsam: Küçük miktarla ALIM düşünürdüm.\n\n"
            "Sebep: Aşırı satımdan çıkış + fiyat kısa vadede toparlanıyor."
        )
    elif son["RSI"] > 65:
        st.warning(
            "Ben olsam: KÂR ALMAYI düşünürdüm.\n\n"
            "Sebep: Aşırı alım bölgesi."
        )
    else:
        st.info(
            "Ben olsam: BEKLERDİM.\n\n"
            "Sebep: Net bir avantaj yok."
        )

    # -------------------------
    # SATIŞ / HEDEF MANTIĞI
    # -------------------------
    st.subheader("🎯 Hedef & Risk Mantığı")

    destek = data["Low"].rolling(20).min().iloc[-1]
    direnç = data["High"].rolling(20).max().iloc[-1]

    st.write(f"""
    • Yakın Destek: **{destek:.2f}**
    • Yakın Direnç: **{direnç:.2f}**

    Ben olsam:
    - Alım yaptıysam **destek altını zarar kes** kabul ederdim.
    - Dirence yaklaştıkça **satışı düşünürdüm**.
    """)

    # -------------------------
    # YORUMLARI YAZDIR
    # -------------------------
    st.subheader("📌 Detaylı Açıklamalar")
    for y in yorumlar:
        st.write("•", y)
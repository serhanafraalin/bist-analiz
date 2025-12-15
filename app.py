import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="BIST Analiz", layout="wide")

st.title("📊 BIST Bilgi Amaçlı Analiz Sistemi")
st.markdown("""
Bu sistem **yatırım tavsiyesi vermez**.  
📌 *“Ben olsam neye bakardım?”* mantığıyla bilgi sunar.
""")

# ---------------- INPUT ----------------
hisse = st.text_input("Hisse Kodu (Örn: THYAO.IS)", "THYAO.IS")

if hisse:
    data = yf.download(hisse, period="6mo", interval="1d", auto_adjust=True)

    if data.empty:
        st.error("Veri çekilemedi.")
        st.stop()

    # ---- SÜTUNLARI DÜZLEŞTİR ----
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # ---- İNDİKATÖRLER ----
    data["EMA20"] = data["Close"].ewm(span=20).mean()
    data["EMA50"] = data["Close"].ewm(span=50).mean()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    data["RSI"] = 100 - (100 / (1 + rs))

    data["Hacim_Ort"] = data["Volume"].rolling(20).mean()

    son = data.iloc[-1]

    # ---------------- GRAFİKLER ----------------
    st.subheader("📈 Fiyat")
    st.line_chart(data["Close"])

    st.subheader("📊 Hareketli Ortalamalar")
    st.line_chart(data[["EMA20", "EMA50"]])

    st.subheader("📉 RSI")
    st.line_chart(data["RSI"])

    st.subheader("📦 Hacim")
    st.line_chart(data["Volume"])

    # ---------------- YORUM ----------------
    st.subheader("🧠 Sistem Yorumu (Bilgi Amaçlı)")

    yorumlar = []

    # RSI Yorumu
    if son["RSI"] < 30:
        yorumlar.append("RSI 30’un altında. Hisse sert düşmüş. **Ben olsam satış yapmaz, tepki arardım.**")
    elif son["RSI"] > 70:
        yorumlar.append("RSI 70’in üzerinde. Hisse çok yükselmiş. **Ben olsam yeni alımda temkinli olurdum.**")
    else:
        yorumlar.append("RSI dengeli. Ne aşırı alım ne aşırı satım var.")

    # Trend Yorumu
    if son["Close"] > son["EMA20"]:
        yorumlar.append("Fiyat kısa vadeli ortalamanın üzerinde. **Kısa vadede olumlu.**")
    else:
        yorumlar.append("Fiyat kısa vadeli ortalamanın altında. **Kısa vadede zayıf.**")

    # Hacim Yorumu
    if son["Volume"] > son["Hacim_Ort"]:
        yorumlar.append("Hacim ortalamanın üzerinde. **Hareket dikkat çekici.**")
    else:
        yorumlar.append("Hacim düşük. **Güçlü bir ilgi yok.**")

    # BEN OLSAM NE YAPARDIM
    st.markdown("### 🤔 Ben Olsam Ne Yapardım?")
    if son["RSI"] < 35 and son["Close"] < son["EMA20"]:
        st.info("Ben olsam **izlerdim**, acele almazdım. Tepki gelirse değerlendirirdim.")
    elif son["RSI"] > 65:
        st.warning("Ben olsam **kârı korurdum**, yeni alım yapmazdım.")
    else:
        st.success("Ben olsam **beklerdim**. Net sinyal yok.")

    for y in yorumlar:
        st.write("•", y)
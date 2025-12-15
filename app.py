st.subheader("🧠 Sistem Yorumu (Bilgi Amaçlı)")

son_fiyat = data["Close"].iloc[-1]
ema20 = data["EMA20"].iloc[-1]
ema50 = data["EMA50"].iloc[-1]
son_rsi = data["RSI"].iloc[-1]

hacim_ort = data["Volume"].rolling(20).mean().iloc[-1]
son_hacim = data["Volume"].iloc[-1]

st.markdown(f"""
📉 **Fiyat Durumu**  
Hissenin güncel fiyatı **{son_fiyat:.2f}**. Son günlerde fiyat baskı altında.

📊 **RSI Yorumu**  
RSI değeri **{son_rsi:.1f}** seviyesinde.
- Bu seviye hissenin **çok satıldığını** gösterir.
- Genelde bu bölgelerde **kısa vadeli tepki hareketleri** görülebilir.
- Ancak bu, düşüşün bittiği anlamına gelmez.

📉 **Trend Durumu**  
- Kısa vadeli ortalama (EMA20): **{ema20:.2f}**
- Orta vadeli ortalama (EMA50): **{ema50:.2f}**

Fiyat bu ortalamaların **altında**, yani genel yön hâlâ zayıf.

📦 **Hacim Yorumu**  
Son işlem hacmi: **{son_hacim:,.0f}**  
20 günlük ortalama hacim: **{hacim_ort:,.0f}**

""")

if son_hacim > hacim_ort:
    st.warning("Son hareketlerde hacim yüksek → piyasada güçlü bir karar süreci var.")
else:
    st.info("Hacim düşük → hareketler kararsız olabilir, net yön henüz oluşmamış.")

st.markdown("""
🧠 **Genel Okuma**  
Bu tarz bölgeler genelde **izleme bölgeleri** olarak değerlendirilir.  
Net yön için:
- Fiyatın düşüşü durdurması
- Hacmin artması
- Ortalama seviyelerin üzerine çıkması  
beklenir.
""")
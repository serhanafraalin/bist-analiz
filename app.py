import math
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="BIST 50 Tarayıcı", layout="wide")

st.title("📋 BIST 50 Tarayıcı — Kart Kart Liste")
st.caption("Bu sistem yatırım tavsiyesi vermez. 'Ben olsam' bölümü, **örnek plan şablonu** olarak bilgi verir. Karar %100 sende.")

# -----------------------------
# BIST 50 (yaklaşık liste - zamanla değişebilir)
# Not: İstersen bunu sen güncel tutarsın. Şimdilik BIST 50 ağırlıklı bir liste verdim.
# -----------------------------
BIST50 = [
    "AKBNK.IS","ALARK.IS","ARCLK.IS","ASELS.IS","ASTOR.IS","BIMAS.IS","BRISA.IS","CCOLA.IS","DOAS.IS","EKGYO.IS",
    "ENJSA.IS","ENKAI.IS","EREGL.IS","FROTO.IS","GARAN.IS","GUBRF.IS","HEKTS.IS","ISCTR.IS","KCHOL.IS","KOZAA.IS",
    "KOZAL.IS","KRDMD.IS","MAVI.IS","ODAS.IS","OTKAR.IS","PETKM.IS","PGSUS.IS","SAHOL.IS","SASA.IS","SISE.IS",
    "SKBNK.IS","SMRTG.IS","SOKM.IS","TCELL.IS","THYAO.IS","TKFEN.IS","TOASO.IS","TSKB.IS","TTKOM.IS","TUPRS.IS",
    "TTRAK.IS","VAKBN.IS","VESBE.IS","VESTL.IS","YKBNK.IS","ZOREN.IS","HALKB.IS","KONTR.IS","ULKER.IS","CIMSA.IS"
]

# -----------------------------
# Helpers
# -----------------------------
def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        return float(x)
    except Exception:
        return np.nan

def fmt(x, n=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{x:.{n}f}"

def normalize_yf(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance bazen MultiIndex kolon döndürüyor.
    Biz tek sütunlu standart OHLCV yapısına çeviriyoruz.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Index adı yoksa (Date gibi)
    df.index = pd.to_datetime(df.index)

    # MultiIndex kolon ise (('Close','ASELS.IS') gibi)
    if isinstance(df.columns, pd.MultiIndex):
        # Bazı durumlarda ilk seviye OHLCV olur
        # OHLCV'i seçip tek seviye haline getiriyoruz
        lvl0 = df.columns.get_level_values(0).astype(str)
        if set(["Open","High","Low","Close","Adj Close","Volume"]).intersection(set(lvl0)):
            # Önce OHLCV tarafını al
            cols = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in lvl0.values]
            out = {}
            for c in cols:
                # Bu kolonun altındaki ilk ticker'ı al (tek hisse indirirken zaten 1 tane olur)
                sub = df.loc[:, df.columns.get_level_values(0) == c]
                # sub dataframe -> ilk sütunu seç
                out[c] = sub.iloc[:, 0]
            df = pd.DataFrame(out, index=df.index)
        else:
            # Fallback: ilk sütun setini al
            df.columns = [str(c[0]) for c in df.columns]

    # Bazı durumlarda 'Adj Close' gelmez; Close yeterli
    if "Close" not in df.columns:
        # yfinance bazen 'Adj Close' ağırlıklı döner
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            return pd.DataFrame()

    # Volume yoksa üretme
    if "Volume" not in df.columns:
        df["Volume"] = np.nan

    # Gereksizleri at
    keep = ["Open","High","Low","Close","Volume"]
    for k in keep:
        if k not in df.columns:
            df[k] = np.nan
    df = df[keep].dropna(subset=["Close"])

    return df

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=period).mean()
    ma_down = down.rolling(period, min_periods=period).mean()
    rs = ma_up / ma_down
    out = 100 - (100 / (1 + rs))
    return out

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20, min_periods=20).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=50).mean()
    df["RSI14"] = rsi(df["Close"], 14)
    df["ATR14"] = atr(df, 14)
    df["VOL_MA20"] = df["Volume"].rolling(20, min_periods=20).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA20"]

    # 120 gün zirveden düşüş
    roll_high_120 = df["High"].rolling(120, min_periods=60).max()
    df["DROP_120"] = (roll_high_120 - df["Close"]) / roll_high_120 * 100

    # 1 yıllık aralık konumu (0-100)
    roll_low_252 = df["Low"].rolling(252, min_periods=120).min()
    roll_high_252 = df["High"].rolling(252, min_periods=120).max()
    rng = (roll_high_252 - roll_low_252)
    df["RANGE_POS_1Y"] = np.where(rng > 0, (df["Close"] - roll_low_252) / rng * 100, np.nan)

    # 60 günlük tepe (direnç adayı)
    df["HIGH_60"] = df["High"].rolling(60, min_periods=30).max()

    # Son 20 gün dip (stop referansı)
    df["LOW_20"] = df["Low"].rolling(20, min_periods=10).min()

    return df

def classify_and_plan(df: pd.DataFrame) -> dict:
    """
    'Ben olsam alırdım' ve 'şu fiyata gelince satardım' benzeri
    net seviyeler üretir.
    """
    last = df.iloc[-1]

    close = safe_float(last["Close"])
    ma20 = safe_float(last["MA20"])
    ma50 = safe_float(last["MA50"])
    rsi14 = safe_float(last["RSI14"])
    drop120 = safe_float(last["DROP_120"])
    vr = safe_float(last["VOL_RATIO"])
    rp = safe_float(last["RANGE_POS_1Y"])
    atr14 = safe_float(last["ATR14"])
    hi60 = safe_float(last["HIGH_60"])
    low20 = safe_float(last["LOW_20"])

    reasons = []

    # "Usta mantığı": ucuz bölge + risk kontrol + hacim
    # Buy-candidate (ben olsam alırdım) kriterleri:
    cond_range = (not math.isnan(rp)) and rp <= 35
    cond_rsi = (not math.isnan(rsi14)) and (30 <= rsi14 <= 55)
    cond_trend = (not math.isnan(ma20)) and close >= ma20 * 0.98  # MA20'ye yakın/üstü
    cond_drop = (not math.isnan(drop120)) and drop120 >= 15
    cond_vol = (not math.isnan(vr)) and vr >= 0.9  # hacim en az ortalamaya yakın

    ben_olsam_alirdim = bool((cond_range and cond_rsi and cond_trend) or (cond_drop and cond_rsi and cond_vol and cond_trend))

    # Sebep yazıları (sen anlaman için)
    if not math.isnan(rp):
        if rp <= 35:
            reasons.append(f"Fiyat 1 yıllık aralığın alt bölgesinde ({fmt(rp,0)}/100) → daha 'ucuz' bölge.")
        elif rp >= 75:
            reasons.append(f"Fiyat 1 yıllık aralığın üst bölgesinde ({fmt(rp,0)}/100) → daha 'pahalı' bölge.")
        else:
            reasons.append(f"Fiyat 1 yıllık aralığın orta bölgesinde ({fmt(rp,0)}/100).")

    if not math.isnan(rsi14):
        if rsi14 < 30:
            reasons.append(f"RSI {fmt(rsi14,0)} → aşırı satıma yakın (tepki ihtimali artar ama risk de var).")
        elif rsi14 > 70:
            reasons.append(f"RSI {fmt(rsi14,0)} → aşırı alıma yakın (kısa vadede ısınmış olabilir).")
        else:
            reasons.append(f"RSI {fmt(rsi14,0)} → dengeli bölgede.")

    if not math.isnan(ma20):
        if close >= ma20:
            reasons.append("Fiyat MA20 üstünde/çevresinde → kısa vadede daha güçlü duruyor.")
        else:
            reasons.append("Fiyat MA20 altında → kısa vadede zayıf görünüm.")

    if not math.isnan(ma50):
        if close >= ma50:
            reasons.append("Fiyat MA50 üstünde → orta vade trend daha güçlü.")
        else:
            reasons.append("Fiyat MA50 altında → orta vadede temkin gerekir.")

    if not math.isnan(vr):
        if vr >= 1.2:
            reasons.append(f"Hacim güçlü (20g ortalamaya göre {fmt(vr,2)}x) → hareket daha 'anlamlı' olabilir.")
        elif vr <= 0.8:
            reasons.append(f"Hacim zayıf ({fmt(vr,2)}x) → hareketler daha çabuk sönümlenebilir.")
        else:
            reasons.append(f"Hacim normal civarı ({fmt(vr,2)}x).")

    # "Ben olsam" plan seviyeleri:
    # Stop: son 20 gün dibi veya ATR tabanlı
    # Take Profit 1: MA50 (fiyatın üstünde/altında durumuna göre)
    # Take Profit 2: 60 gün tepe (direnç adayı)
    # Eğer MA50 mantıksızsa (NaN), TP1 = close * 1.08 (örnek)
    if math.isnan(atr14) or atr14 <= 0:
        atr14 = max(0.0, close * 0.02)

    stop_ref = low20 if (not math.isnan(low20) and low20 > 0) else (close - 1.5 * atr14)
    stop_level = min(stop_ref, close - 1.2 * atr14)  # biraz daha güvenli

    if (not math.isnan(ma50)) and ma50 > 0:
        tp1 = ma50 if ma50 > close else close * 1.06
    else:
        tp1 = close * 1.06

    tp2 = hi60 if (not math.isnan(hi60) and hi60 > 0) else close * 1.12

    # Düzen: tp2 tp1'den küçükse düzelt
    if tp2 <= tp1:
        tp2 = max(tp1 * 1.03, close * 1.10)

    # Plan cümlesi: "ben olsam alırdım" TRUE ise daha net konuş
    plan_lines = []
    if ben_olsam_alirdim:
        plan_lines.append("✅ **Ben olsam bu bölgeyi 'ALIM İÇİN İZLEME/DEĞERLENDİRME' listeme koyardım.**")
    else:
        plan_lines.append("🟡 **Ben olsam şu an 'izlerdim' (acele etmezdim).**")

    plan_lines.append(f"• Referans (son kapanış): **{fmt(close,2)}**")
    plan_lines.append(f"• Ben olsam **zarar-kes/temkin** seviyesini yaklaşık: **{fmt(stop_level,2)}** civarı takip ederdim.")
    plan_lines.append(f"• Ben olsam **kâr alma 1** (ilk hedef): **{fmt(tp1,2)}** civarı (MA50/ilk eşik mantığı).")
    plan_lines.append(f"• Ben olsam **kâr alma 2** (ikinci hedef): **{fmt(tp2,2)}** civarı (son 60g tepe/direnç adayı).")
    plan_lines.append("• Kural şablonu: Fiyat hedefe yaklaşırken **hacim düşüyorsa** temkin, **hacim artıyorsa** hareket daha sağlıklı olabilir.")

    return {
        "close": close,
        "drop120": drop120,
        "rsi14": rsi14,
        "vol_ratio": vr,
        "range_pos": rp,
        "ben_olsam_alirdim": ben_olsam_alirdim,
        "reasons": reasons,
        "plan": plan_lines,
        "stop_level": stop_level,
        "tp1": tp1,
        "tp2": tp2
    }

@st.cache_data(ttl=60 * 60)  # 1 saat cache (çok çağrı olmasın)
def fetch_one(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, period="1y", interval="1d", auto_adjust=False, progress=False)
    df = normalize_yf(raw)
    return df

def ticker_short(t: str) -> str:
    return t.replace(".IS", "")

# -----------------------------
# UI Controls
# -----------------------------
with st.sidebar:
    st.header("Ayarlar")
    only_buy = st.toggle("Sadece 'Ben olsam alırdım' listesi", value=True)
    max_cards = st.slider("Gösterilecek maksimum kart", 10, 50, 25)
    st.divider()
    st.caption("Not: Çok hızlı yenilersen veri sağlayıcı limitleyebilir. Cache var.")

st.markdown("Aşağıdaki liste, her açılışta en güncel kapanış verileriyle yeniden hesaplanır (cache: 1 saat).")

# -----------------------------
# Scan
# -----------------------------
results = []
errors = []

progress = st.progress(0, text="BIST 50 taranıyor...")
total = len(BIST50)

for i, ticker in enumerate(BIST50, start=1):
    try:
        df = fetch_one(ticker)
        if df is None or df.empty or len(df) < 80:
            raise ValueError("Yetersiz veri (tarihçe kısa veya boş).")

        df = build_features(df)
        df = df.dropna(subset=["MA20","MA50","RSI14","ATR14"], how="any")
        if df.empty:
            raise ValueError("Göstergeler hesaplanamadı (NaN).")

        info = classify_and_plan(df)

        results.append({
            "ticker": ticker,
            "name": ticker_short(ticker),
            "close": info["close"],
            "drop120": info["drop120"],
            "rsi14": info["rsi14"],
            "vol_ratio": info["vol_ratio"],
            "range_pos": info["range_pos"],
            "buy": info["ben_olsam_alirdim"],
            "reasons": info["reasons"],
            "plan": info["plan"],
            "stop": info["stop_level"],
            "tp1": info["tp1"],
            "tp2": info["tp2"],
        })

    except Exception as e:
        errors.append((ticker, str(e)))

    progress.progress(i / total, text=f"Taranıyor: {i}/{total}")

progress.empty()

if not results:
    st.error("Hiç veri çekilemedi. (Bağlantı, veri sağlayıcı limiti veya ticker listesi sorunu olabilir.)")
    if errors:
        st.write("Hatalar:")
        st.write(errors[:10])
    st.stop()

res_df = pd.DataFrame(results)

# Sırala: buy=True olanlar üstte, sonra range_pos düşük, sonra drop120 yüksek
res_df["buy_rank"] = res_df["buy"].astype(int)
res_df = res_df.sort_values(by=["buy_rank","range_pos","drop120"], ascending=[False, True, False]).reset_index(drop=True)

if only_buy:
    res_df = res_df[res_df["buy"] == True].reset_index(drop=True)

st.subheader("🧾 Kart Kart Liste")

shown = 0
for _, row in res_df.iterrows():
    if shown >= max_cards:
        break

    shown += 1
    t = row["ticker"]
    name = row["name"]

    col1, col2, col3, col4, col5 = st.columns([1.2,1,1,1,1])
    with col1:
        st.markdown(f"### {name}")
        st.caption(t)
    with col2:
        st.metric("Kapanış", fmt(row["close"],2))
    with col3:
        st.metric("Zirveden düşüş (120g)", f"%{fmt(row['drop120'],0)}")
    with col4:
        st.metric("RSI(14)", fmt(row["rsi14"],0))
    with col5:
        st.metric("Hacim / 20g Ort", f"{fmt(row['vol_ratio'],2)}x")

    # Durum + Plan
    st.markdown("**🧠 Sistem Durumu (Bilgi Amaçlı)**")
    for r in row["reasons"]:
        st.write(f"• {r}")

    st.markdown("**🧭 Ben olsam (örnek plan şablonu)**")
    for p in row["plan"]:
        st.write(p)

    st.divider()

if errors:
    with st.expander("⚠️ Bazı hisselerde veri/hesaplama hatası oldu (gizli)"):
        st.write(pd.DataFrame(errors, columns=["Ticker", "Hata"]).head(20))
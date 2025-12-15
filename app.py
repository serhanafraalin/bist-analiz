import math
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ---------------------------------
# App Config
# ---------------------------------
st.set_page_config(page_title="BIST 50 Tarayıcı", layout="wide")
st.title("📋 BIST 50 Tarayıcı — Kart Kart Liste")
st.caption(
    "Bu sistem yatırım tavsiyesi değildir. "
    "'Ben olsam' bölümü, **örnek plan şablonu** olarak bilgi verir; karar %100 sende."
)

# ---------------------------------
# BIST50 (yaklaşık liste)
# ---------------------------------
BIST50 = [
    "AKBNK.IS","ALARK.IS","ARCLK.IS","ASELS.IS","ASTOR.IS","BIMAS.IS","BRISA.IS","CCOLA.IS","DOAS.IS","EKGYO.IS",
    "ENJSA.IS","ENKAI.IS","EREGL.IS","FROTO.IS","GARAN.IS","GUBRF.IS","HEKTS.IS","ISCTR.IS","KCHOL.IS","KOZAA.IS",
    "KOZAL.IS","KRDMD.IS","MAVI.IS","ODAS.IS","OTKAR.IS","PETKM.IS","PGSUS.IS","SAHOL.IS","SASA.IS","SISE.IS",
    "SKBNK.IS","SMRTG.IS","SOKM.IS","TCELL.IS","THYAO.IS","TKFEN.IS","TOASO.IS","TSKB.IS","TTKOM.IS","TUPRS.IS",
    "TTRAK.IS","VAKBN.IS","VESBE.IS","VESTL.IS","YKBNK.IS","ZOREN.IS","HALKB.IS","KONTR.IS","ULKER.IS","CIMSA.IS"
]

# ---------------------------------
# Helpers
# ---------------------------------
def fmt(x, n=2):
    try:
        if x is None:
            return "-"
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return "-"
        return f"{x:.{n}f}"
    except Exception:
        return "-"

def ticker_short(t: str) -> str:
    return t.replace(".IS", "")

def _to_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Close/High/Low gibi kolonlar bazen DataFrame gibi gelebiliyor; kesin Series'e çevir."""
    x = df[col]
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x

def normalize_yf(raw: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance bazen MultiIndex / garip kolon döndürür.
    Burada OHLCV'yi tek-seviye, tekil sütunlara indiriyoruz.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df.index = pd.to_datetime(df.index)

    # MultiIndex kolon
    if isinstance(df.columns, pd.MultiIndex):
        # group_by='column' gibi geldiğinde level0 OHLCV olur
        lvl0 = df.columns.get_level_values(0).astype(str)
        if set(["Open","High","Low","Close","Adj Close","Volume"]).intersection(set(lvl0)):
            out = {}
            for c in ["Open","High","Low","Close","Adj Close","Volume"]:
                if c in lvl0.values:
                    sub = df.loc[:, df.columns.get_level_values(0) == c]
                    out[c] = sub.iloc[:, 0]
            df = pd.DataFrame(out, index=df.index)
        else:
            # başka MultiIndex ise basitleştir
            df.columns = [str(c[0]) for c in df.columns]

    # Kolon isimlerini temizle
    df.columns = [str(c).strip() for c in df.columns]

    # Close yoksa Adj Close'tan üret
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            return pd.DataFrame()

    if "Volume" not in df.columns:
        df["Volume"] = np.nan
    for k in ["Open","High","Low"]:
        if k not in df.columns:
            df[k] = np.nan

    df = df[["Open","High","Low","Close","Volume"]].copy()

    # numeric yap
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Close olmayanları at
    df = df.dropna(subset=["Close"])
    df = df[~df.index.duplicated(keep="last")]
    return df

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=period).mean()
    ma_down = down.rolling(period, min_periods=period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = _to_series(df, "High")
    low = _to_series(df, "Low")
    close = _to_series(df, "Close")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # kesin Series
    df["Close"] = _to_series(df, "Close")
    df["High"]  = _to_series(df, "High")
    df["Low"]   = _to_series(df, "Low")
    df["Volume"]= _to_series(df, "Volume")

    df["MA20"] = df["Close"].rolling(20, min_periods=20).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=50).mean()
    df["RSI14"] = rsi(df["Close"], 14)
    df["ATR14"] = atr(df, 14)

    df["VOL_MA20"] = df["Volume"].rolling(20, min_periods=20).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA20"]

    roll_high_120 = df["High"].rolling(120, min_periods=60).max()
    df["DROP_120"] = (roll_high_120 - df["Close"]) / roll_high_120 * 100

    roll_low_252 = df["Low"].rolling(252, min_periods=120).min()
    roll_high_252 = df["High"].rolling(252, min_periods=120).max()
    rng = (roll_high_252 - roll_low_252)
    df["RANGE_POS_1Y"] = np.where(rng > 0, (df["Close"] - roll_low_252) / rng * 100, np.nan)

    df["HIGH_60"] = df["High"].rolling(60, min_periods=30).max()
    df["LOW_20"] = df["Low"].rolling(20, min_periods=10).min()
    return df

def classify_and_plan(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]

    close = float(last["Close"])
    ma20 = float(last["MA20"]) if pd.notna(last["MA20"]) else np.nan
    ma50 = float(last["MA50"]) if pd.notna(last["MA50"]) else np.nan
    rsi14 = float(last["RSI14"]) if pd.notna(last["RSI14"]) else np.nan
    drop120 = float(last["DROP_120"]) if pd.notna(last["DROP_120"]) else np.nan
    vr = float(last["VOL_RATIO"]) if pd.notna(last["VOL_RATIO"]) else np.nan
    rp = float(last["RANGE_POS_1Y"]) if pd.notna(last["RANGE_POS_1Y"]) else np.nan
    atr14 = float(last["ATR14"]) if pd.notna(last["ATR14"]) else np.nan
    hi60 = float(last["HIGH_60"]) if pd.notna(last["HIGH_60"]) else np.nan
    low20 = float(last["LOW_20"]) if pd.notna(last["LOW_20"]) else np.nan

    reasons = []

    # “Ben olsam alırdım” — basit, okunabilir filtre:
    # - 1Y aralık alt/orta-alt
    # - RSI 30-60
    # - MA20’ye yakın/üst
    # - hacim en az ortalamaya yakın
    cond_range = (not math.isnan(rp)) and rp <= 45
    cond_rsi = (not math.isnan(rsi14)) and (30 <= rsi14 <= 60)
    cond_trend = (not math.isnan(ma20)) and close >= ma20 * 0.98
    cond_vol = (not math.isnan(vr)) and vr >= 0.9
    cond_drop = (not math.isnan(drop120)) and drop120 >= 10

    ben_olsam_alirdim = bool((cond_range and cond_rsi and cond_trend and cond_vol) or (cond_drop and cond_rsi and cond_trend))

    # Durum cümleleri
    if not math.isnan(rp):
        if rp <= 35:
            reasons.append(f"Fiyat 1 yıllık aralığın alt bölgesinde ({fmt(rp,0)}/100).")
        elif rp >= 75:
            reasons.append(f"Fiyat 1 yıllık aralığın üst bölgesinde ({fmt(rp,0)}/100).")
        else:
            reasons.append(f"Fiyat 1 yıllık aralığın orta bölgesinde ({fmt(rp,0)}/100).")

    if not math.isnan(rsi14):
        if rsi14 < 30:
            reasons.append(f"RSI {fmt(rsi14,0)} → aşırı satıma yakın.")
        elif rsi14 > 70:
            reasons.append(f"RSI {fmt(rsi14,0)} → aşırı alıma yakın.")
        else:
            reasons.append(f"RSI {fmt(rsi14,0)} → dengeli bölgede.")

    if not math.isnan(ma20):
        reasons.append("Fiyat MA20 üstünde/çevresinde." if close >= ma20 else "Fiyat MA20 altında.")

    if not math.isnan(ma50):
        reasons.append("Fiyat MA50 üstünde (orta vade daha güçlü)." if close >= ma50 else "Fiyat MA50 altında (orta vade temkin).")

    if not math.isnan(vr):
        if vr >= 1.2:
            reasons.append(f"Hacim güçlü ({fmt(vr,2)}x).")
        elif vr <= 0.8:
            reasons.append(f"Hacim zayıf ({fmt(vr,2)}x).")
        else:
            reasons.append(f"Hacim normal ({fmt(vr,2)}x).")

    # Plan: “Ben olsam alırdım + şu fiyata gelince satardım” (net seviyeler)
    if math.isnan(atr14) or atr14 <= 0:
        atr14 = max(0.0, close * 0.02)

    # stop: low20 veya close - 1.5 ATR (hangisi daha güvenliyse)
    stop_candidate = low20 if (not math.isnan(low20) and low20 > 0) else (close - 1.5 * atr14)
    stop_level = min(stop_candidate, close - 1.2 * atr14)

    # hedef1: close + 2 ATR veya MA50 (hangisi daha yakın/gerçekçi ise)
    target_atr1 = close + 2.0 * atr14
    if (not math.isnan(ma50)) and ma50 > close:
        tp1 = min(target_atr1, ma50)
    else:
        tp1 = target_atr1

    # hedef2: 60g tepe, yoksa close + 3.5 ATR
    tp2 = hi60 if (not math.isnan(hi60) and hi60 > 0) else (close + 3.5 * atr14)
    if tp2 <= tp1:
        tp2 = tp1 + 1.0 * atr14

    plan = []
    if ben_olsam_alirdim:
        plan.append("✅ **Ben olsam bu hisseyi ALIM İÇİN listeye koyardım.**")
        plan.append(f"• **Ben olsam satış planı**: fiyat **{fmt(tp1,2)}** civarına gelince *kârın bir kısmını* alırdım; "
                    f"**{fmt(tp2,2)}** civarı *ikinci kâr bölgesi* diye izlerdim.")
        plan.append(f"• **Ben olsam temkin/stop**: **{fmt(stop_level,2)}** altı olursa planı bozar, temkinli olur/çıkarım derdim.")
    else:
        plan.append("🟡 **Ben olsam şu an acele etmezdim; izleme listesine alırdım.**")
        plan.append(f"• Eğer işlem düşünseydim: önce MA20/MA50 davranışını ve hacmi izlerdim.")
        plan.append(f"• Plan şablonu yine aynı mantık: temkin **{fmt(stop_level,2)}**, hedefler **{fmt(tp1,2)}** / **{fmt(tp2,2)}** (bilgi amaçlı).")

    plan.append("• Not: Hedefe yaklaşırken **hacim artıyorsa** hareket daha sağlıklı; **hacim düşüyorsa** daha temkinli olurum.")

    return {
        "close": close, "drop120": drop120, "rsi14": rsi14, "vol_ratio": vr, "range_pos": rp,
        "buy": ben_olsam_alirdim, "reasons": reasons, "plan": plan,
        "stop": stop_level, "tp1": tp1, "tp2": tp2
    }

@st.cache_data(ttl=60*60)
def fetch_one(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, period="1y", interval="1d", auto_adjust=False, progress=False)
    return normalize_yf(raw)

# ---------------------------------
# UI Controls
# ---------------------------------
with st.sidebar:
    st.header("Ayarlar")
    only_buy = st.toggle("Sadece 'Ben olsam alırdım' listesi", value=True)
    max_cards = st.slider("Gösterilecek maksimum kart", 10, 50, 25)
    slow = st.toggle("Yavaş tarama (limit riskini azaltır)", value=True)
    st.divider()
    st.caption("Not: Çok hızlı yenilersen veri sağlayıcı limitleyebilir. Cache 1 saat.")

st.markdown("Aşağıdaki liste her açılışta en güncel kapanış verileriyle hesaplanır (cache: **1 saat**).")

# ---------------------------------
# Scan
# ---------------------------------
results = []
errors = []

prog = st.progress(0, text="BIST 50 taranıyor...")
total = len(BIST50)

for i, ticker in enumerate(BIST50, start=1):
    try:
        df = fetch_one(ticker)
        if df.empty or len(df) < 120:
            raise ValueError("Yetersiz veri (tarihçe kısa/boş).")

        df = build_features(df)
        # göstergeler oluşsun diye MA50/RSI/ATR olmayanları at
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
            "buy": info["buy"],
            "reasons": info["reasons"],
            "plan": info["plan"],
            "stop": info["stop"],
            "tp1": info["tp1"],
            "tp2": info["tp2"],
        })
    except Exception as e:
        errors.append((ticker, str(e)))

    prog.progress(i/total, text=f"Taranıyor: {i}/{total}")

    if slow:
        time.sleep(0.25)

prog.empty()

if not results:
    st.error("Hiç veri çekilemedi. (Bağlantı / veri sağlayıcı limiti / ticker listesi sorunu olabilir.)")
    if errors:
        st.write(pd.DataFrame(errors, columns=["Ticker","Hata"]).head(20))
    st.stop()

res_df = pd.DataFrame(results)
res_df["buy_rank"] = res_df["buy"].astype(int)

# buy olanlar üstte; sonra range_pos düşük; sonra drop yüksek
res_df = res_df.sort_values(by=["buy_rank","range_pos","drop120"], ascending=[False, True, False]).reset_index(drop=True)

if only_buy:
    res_df = res_df[res_df["buy"] == True].reset_index(drop=True)

st.subheader("🧾 Kart Kart Liste")

shown = 0
for _, row in res_df.iterrows():
    if shown >= max_cards:
        break
    shown += 1

    st.markdown(f"### {row['name']}  \n`{row['ticker']}`")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Kapanış", fmt(row["close"], 2))
    c2.metric("Zirveden düşüş (120g)", f"%{fmt(row['drop120'],0)}")
    c3.metric("RSI(14)", fmt(row["rsi14"], 0))
    c4.metric("Hacim / 20g", f"{fmt(row['vol_ratio'],2)}x")
    c5.metric("1Y Konum", f"{fmt(row['range_pos'],0)}/100")

    st.markdown("**🧠 Sistem Durumu (Bilgi Amaçlı)**")
    for r in row["reasons"]:
        st.write(f"• {r}")

    st.markdown("**🧭 Ben olsam (net fiyatlı plan şablonu)**")
    for p in row["plan"]:
        st.write(p)

    st.caption(f"Plan seviyeleri (bilgi): stop≈ {fmt(row['stop'],2)} | hedef1≈ {fmt(row['tp1'],2)} | hedef2≈ {fmt(row['tp2'],2)}")
    st.divider()

if errors:
    with st.expander("⚠️ Bazı hisselerde veri/hesaplama hatası oldu (göster)"):
        st.write(pd.DataFrame(errors, columns=["Ticker","Hata"]).head(50))
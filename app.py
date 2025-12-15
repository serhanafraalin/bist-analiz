import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="BIST 50 Tarayıcı", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance bazen MultiIndex döndürür. Close/High/Low/Volume tek level'e indir."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        # En yaygın: level0 = OHLCV alanları
        # Örn: ('Close','THYAO.IS') gibi
        cols = []
        for c in df.columns:
            if isinstance(c, tuple) and len(c) >= 1:
                cols.append(str(c[0]))
            else:
                cols.append(str(c))
        df = df.copy()
        df.columns = cols
    return df

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.rolling(period, min_periods=period).mean()
    roll_down = down.rolling(period, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def fmt_price(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    if x >= 100:
        return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{x:.3f}".replace(".", ",")

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ---------------------------
# BIST 50 list (semboller)
# Not: Yahoo Finance BIST için .IS gerekir.
# Burayı ileride güncelleyebiliriz; ama çalışması için sabit liste yeter.
# ---------------------------
BIST50 = [
    "AKBNK.IS","ALARK.IS","ARCLK.IS","ASELS.IS","ASTOR.IS","BIMAS.IS","BRSAN.IS","DOAS.IS",
    "EKGYO.IS","ENKAI.IS","EREGL.IS","FROTO.IS","GARAN.IS","GUBRF.IS","HEKTS.IS","ISCTR.IS",
    "KCHOL.IS","KOZAL.IS","KRDMD.IS","MAVI.IS","ODAS.IS","OTKAR.IS","PETKM.IS","PGSUS.IS",
    "SAHOL.IS","SASA.IS","SISE.IS","SKBNK.IS","TAVHL.IS","TCELL.IS","THYAO.IS","TKFEN.IS",
    "TOASO.IS","TTKOM.IS","TUPRS.IS","ULKER.IS","VAKBN.IS","VESBE.IS","YKBNK.IS",
    # (BIST50 değişebilir; listeyi sonra güncelleriz. Sistemi bozmaz.)
]

# ---------------------------
# Data download
# ---------------------------
@st.cache_data(ttl=60 * 60)  # 1 saat cache (sen açtıkça güncellenir, gün içinde de yenilenir)
def load_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    df = _flatten_yf_columns(df)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # yfinance bazen "Date" bazen "Datetime" döndürebilir
    if "Date" not in df.columns:
        if "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)
    # Temizlik
    keep = [c for c in ["Date","Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    df = df[keep].dropna(subset=["Close"])
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    # MA
    df["MA20"] = df["Close"].rolling(20, min_periods=20).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=50).mean()

    # RSI, ATR
    df["RSI14"] = rsi(df["Close"], 14)
    df["ATR14"] = atr(df, 14)

    # Volume ratio
    if "Volume" in df.columns:
        df["VOL_MA20"] = df["Volume"].rolling(20, min_periods=20).mean()
        df["VOL_RATIO"] = df["Volume"] / df["VOL_MA20"]
    else:
        df["VOL_RATIO"] = np.nan

    # 120g drop from high
    roll_high_120 = df["High"].rolling(120, min_periods=60).max()
    df["DROP_120"] = (roll_high_120 - df["Close"]) / roll_high_120 * 100

    # 1y range position (0=alt,100=üst)
    roll_low_252 = df["Low"].rolling(252, min_periods=120).min()
    roll_high_252 = df["High"].rolling(252, min_periods=120).max()
    span = (roll_high_252 - roll_low_252).replace(0, np.nan)
    df["RANGE_POS_1Y"] = ((df["Close"] - roll_low_252) / span) * 100

    return df

def score_candidate(last: pd.Series) -> float:
    """'Ben olsam alırdım' listesi için puanlama (tavsiye değil, filtre)."""
    rsi14 = safe_float(last.get("RSI14"))
    pos = safe_float(last.get("RANGE_POS_1Y"))
    drop = safe_float(last.get("DROP_120"))
    volr = safe_float(last.get("VOL_RATIO"))

    # Daha düşük range_pos + orta/düşük RSI + anlamlı düşüş + hacim toparlıyorsa iyi puan
    s = 0.0
    if not math.isnan(pos):
        s += (50 - pos) * 1.2  # alt bölgeye yakınsa artar
    if not math.isnan(rsi14):
        # 30-55 aralığı "soğuk/normal" bölge, aşırı sıcak değil
        s += (55 - rsi14) * 0.8
    if not math.isnan(drop):
        s += clamp(drop, 0, 50) * 0.7
    if not math.isnan(volr):
        # Hacim 1x üstüyse hafif bonus
        s += clamp((volr - 1.0) * 10, -10, 10)
    return s

def plan_levels(df: pd.DataFrame) -> dict:
    """Ben olsam plan şablonu: stop + hedef1 + hedef2 (fiyat olarak)."""
    last = df.iloc[-1]
    close = safe_float(last["Close"])
    ma50 = safe_float(last.get("MA50"))
    atr14 = safe_float(last.get("ATR14"))
    # son 20g swing low (koruma/stop mantığı)
    swing_low = safe_float(df["Low"].tail(20).min())
    # son 60g tepe (olası direnç/hedef)
    high60 = safe_float(df["High"].tail(60).max())

    # Stop: swing_low - 0.3*ATR (ATR yoksa %4 aşağı)
    if not math.isnan(atr14):
        stop = swing_low - 0.30 * atr14
    else:
        stop = close * 0.96

    # Hedef1: MA50 (yukarıdaysa close'a yakın bir "kâr alma bölgesi" olmaz; o zaman MA20 kullanırız)
    ma20 = safe_float(last.get("MA20"))
    if not math.isnan(ma50) and ma50 > 0:
        target1 = ma50
    elif not math.isnan(ma20) and ma20 > 0:
        target1 = ma20
    else:
        target1 = close * 1.05

    # Hedef2: 60g tepe (direnç adayı)
    target2 = high60 if (not math.isnan(high60) and high60 > 0) else close * 1.10

    # "Ben olsam alırdım" dediğimizde bile AL demiyoruz; sadece olası senaryo:
    return {
        "ref": close,
        "stop": stop,
        "target1": target1,
        "target2": target2,
    }

def write_card(ticker: str, name: str, df: pd.DataFrame):
    last = df.iloc[-1]
    close = safe_float(last["Close"])
    drop120 = safe_float(last.get("DROP_120"))
    rsi14 = safe_float(last.get("RSI14"))
    volr = safe_float(last.get("VOL_RATIO"))
    pos = safe_float(last.get("RANGE_POS_1Y"))
    ma20 = safe_float(last.get("MA20"))
    ma50 = safe_float(last.get("MA50"))

    # Durum cümleleri (daha anlaşılır)
    dur = []
    if not math.isnan(pos):
        dur.append(f"• 1 yıllık aralık konumu: **{int(round(pos))}/100** (0=ucuz bölge, 100=pahalı bölge)")
    if not math.isnan(rsi14):
        if rsi14 < 30:
            dur.append(f"• RSI **{int(round(rsi14))}** → piyasa kısa vadede **aşırı satım** tarafına yakın.")
        elif rsi14 > 70:
            dur.append(f"• RSI **{int(round(rsi14))}** → piyasa kısa vadede **ısınmış** olabilir.")
        else:
            dur.append(f"• RSI **{int(round(rsi14))}** → **normal/dengeli** bölge.")
    if not math.isnan(ma50):
        dur.append("• Fiyat **MA50** " + ("üzerinde (trend güçlü)" if close > ma50 else "altında (trend zayıf)"))
    if not math.isnan(ma20):
        dur.append("• Fiyat **MA20** " + ("üzerinde (kısa vade pozitif)" if close > ma20 else "altında (kısa vade negatif)"))
    if not math.isnan(volr):
        if volr >= 1.2:
            dur.append(f"• Hacim güçlü: **{volr:.2f}x** (20g ortalamanın üstü)")
        elif volr <= 0.8:
            dur.append(f"• Hacim zayıf: **{volr:.2f}x** (20g ortalamanın altı)")
        else:
            dur.append(f"• Hacim normal: **{volr:.2f}x** (20g ortalamaya yakın)")
    if not math.isnan(drop120):
        dur.append(f"• 120g zirveye uzaklık: **-%{int(round(drop120))}**")

    levels = plan_levels(df)

    # Burada senin istediğin gibi: "Ben olsam ALIRDIM ve ŞU FİYATA GELİNCE SATARDIM" diyoruz.
    # Bu tavsiye değildir; örnek plan şablonu.
    plan = []
    plan.append(f"• **Ben olsam (senaryo):** Bu karttaki şartlar içime sinerse **kademeli alım** düşünürdüm.")
    plan.append(f"• **Korunma/Stop seviyesi:** **{fmt_price(levels['stop'])}** altına kalıcı düşerse planı bozarım.")
    plan.append(f"• **Satış (Kâr alma) 1:** **{fmt_price(levels['target1'])}** civarında **bir kısmını** masaya bırakırım.")
    plan.append(f"• **Satış (Kâr alma) 2:** **{fmt_price(levels['target2'])}** civarında **kalanı** azaltmayı düşünürüm.")
    plan.append("• Not: Hacim artarak yükseliyorsa hareket daha sağlıklı; hacim düşerek yükseliyorsa daha temkinli olurum.")

    # Kart UI
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2,1,1,1])
        c1.markdown(f"### {name} ({ticker.replace('.IS','')})")
        c2.metric("Kapanış", fmt_price(close))
        c3.metric("RSI(14)", "-" if math.isnan(rsi14) else str(int(round(rsi14))))
        c4.metric("Hacim/20g", "-" if math.isnan(volr) else f"{volr:.2f}x")

        st.markdown("**🧠 Sistem Durumu (Bilgi Amaçlı)**")
        st.markdown("\n".join(dur) if dur else "Veri yetersiz.")

        st.markdown("**🧭 Ben olsam (örnek plan)**")
        st.markdown("\n".join(plan))

        # Mini grafik (hatasız)
        chart_df = df.tail(120).copy()
        chart_df = chart_df.set_index("Date")
        cols = [c for c in ["Close","MA20","MA50"] if c in chart_df.columns]
        if len(cols) >= 1:
            st.line_chart(chart_df[cols], height=220)

# ---------------------------
# App
# ---------------------------
st.title("📋 BIST 50 Tarayıcı — Kart Kart Liste")
st.caption("Liste her açtığında en güncel **kapanış** verisiyle yeniden hesaplanır. (Bildirim yok.)")

colA, colB, colC = st.columns([2,1,1])
max_cards = colB.slider("Gösterilecek kart", 5, 30, 15)
mode = colC.selectbox("Liste Modu", ["Ben olsam alırdım (filtreli)", "Tümü (BIST50)"])

# Basit isim eşlemesi (istersen sonra genişletiriz)
NAME_MAP = {
    "ASELS.IS": "ASELS", "THYAO.IS": "THYAO", "GARAN.IS": "GARAN", "SISE.IS": "SISE",
    "EREGL.IS": "EREGL", "KCHOL.IS": "KCHOL", "SAHOL.IS": "SAHOL", "TUPRS.IS": "TUPRS",
    "BIMAS.IS": "BIMAS", "FROTO.IS": "FROTO", "TTKOM.IS": "TTKOM", "TCELL.IS": "TCELL",
}

results = []

with st.spinner("BIST50 taranıyor..."):
    for t in BIST50:
        try:
            raw = load_ohlcv(t, "1y")
            if raw.empty or len(raw) < 80:
                continue
            df = build_features(raw)
            if df.empty:
                continue
            last = df.iloc[-1]
            # gerekli kolonlar var mı
            if "Close" not in df.columns:
                continue

            sc = score_candidate(last)
            results.append({
                "ticker": t,
                "name": NAME_MAP.get(t, t.replace(".IS","")),
                "score": sc,
                "df": df,
            })
        except Exception:
            # bir hisse patlarsa tüm app patlamasın
            continue

if not results:
    st.error("Veri çekilemedi. (Yahoo kaynaklı geçici olabilir.) Biraz sonra tekrar dene.")
    st.stop()

# Sıralama
results = sorted(results, key=lambda x: x["score"], reverse=True)

if mode.startswith("Ben olsam"):
    # filtre (aşırı pahalı/ısınmış olanları aşağı iter)
    filtered = []
    for r in results:
        df = r["df"]
        last = df.iloc[-1]
        pos = safe_float(last.get("RANGE_POS_1Y"))
        rsi14 = safe_float(last.get("RSI14"))
        # Filtre: pahalı üst bölge + aşırı RSI çok sıcak ise liste dışı
        if (not math.isnan(pos) and pos > 75) and (not math.isnan(rsi14) and rsi14 > 70):
            continue
        filtered.append(r)
    results = filtered

st.markdown("---")

# Kartları bas
shown = 0
for r in results:
    if shown >= max_cards:
        break
    write_card(r["ticker"], r["name"], r["df"])
    shown += 1

st.markdown("---")
st.caption("⚠️ Bu ekran **yatırım tavsiyesi değildir**. 'Ben olsam' kısmı **örnek plan şablonu**dur; karar ve risk tamamen kullanıcıya aittir.")

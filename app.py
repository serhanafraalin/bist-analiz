import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ----------------------------
# Helpers
# ----------------------------
def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance bazen MultiIndex kolon döndürür. Bunu tek seviyeye indirir."""
    if isinstance(df.columns, pd.MultiIndex):
        # ('Close', 'THYAO.IS') gibi gelir -> 'Close'
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def pct_drop_from_high(close: pd.Series, lookback: int = 120) -> float:
    window = close.tail(lookback)
    if window.empty:
        return np.nan
    h = float(window.max())
    c = float(window.iloc[-1])
    if h == 0:
        return np.nan
    return (h - c) / h * 100.0

def price_position_in_range(close: pd.Series, lookback: int = 252) -> float:
    """0-100: son 1 yıl aralığında fiyat nerde? 0=dip, 100=tepe"""
    window = close.tail(lookback)
    if len(window) < 20:
        return np.nan
    lo = float(window.min())
    hi = float(window.max())
    c = float(window.iloc[-1])
    if hi == lo:
        return 50.0
    return (c - lo) / (hi - lo) * 100.0

def safe_download(ticker: str, period="1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = _flatten_yf_columns(df)
    df = df.reset_index()  # Date kolonu gelsin
    # Bazı ortamlarda 'Date' yerine 'Datetime' gelebiliyor:
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20, min_periods=20).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=50).mean()
    df["RSI14"] = rsi(df["Close"], 14)
    df["VOL_MA20"] = df["Volume"].rolling(20, min_periods=20).mean()
    df["ATR14"] = atr(df, 14)
    return df

def score_and_reasons(df: pd.DataFrame) -> tuple[float, list[str]]:
    """
    'Ben olsam alırdım' listesi için puan:
    - Düşüş (%25+)
    - RSI düşük/denge
    - Fiyat MA50 altında ama stabil (tepki adayı)
    - Hacim artışı (bugün > 20g ort)
    Not: Bu bir öneri değil, 'izleme için sıralama'.
    """
    reasons = []
    c = float(df["Close"].iloc[-1])
    ma20 = df["MA20"].iloc[-1]
    ma50 = df["MA50"].iloc[-1]
    r = df["RSI14"].iloc[-1]
    v = df["Volume"].iloc[-1]
    vma = df["VOL_MA20"].iloc[-1]

    drop = pct_drop_from_high(df["Close"], 120)
    pos = price_position_in_range(df["Close"], 252)

    score = 0.0

    # Drop
    if not np.isnan(drop):
        if drop >= 35:
            score += 3; reasons.append(f"Zirveden düşüş yüksek (≈ %{drop:.0f}).")
        elif drop >= 25:
            score += 2; reasons.append(f"Zirveden anlamlı düşüş (≈ %{drop:.0f}).")
        else:
            score += 0.5

    # RSI
    if not np.isnan(r):
        if r < 32:
            score += 2.5; reasons.append(f"RSI düşük ({r:.0f}) → satış baskısı sonrası tepki potansiyeli.")
        elif 32 <= r <= 55:
            score += 1.5; reasons.append(f"RSI dengeli ({r:.0f}) → panik yok, takip edilebilir.")
        elif r > 70:
            score -= 1.5; reasons.append(f"RSI yüksek ({r:.0f}) → kısa vadede ısınmış olabilir.")

    # MA ilişkisi
    if not np.isnan(ma50):
        if c < ma50:
            score += 1.5; reasons.append("Fiyat MA50 altında → 'toparlanma/tepki' izleme bölgesi.")
        else:
            score += 0.8; reasons.append("Fiyat MA50 üzerinde → trend daha güçlü görünüyor.")

    if not np.isnan(ma20):
        if c >= ma20:
            score += 1.0; reasons.append("Fiyat MA20 üzerinde → kısa vadede pozitif.")
        else:
            score += 0.5; reasons.append("Fiyat MA20 altında → kısa vadede zayıf.")

    # Hacim
    if not np.isnan(vma) and vma > 0:
        if v > vma * 1.2:
            score += 1.8; reasons.append("Hacim 20g ortalamanın belirgin üstünde → hareket daha anlamlı.")
        elif v > vma:
            score += 1.0; reasons.append("Hacim 20g ortalamanın üstünde.")
        else:
            reasons.append("Hacim zayıf → hareketler daha kolay sönümlenebilir.")

    # Aralık pozisyonu
    if not np.isnan(pos):
        if pos < 25:
            score += 1.0; reasons.append("1 yıllık aralığın alt bölgesi → ucuz bölgeye yakın.")
        elif pos > 80:
            score -= 1.0; reasons.append("1 yıllık aralığın üst bölgesi → daha pahalı bölge.")

    return score, reasons

def plan_levels(df: pd.DataFrame) -> dict:
    """
    'Ben olsam ne yapardım?' için sayısal seviyeler:
    - Referans: güncel kapanış (C)
    - İzlenecek kâr bölgeleri: MA50, son 60 gün tepe
    - Risk bölgesi: C - 1.5*ATR veya son 20 gün dip (hangisi yakınsa)
    """
    c = float(df["Close"].iloc[-1])
    ma50 = float(df["MA50"].iloc[-1]) if not np.isnan(df["MA50"].iloc[-1]) else np.nan
    atr14 = float(df["ATR14"].iloc[-1]) if not np.isnan(df["ATR14"].iloc[-1]) else np.nan

    swing_high = float(df["Close"].tail(60).max()) if len(df) >= 60 else float(df["Close"].max())
    swing_low20 = float(df["Close"].tail(20).min()) if len(df) >= 20 else float(df["Close"].min())

    # "Ben olsam" risk seviyesi: ATR varsa kullan, yoksa 20g dip
    if not np.isnan(atr14) and atr14 > 0:
        risk_stop = c - 1.5 * atr14
    else:
        risk_stop = swing_low20

    # Kâr hedefleri (bilgi amaçlı): MA50 ve son 60g tepe
    targets = []
    if not np.isnan(ma50):
        targets.append(("MA50 (orta vade eşik)", ma50))
    targets.append(("Son 60 gün tepe (direnç adayı)", swing_high))

    return {
        "current": c,
        "stop_ref": risk_stop,
        "targets": targets,
        "swing_low20": swing_low20
    }

# ----------------------------
# BIST 50 List (basit, pratik)
# ----------------------------
BIST50 = [
    ("AKBNK", "AKBNK.IS"),
    ("ARCLK", "ARCLK.IS"),
    ("ASELS", "ASELS.IS"),
    ("BIMAS", "BIMAS.IS"),
    ("DOHOL", "DOHOL.IS"),
    ("EKGYO", "EKGYO.IS"),
    ("ENKAI", "ENKAI.IS"),
    ("EREGL", "EREGL.IS"),
    ("FROTO", "FROTO.IS"),
    ("GARAN", "GARAN.IS"),
    ("GUBRF", "GUBRF.IS"),
    ("HEKTS", "HEKTS.IS"),
    ("ISCTR", "ISCTR.IS"),
    ("KCHOL", "KCHOL.IS"),
    ("KOZAA", "KOZAA.IS"),
    ("KOZAL", "KOZAL.IS"),
    ("KRDMD", "KRDMD.IS"),
    ("MGROS", "MGROS.IS"),
    ("ODAS", "ODAS.IS"),
    ("PETKM", "PETKM.IS"),
    ("PGSUS", "PGSUS.IS"),
    ("SAHOL", "SAHOL.IS"),
    ("SASA", "SASA.IS"),
    ("SISE", "SISE.IS"),
    ("TAVHL", "TAVHL.IS"),
    ("TCELL", "TCELL.IS"),
    ("THYAO", "THYAO.IS"),
    ("TKFEN", "TKFEN.IS"),
    ("TOASO", "TOASO.IS"),
    ("TTKOM", "TTKOM.IS"),
    ("TUPRS", "TUPRS.IS"),
    ("VAKBN", "VAKBN.IS"),
    ("YKBNK", "YKBNK.IS"),
    ("HALKB", "HALKB.IS"),
    ("ALARK", "ALARK.IS"),
    ("SOKM", "SOKM.IS"),
    ("BRSAN", "BRSAN.IS"),
    ("KONTR", "KONTR.IS"),
    ("OYAKC", "OYAKC.IS"),
    ("CIMSA", "CIMSA.IS"),
    ("ENJSA", "ENJSA.IS"),
    ("ASTOR", "ASTOR.IS"),
    ("KARSN", "KARSN.IS"),
    ("KLGYO", "KLGYO.IS"),
    ("ULKER", "ULKER.IS"),
    ("TTKOM", "TTKOM.IS"),
    ("TTRAK", "TTRAK.IS"),
    ("SKBNK", "SKBNK.IS"),
    ("BRYAT", "BRYAT.IS"),
    ("BAGFS", "BAGFS.IS"),
]

# Düzgün görünüm için tekilleştir
seen = set()
BIST50 = [(n, t) for (n, t) in BIST50 if (t not in seen and not seen.add(t))]

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="BIST 50 Bilgi Sistemi", layout="wide")
st.title("📊 BIST 50 — Bilgi Amaçlı Analiz Sistemi")

st.caption(
    "Bu uygulama yatırım tavsiyesi vermez. "
    "“Ben olsam” kısmı **kişisel örnek plan şablonu** gibi düşün: "
    "karar %100 sende."
)

colA, colB, colC = st.columns([1.2, 1, 1])

with colA:
    mode = st.radio("Mod", ["📋 BIST50 Tarayıcı (Ben olsam listesi)", "🔎 Tek Hisse Detay"], index=0)

with colB:
    min_drop = st.slider("Zirveden düşüş filtresi (%, son 120 gün)", 0, 60, 20)

with colC:
    top_n = st.slider("Listede gösterilecek hisse", 5, 25, 12)

st.divider()

def render_card(name: str, ticker: str, df: pd.DataFrame):
    c = float(df["Close"].iloc[-1])
    r = float(df["RSI14"].iloc[-1]) if not np.isnan(df["RSI14"].iloc[-1]) else np.nan
    v = float(df["Volume"].iloc[-1])
    vma = float(df["VOL_MA20"].iloc[-1]) if not np.isnan(df["VOL_MA20"].iloc[-1]) else np.nan
    drop = pct_drop_from_high(df["Close"], 120)
    pos = price_position_in_range(df["Close"], 252)

    score, reasons = score_and_reasons(df)
    levels = plan_levels(df)

    with st.container(border=True):
        st.subheader(f"{name}  ({ticker.replace('.IS','')})")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Kapanış", f"{c:.2f}")
        m2.metric("Zirveden düşüş (120g)", f"%{drop:.0f}" if not np.isnan(drop) else "—")
        m3.metric("RSI(14)", f"{r:.0f}" if not np.isnan(r) else "—")
        if not np.isnan(vma) and vma > 0:
            m4.metric("Hacim / 20g Ort", f"{(v/vma):.2f}x")
        else:
            m4.metric("Hacim / 20g Ort", "—")

        st.markdown("**🧠 Sistem Durumu (Bilgi Amaçlı)**")
        st.write(
            f"- 1 yıllık aralık konumu: **{pos:.0f}/100**" if not np.isnan(pos) else "- 1 yıllık aralık konumu: —"
        )
        for rr in reasons[:6]:
            st.write(f"- {rr}")

        st.markdown("**🧭 Ben olsam (örnek plan şablonu)**")
        # Not: Bu bir örnek; al/sat demiyoruz ama sayısal seviyeler veriyoruz.
        c = levels["current"]
        stop_ref = levels["stop_ref"]
        t1_name, t1_val = levels["targets"][0]
        t2_name, t2_val = levels["targets"][-1]

        st.write(
            f"- **Referans fiyat (bugünkü kapanış):** {c:.2f}\n"
            f"- **Temkin seviyesi (izleme/korunma):** {stop_ref:.2f} civarı (ATR/son dip mantığı)\n"
            f"- **Kâr bölgesi 1:** {t1_name} → {t1_val:.2f}\n"
            f"- **Kâr bölgesi 2:** {t2_name} → {t2_val:.2f}\n"
            f"- **Not:** Hacim zayıflayıp fiyat yükseliyorsa hareket daha çabuk sönümlenebilir; hacim artıyorsa hareket daha anlamlı olabilir."
        )

        with st.expander("📈 Grafik (Kapanış + MA20 + MA50)"):
            chart = df[["Date", "Close", "MA20", "MA50"]].dropna()
            chart = chart.set_index("Date")
            st.line_chart(chart, height=220)

# ----------------------------
# Main
# ----------------------------
if mode.startswith("📋"):
    st.subheader("📋 BIST 50 Tarayıcı — Kart Kart Liste")
    st.caption("Aşağıdaki liste her açtığında en güncel kapanış verisiyle yeniden hesaplanır.")

    rows = []
    for name, ticker in BIST50:
        df = safe_download(ticker, period="1y")
        if df.empty or len(df) < 80:
            continue
        df = build_features(df)
        if df[["MA20", "MA50", "RSI14", "VOL_MA20"]].tail(1).isna().any(axis=1).iloc[-1]:
            continue

        drop = pct_drop_from_high(df["Close"], 120)
        if np.isnan(drop) or drop < float(min_drop):
            continue

        score, reasons = score_and_reasons(df)
        rows.append({
            "name": name,
            "ticker": ticker,
            "score": score,
            "df": df,
            "drop": drop,
        })

    if not rows:
        st.warning("Filtreye uyan hisse bulunamadı. Düşüş filtresini biraz azaltmayı dene.")
    else:
        rows = sorted(rows, key=lambda x: x["score"], reverse=True)[:top_n]

        st.markdown("### ✅ “Ben olsam listesi” (izleme sıralaması)")
        st.caption("Bu bir öneri değil; izleme için sıralamadır. Karar %100 sende.")

        for item in rows:
            render_card(item["name"], item["ticker"], item["df"])

else:
    st.subheader("🔎 Tek Hisse Detay")
    options = [f"{n} ({t.replace('.IS','')})" for (n, t) in BIST50]
    choice = st.selectbox("BIST50 içinden seç", options, index=options.index("THYAO (THYAO)") if "THYAO (THYAO)" in options else 0)
    idx = options.index(choice)
    name, ticker = BIST50[idx]

    df = safe_download(ticker, period="1y")
    if df.empty:
        st.error("Veri çekilemedi. Biraz sonra tekrar dene.")
    else:
        df = build_features(df)
        render_card(name, ticker, df)
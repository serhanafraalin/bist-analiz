import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# -----------------------------
# Ayarlar
# -----------------------------
st.set_page_config(page_title="BIST50 Tarayıcı", layout="wide")

st.title("📋 BIST 50 Tarayıcı — Kart Kart Liste")
st.caption("Bu uygulama yatırım tavsiyesi değildir. 'Ben olsam' kısmı, **kural tabanlı örnek bir işlem planı şablonudur**. Karar tamamen sende.")

# -----------------------------
# BIST 50 (pratik başlangıç listesi)
# Not: Endeks bileşenleri zamanla değişebilir. Bu listeyi istersen sonra güncelleriz.
# -----------------------------
BIST50 = [
    "AEFES.IS","AKBNK.IS","ARCLK.IS","ASELS.IS","ASTOR.IS","BIMAS.IS","BRSAN.IS","DOAS.IS",
    "EKGYO.IS","ENJSA.IS","ENKAI.IS","EREGL.IS","FROTO.IS","GARAN.IS","GUBRF.IS","HEKTS.IS",
    "ISCTR.IS","KCHOL.IS","KONTR.IS","KOZAA.IS","KOZAL.IS","KRDMD.IS","MGROS.IS","ODAS.IS",
    "OTKAR.IS","PETKM.IS","PGSUS.IS","SAHOL.IS","SASA.IS","SISE.IS","SKBNK.IS","SOKM.IS",
    "TABGD.IS","TCELL.IS","THYAO.IS","TKFEN.IS","TAVHL.IS","TOASO.IS","TTKOM.IS","TTRAK.IS",
    "TUPRS.IS","ULKER.IS","VESBE.IS","VESTL.IS","YKBNK.IS","ZOREN.IS","ALARK.IS","CIMSA.IS",
    "GRSEL.IS","KAYSE.IS"
]

# -----------------------------
# Yardımcılar
# -----------------------------
def _flatten_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    yfinance bazen MultiIndex kolon döndürür.
    Tek ticker bile olsa güvenli olsun diye düzleştiriyoruz.
    """
    if df is None or df.empty:
        return df

    # Eğer kolonlar MultiIndex ise (('Close','THYAO.IS') gibi)
    if isinstance(df.columns, pd.MultiIndex):
        # tercih: bu ticker'ın alt kolonlarını çek
        if ticker in df.columns.get_level_values(-1):
            sub = df.xs(ticker, axis=1, level=-1, drop_level=True).copy()
            # sub kolonları: Open High Low Close Adj Close Volume
            df = sub
        else:
            # MultiIndex'i düz stringe indir
            df.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in df.columns]

    # Standartlaştır
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Bazı durumlarda 'Adj Close' gelmeyebilir, sorun değil
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    return df[keep].dropna()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
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
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Minimum kontrol
    need = ["Open","High","Low","Close","Volume"]
    if any(c not in df.columns for c in need):
        return pd.DataFrame()

    out = df.copy()
    out["MA20"] = out["Close"].rolling(20, min_periods=20).mean()
    out["MA50"] = out["Close"].rolling(50, min_periods=50).mean()
    out["RSI14"] = rsi(out["Close"], 14)
    out["ATR14"] = atr(out, 14)
    out["VOL_MA20"] = out["Volume"].rolling(20, min_periods=20).mean()

    # 1 yıllık aralık konumu (0-100) için 252 iş günü
    window = 252 if len(out) >= 252 else min(len(out), 200)
    if window >= 60:
        roll_low = out["Close"].rolling(window, min_periods=window).min()
        roll_high = out["Close"].rolling(window, min_periods=window).max()
        out["RANGE_POS"] = (out["Close"] - roll_low) / (roll_high - roll_low).replace(0, np.nan) * 100
    else:
        out["RANGE_POS"] = np.nan

    # Son 120 gün zirveden düşüş %
    w = 120 if len(out) >= 120 else min(len(out), 60)
    if w >= 30:
        hh = out["Close"].rolling(w, min_periods=w).max()
        out["DROP_FROM_HH"] = (hh - out["Close"]) / hh.replace(0, np.nan) * 100
    else:
        out["DROP_FROM_HH"] = np.nan

    # Hacim oranı (bugün / 20g ort)
    out["VOL_X"] = out["Volume"] / out["VOL_MA20"]

    return out

def plan_levels(last: pd.Series) -> dict:
    """
    'Ben olsam' planı: tamamen kural tabanlı.
    Entry: referans fiyat = son kapanış
    Stop: ATR tabanlı + MA20/MA50 altı koruma
    Targets: son 60 gün tepe + R:R mantığı
    """
    close = float(last["Close"])
    ma20 = float(last["MA20"]) if pd.notna(last["MA20"]) else np.nan
    ma50 = float(last["MA50"]) if pd.notna(last["MA50"]) else np.nan
    a = float(last["ATR14"]) if pd.notna(last["ATR14"]) else (close * 0.03)

    # Stop: 1.5 ATR aşağı veya MA20'nin biraz altı (hangisi daha "korumacı" ise)
    stop1 = close - 1.5 * a
    stop2 = ma20 - 0.5 * a if not np.isnan(ma20) else stop1
    stop = min(stop1, stop2)

    # Hedef-1: yakın direnç -> MA50 üstünde ise 60g tepe, değilse MA50 çevresi
    # Hedef-2: 60g tepe veya 2.5R
    # Not: burada "sat" demiyoruz; "ben olsam kâr bölgesi" diyoruz.
    return {
        "ref": close,
        "stop": stop
    }

def score_candidate(last: pd.Series) -> tuple[bool, list[str]]:
    """
    'Ben olsam alırdım' filtresi (kural seti):
    - MA50 üstünde (trend lehine) VEYA (drop>=15 ve RSI<45 ile dip bölgesi)
    - Hacim en az ortalama civarı (VOL_X >= 0.9) tercih
    """
    reasons = []
    ok = False

    rsi14 = float(last["RSI14"]) if pd.notna(last["RSI14"]) else np.nan
    drop = float(last["DROP_FROM_HH"]) if pd.notna(last["DROP_FROM_HH"]) else np.nan
    volx = float(last["VOL_X"]) if pd.notna(last["VOL_X"]) else np.nan
    close = float(last["Close"])
    ma50 = float(last["MA50"]) if pd.notna(last["MA50"]) else np.nan
    ma20 = float(last["MA20"]) if pd.notna(last["MA20"]) else np.nan

    trend_up = (not np.isnan(ma50)) and (close > ma50)
    deep_pullback = (not np.isnan(drop)) and (drop >= 15) and (not np.isnan(rsi14)) and (rsi14 <= 45)

    if trend_up:
        reasons.append("Trend: Fiyat MA50 üstünde (güçlü/pozitif).")
    if deep_pullback:
        reasons.append("Düşüş: Zirveden %15+ geri çekilme + RSI düşük (tepki ihtimali).")
    if not np.isnan(ma20) and close > ma20:
        reasons.append("Kısa vade: Fiyat MA20 üstünde (kısa vade pozitif).")
    if not np.isnan(volx):
        if volx >= 1.2:
            reasons.append("Hacim: 20g ortalamanın belirgin üstünde (hareket daha anlamlı).")
        elif volx >= 0.9:
            reasons.append("Hacim: 20g ortalamasına yakın (nötr).")
        else:
            reasons.append("Hacim: 20g ortalamanın altında (hareket daha kolay sönebilir).")

    ok = trend_up or deep_pullback
    return ok, reasons

@st.cache_data(ttl=60*60, show_spinner=False)
def fetch_one(ticker: str, period="1y") -> pd.DataFrame:
    raw = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    df = _flatten_ohlcv(raw, ticker)
    return df

def fmt(x, d=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "-"
    return f"{x:,.{d}f}"

# -----------------------------
# UI Kontroller
# -----------------------------
colA, colB, colC = st.columns([1.2, 1.2, 1.6])
with colA:
    show_only_candidates = st.toggle("Sadece 'Ben olsam alırdım' olanları göster", value=True)
with colB:
    sort_key = st.selectbox("Sırala", ["Ben olsam skoru", "Zirveden düşüş (yüksekten)", "Hacim oranı (yüksekten)", "RSI (düşükten)"])
with colC:
    st.write("ℹ️ Kartlar günlük kapanış verisiyle hesaplanır. BIST50 listesi sabittir (istersen güncelleriz).")

st.divider()

# -----------------------------
# Tarama
# -----------------------------
rows = []
errors = []

with st.spinner("BIST50 taranıyor..."):
    for t in BIST50:
        try:
            df = fetch_one(t, period="1y")
            if df is None or df.empty or len(df) < 80:
                continue
            feat = build_features(df)
            if feat.empty:
                continue
            last = feat.iloc[-1].copy()
            ok, reasons = score_candidate(last)

            # Skor: sadece sıralama için (tavsiye değil)
            score = 0
            if pd.notna(last["MA50"]) and float(last["Close"]) > float(last["MA50"]):
                score += 2
            if pd.notna(last["DROP_FROM_HH"]) and float(last["DROP_FROM_HH"]) >= 15:
                score += 1
            if pd.notna(last["RSI14"]) and float(last["RSI14"]) <= 45:
                score += 1
            if pd.notna(last["VOL_X"]) and float(last["VOL_X"]) >= 1.2:
                score += 1

            rows.append({
                "ticker": t,
                "close": float(last["Close"]),
                "drop": float(last["DROP_FROM_HH"]) if pd.notna(last["DROP_FROM_HH"]) else np.nan,
                "rsi": float(last["RSI14"]) if pd.notna(last["RSI14"]) else np.nan,
                "volx": float(last["VOL_X"]) if pd.notna(last["VOL_X"]) else np.nan,
                "range_pos": float(last["RANGE_POS"]) if pd.notna(last["RANGE_POS"]) else np.nan,
                "ma20": float(last["MA20"]) if pd.notna(last["MA20"]) else np.nan,
                "ma50": float(last["MA50"]) if pd.notna(last["MA50"]) else np.nan,
                "atr": float(last["ATR14"]) if pd.notna(last["ATR14"]) else np.nan,
                "ok": ok,
                "reasons": reasons,
                "score": score,
                "feat": feat  # kart detayında grafik/direnç için
            })
        except Exception as e:
            errors.append((t, str(e)))

# Filtrele
if show_only_candidates:
    rows = [r for r in rows if r["ok"]]

# Sırala
if sort_key == "Ben olsam skoru":
    rows = sorted(rows, key=lambda r: (r["score"], r["volx"] if not np.isnan(r["volx"]) else 0), reverse=True)
elif sort_key == "Zirveden düşüş (yüksekten)":
    rows = sorted(rows, key=lambda r: (r["drop"] if not np.isnan(r["drop"]) else -1), reverse=True)
elif sort_key == "Hacim oranı (yüksekten)":
    rows = sorted(rows, key=lambda r: (r["volx"] if not np.isnan(r["volx"]) else -1), reverse=True)
elif sort_key == "RSI (düşükten)":
    rows = sorted(rows, key=lambda r: (r["rsi"] if not np.isnan(r["rsi"]) else 999))

st.subheader(f"📌 Liste ({len(rows)} hisse)")

# -----------------------------
# Kart Kart Gösterim
# -----------------------------
if not rows:
    st.info("Filtrelere uyan hisse bulunamadı. Filtreyi kapatıp tüm listeyi görebilirsin.")
else:
    for r in rows:
        t = r["ticker"]
        feat = r["feat"]
        last = feat.iloc[-1]

        # Direnç/tepe (60g)
        w = 60 if len(feat) >= 60 else len(feat)
        top60 = float(feat["Close"].tail(w).max())
        # Basit hedefler (kural tabanlı)
        base = plan_levels(last)
        ref = base["ref"]
        stop = base["stop"]

        # Hedef 1: 1.5R veya MA50/Top60 (yakın olan)
        R = max(ref - stop, ref * 0.01)
        t1_rr = ref + 1.5 * R
        t2_rr = ref + 2.5 * R

        # Yakın direnç adayı: top60
        target1 = min(max(t1_rr, ref), top60)  # ref üstü olsun
        target2 = max(t2_rr, top60)

        # "Ben olsam alırdım" metni (artık daha net)
        ben_olsam = []
        if r["ok"]:
            ben_olsam.append(f"✅ **Ben olsam almayı düşünürdüm** (kural filtresini geçti).")
        else:
            ben_olsam.append("⛔ **Ben olsam almazdım** (kural filtresini geçmedi).")

        ben_olsam.append(f"• **Referans (kapanış):** {fmt(ref)}")
        ben_olsam.append(f"• **Stop/Temkin seviyesi (örnek):** {fmt(stop)}  _(altına sarkarsa plan bozulur)_")
        ben_olsam.append(f"• **Kâr bölgesi 1 (örnek):** {fmt(target1)}  _(ilk kısmi kâr için)_")
        ben_olsam.append(f"• **Kâr bölgesi 2 (örnek):** {fmt(target2)}  _(güç devam ederse)_")
        ben_olsam.append("• **Not:** Hacim düşerken fiyat yükseliyorsa hareket çabuk sönebilir; hacim artıyorsa hareket daha anlamlı olur.")

        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1])
            with c1:
                st.markdown(f"### {t.replace('.IS','')}")
                st.caption(t)

            with c2:
                st.metric("Kapanış", fmt(r["close"]))
            with c3:
                st.metric("Zirveden düşüş (120g)", f"%{int(round(r['drop']))}" if not np.isnan(r["drop"]) else "-")
            with c4:
                st.metric("RSI(14)", f"{int(round(r['rsi']))}" if not np.isnan(r["rsi"]) else "-")
            with c5:
                st.metric("Hacim / 20g Ort", f"{fmt(r['volx'],2)}x" if not np.isnan(r["volx"]) else "-")

            # Sistem durumu
            st.markdown("#### 🧠 Sistem Durumu (Bilgi Amaçlı)")
            bullets = []
            if not np.isnan(r["range_pos"]):
                bullets.append(f"• 1 yıllık aralık konumu: **{int(round(r['range_pos']))}/100**")
            if not np.isnan(r["rsi"]):
                if r["rsi"] >= 70:
                    bullets.append(f"• RSI yüksek (**{int(round(r['rsi']))}**) → kısa vadede **ısınmış** olabilir.")
                elif r["rsi"] <= 30:
                    bullets.append(f"• RSI düşük (**{int(round(r['rsi']))}**) → **aşırı satım**, tepki ihtimali artabilir.")
                else:
                    bullets.append(f"• RSI dengeli (**{int(round(r['rsi']))}**) → aşırı alım/satım yok.")
            if not np.isnan(r["ma50"]):
                bullets.append("• Fiyat **MA50 üzerinde** → orta vadede trend daha güçlü." if r["close"] > r["ma50"] else "• Fiyat **MA50 altında** → orta vadede zayıf.")
            if not np.isnan(r["ma20"]):
                bullets.append("• Fiyat **MA20 üzerinde** → kısa vadede pozitif." if r["close"] > r["ma20"] else "• Fiyat **MA20 altında** → kısa vadede zayıf.")
            if not np.isnan(r["volx"]):
                if r["volx"] >= 1.2:
                    bullets.append("• Hacim güçlü → hareket daha dikkat çekici olabilir.")
                elif r["volx"] >= 0.9:
                    bullets.append("• Hacim nötr → ortalama seviyelerde.")
                else:
                    bullets.append("• Hacim zayıf → hareket daha kolay sönümlenebilir.")
            if not np.isnan(r["range_pos"]):
                if r["range_pos"] >= 80:
                    bullets.append("• 1 yıllık aralığın **üst bölgesi** → daha pahalı bölge.")
                elif r["range_pos"] <= 30:
                    bullets.append("• 1 yıllık aralığın **alt bölgesi** → daha ucuz/ilgi az bölge.")

            # Filtre sebepleri
            for rr in r["reasons"]:
                bullets.append(f"• {rr}")

            st.write("\n".join(bullets))

            # Ben olsam plan
            st.markdown("#### 🧭 Ben olsam (örnek plan – net seviyeler)")
            st.write("\n".join(ben_olsam))

            # İsteğe bağlı mini grafik
            with st.expander("📈 Mini grafik (Kapanış + MA20/MA50)", expanded=False):
                plot_df = feat[["Close","MA20","MA50"]].copy()
                st.line_chart(plot_df)

# Hata raporu (opsiyonel)
if errors:
    with st.expander(f"⚠️ Veri çekilemeyenler ({len(errors)})", expanded=False):
        for t, e in errors[:30]:
            st.write(f"- {t}: {e}")
        if len(errors) > 30:
            st.write("... (liste uzadı)")
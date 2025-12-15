import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Test", layout="wide")
st.title("✅ Streamlit Cloud Test Çalışıyor mu?")

st.write("Eğer bu sayfa açılıyorsa: deploy ortamın OK, sorun yfinance/bağlantı tarafında.")

p = st.progress(0)
for i in range(100):
    time.sleep(0.01)
    p.progress((i+1)/100)

df = pd.DataFrame({
    "A": np.random.randn(10),
    "B": np.random.randn(10),
})
st.dataframe(df)
st.success("Test bitti.")
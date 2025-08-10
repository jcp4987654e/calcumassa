
import streamlit as st
from pathlib import Path
import streamlit.components.v1 as components

st.set_page_config(page_title="Calculadoras Eléctricas PRO (AR)", layout="wide")
html_file = "index_pro.html"
src = Path(__file__).with_name(html_file).read_text(encoding="utf-8")
components.html(src, height=1250, scrolling=True)

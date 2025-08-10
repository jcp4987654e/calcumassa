
import streamlit as st
from pathlib import Path
import streamlit.components.v1 as components

st.set_page_config(page_title="Calculadoras Técnicas Eléctricas", layout="wide")
html_file = "index_conservative_fix.html"
components.html(Path(__file__).with_name(html_file).read_text(encoding="utf-8"), height=1200, scrolling=True)

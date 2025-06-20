import streamlit as st
import streamlit.components.v1 as components
import os

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Calculadora Eléctrica Profesional",
    page_icon="⚡",
    layout="wide" # Ocupa todo el ancho de la pantalla
)

# --- Función para cargar y mostrar el archivo HTML ---
def serve_html(file_path):
    """
    Esta función lee un archivo HTML y lo muestra en Streamlit.
    Ajusta la altura del componente para que ocupe la mayor parte de la pantalla.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            # Usamos st.components.v1.html para renderizar el HTML.
            # 'scrolling=True' permite el scroll dentro del iframe si el contenido es muy largo.
            # 'height=1200' le da un tamaño inicial generoso.
            components.html(html_content, height=1200, scrolling=True)
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{file_path}'. Asegúrate de que 'index.html' esté en el mismo directorio que 'app.py'.")

# --- Punto de Entrada Principal ---
if __name__ == "__main__":
    # El nombre del archivo HTML que queremos mostrar.
    html_file = "index.html"
    
    # Llamamos a la función para servir el archivo.
    serve_html(html_file)

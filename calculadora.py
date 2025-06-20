import http.server
import socketserver

# --- Configuración del Servidor ---
# Elige el puerto en el que quieres que se ejecute la calculadora.
# 8000 es una opción común y segura.
PORT = 8000

# --- Creación del Servidor ---
# Esta línea prepara el servidor para que sirva los archivos
# del directorio en el que se encuentra este script.
Handler = http.server.SimpleHTTPRequestHandler

# --- Ejecución del Servidor ---
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("¡Servidor iniciado!")
    print(f"Abre tu navegador y ve a: http://localhost:{PORT}")
    print("Para detener el servidor, presiona Ctrl + C en esta terminal.")
    
    # El servidor se quedará corriendo hasta que lo detengas manualmente.
    httpd.serve_forever()


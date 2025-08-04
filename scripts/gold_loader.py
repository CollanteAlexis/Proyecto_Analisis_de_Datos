import os
import zipfile
import requests

def load_or_download_db(db_name="gold.db", zip_url=None, zipped=True):
    """
    Descarga y extrae la base de datos si no está presente localmente.

    Parámetros:
        db_name (str): Nombre del archivo SQLite a verificar/usar.
        zip_url (str): URL del archivo .zip que contiene la base.
        zipped (bool): Si se espera que el archivo esté comprimido (.zip).

    Retorna:
        str: Ruta local del archivo .db listo para conectar.
    """
    if os.path.exists(db_name):
        print(f"✔ Base de datos {db_name} ya está presente.")
        return db_name

    if zip_url is None:
        raise ValueError("❌ Se debe especificar una URL de descarga si no existe la base local.")

    zip_name = db_name + ".zip" if zipped else db_name
    print(f"⬇️ Descargando base desde {zip_url}...")

    response = requests.get(zip_url)
    with open(zip_name, "wb") as f:
        f.write(response.content)

    if zipped:
        print(f"📦 Extrayendo {zip_name}...")
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_name)

    print(f"✅ Base de datos disponible: {db_name}")
    return db_name

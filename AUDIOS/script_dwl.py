import os
import csv
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

TALKBANK_COOKIE = os.getenv("TALKBANK_COOKIE")

BASE_URLS = {
    "Control": "https://media.talkbank.org:443/dementia/English/Pitt-orig/Control/cookie/",
    "Dementia": "https://media.talkbank.org:443/dementia/English/Pitt-orig/Dementia/cookie/",
}

OUTPUT_DIR = BASE_DIR / "data" / "raw" / "pitt_cookie_audio"
METADATA_PATH = BASE_DIR / "data" / "raw" / "pitt_cookie_metadata.csv"

DEBUG_HTML_PATH = BASE_DIR / "debug_talkbank_page.html"


# ============================================================
# SESIÓN
# ============================================================

def make_session():
    """
    Crea una sesión HTTP usando la cookie copiada desde Chrome.
    """
    if not TALKBANK_COOKIE:
        raise RuntimeError(
            "Falta TALKBANK_COOKIE en el archivo .env.\n"
            "Crea PFCI/.env y coloca:\n"
            "TALKBANK_COOKIE=tu_cookie_completa"
        )

    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Cookie": TALKBANK_COOKIE,
    })

    return session


# ============================================================
# PARSEO DE LINKS
# ============================================================

def get_mp3_links(session, page_url):
    """
    Lee una carpeta de TalkBank y extrae todos los links .mp3.
    """
    print(f"\n[INFO] Visitando: {page_url}")

    response = session.get(page_url, timeout=30)

    print("STATUS:", response.status_code)
    print("FINAL URL:", response.url)
    print("CONTENT TYPE:", response.headers.get("Content-Type"))

    response.raise_for_status()

    # Guardamos el HTML para debug
    with open(DEBUG_HTML_PATH, "w", encoding="utf-8", errors="ignore") as f:
        f.write(response.text)

    if ".mp3" not in response.text.lower():
        print("[WARNING] No se encontraron .mp3 en el HTML recibido.")
        print("[WARNING] Primeros 500 caracteres del HTML:")
        print(response.text[:500])
        print(f"[WARNING] Revisa el archivo debug: {DEBUG_HTML_PATH}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if ".mp3" in href.lower():
            file_url = urljoin(page_url, href)

            # Forzar modo descarga si no viene
            if "?f=save" not in file_url:
                file_url = file_url.split("?")[0] + "?f=save"

            links.append(file_url)

    # Quitar duplicados preservando orden
    unique_links = []
    seen = set()

    for link in links:
        if link not in seen:
            unique_links.append(link)
            seen.add(link)

    print(f"[INFO] MP3 encontrados: {len(unique_links)}")
    print("[INFO] Primeros MP3:", unique_links[:5])

    return unique_links


def filename_from_url(url):
    """
    Extrae el nombre del archivo desde la URL.
    Ejemplo:
    https://.../002-0.mp3?f=save -> 002-0.mp3
    """
    parsed = urlparse(url)
    return os.path.basename(parsed.path)


# ============================================================
# DESCARGA
# ============================================================

def download_file(session, url, output_path):
    """
    Descarga un archivo MP3.
    """
    output_path = Path(output_path)

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[SKIP] Ya existe: {output_path}")
        return

    print(f"[DOWNLOADING] {url}")

    with session.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if "text/html" in content_type:
            raise RuntimeError(
                f"El servidor devolvió HTML en vez de audio para:\n{url}\n"
                "Probablemente la cookie expiró o no es la correcta."
            )

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    if output_path.stat().st_size == 0:
        raise RuntimeError(f"El archivo descargado está vacío: {output_path}")

    print(f"[OK] Guardado: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("========================================")
    print("DESCARGA PITT COOKIE - TALKBANK")
    print("========================================")
    print("BASE_DIR:", BASE_DIR)
    print("ENV_PATH:", ENV_PATH)
    print("Existe .env:", ENV_PATH.exists())
    print("Cookie cargada:", TALKBANK_COOKIE is not None)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    session = make_session()

    metadata_rows = []

    for group_name, url in BASE_URLS.items():
        label_id = 0 if group_name == "Control" else 1
        label_name = "Non-AD" if group_name == "Control" else "AD"

        group_output_dir = OUTPUT_DIR / group_name
        group_output_dir.mkdir(parents=True, exist_ok=True)

        print("\n========================================")
        print(f"LEYENDO CARPETA: {group_name}")
        print("========================================")

        mp3_links = get_mp3_links(session, url)

        print(f"[INFO] Audios encontrados en {group_name}: {len(mp3_links)}")

        for file_url in mp3_links:
            filename = filename_from_url(file_url)
            sample_id = Path(filename).stem

            output_path = group_output_dir / filename

            try:
                download_file(session, file_url, output_path)
            except Exception as e:
                print(f"[ERROR] No se pudo descargar {file_url}")
                print("Motivo:", e)
                continue

            metadata_rows.append({
                "sample_id": sample_id,
                "group": group_name,
                "label_name": label_name,
                "label_id": label_id,
                "audio_path": str(output_path).replace("\\", "/"),
                "source_url": file_url,
            })

            time.sleep(0.2)

    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "group",
                "label_name",
                "label_id",
                "audio_path",
                "source_url",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print("\n========================================")
    print("DESCARGA TERMINADA")
    print("========================================")
    print(f"Metadata guardada en: {METADATA_PATH}")
    print(f"Total audios registrados: {len(metadata_rows)}")

    control_count = sum(1 for row in metadata_rows if row["group"] == "Control")
    dementia_count = sum(1 for row in metadata_rows if row["group"] == "Dementia")

    print(f"Control / Non-AD: {control_count}")
    print(f"Dementia / AD: {dementia_count}")


if __name__ == "__main__":
    main()
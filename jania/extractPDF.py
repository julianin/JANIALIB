import io
import base64
from typing import BinaryIO, Dict, Any, Optional

try:
    import openai
except ImportError:
    openai = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from jania.configenv import env

def extractPDF(
    prompt: str,
    archivo: BinaryIO,
    nombre_archivo: str,
    model: str = "gpt-4-vision-preview",
    max_images: int = 10,
    openai_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Recibe un prompt y un PDF, convierte cada página a imagen (en memoria)
    y consulta el LLM Vision. Devuelve la respuesta del LLM.
    Si no se indica openai_api_key, la busca con env("OPENAI_API_KEY").
    """
    if fitz is None:
        raise ImportError("Falta la librería pymupdf. Instálala con 'pip install pymupdf'.")

    pdf_bytes = archivo.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images_data = []
    for page_num in range(min(max_images, doc.page_count)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=400)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        images_data.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "Analiza el siguiente PDF página por página:"},
            *images_data
        ]}
    ]

    api_key = openai_api_key or env("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No se ha proporcionado clave OpenAI ni encontrada en configuración/env.")

    if openai is None:
        raise ImportError("Falta la librería openai. Instálala con 'pip install openai'.")

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        # max_tokens=10000,
    )

    return {"respuesta_llm": response.choices[0].message.content}


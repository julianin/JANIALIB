import io
import tempfile
from pdf2image import convert_from_path
from typing import BinaryIO, Dict, Any, Optional
import base64

try:
    import openai
except ImportError:
    openai = None

from configenv import env

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
    images_data = []
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = f"{tmpdir}/{nombre_archivo}"
        with open(fp, "wb") as f:
            f.write(archivo.read())

        # Convertir PDF a imágenes en memoria
        images = convert_from_path(fp)
        for img in images[:max_images]:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            images_data.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })
            buf.close()

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

    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )

    return {"respuesta_llm": response.choices[0].message.content}

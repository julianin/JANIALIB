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
        model: str = "gpt-4o",  # Cambiado a gpt-4o que es más actual
        max_images: int = 10,
        openai_api_key: Optional[str] = None,
        dpi: int = 150  # Reducido para mejor rendimiento
) -> Dict[str, Any]:
    """
    Recibe un prompt y un PDF, convierte cada página a imagen (en memoria)
    y consulta el LLM Vision. Devuelve la respuesta del LLM.
    Si no se indica openai_api_key, la busca con env("OPENAI_API_KEY").
    """
    if fitz is None:
        raise ImportError("Falta la librería pymupdf. Instálala con 'pip install pymupdf'.")

    if openai is None:
        raise ImportError("Falta la librería openai. Instálala con 'pip install openai'.")

    # Leer el PDF
    pdf_bytes = archivo.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        images_data = []
        total_pages = min(max_images, doc.page_count)

        for page_num in range(total_pages):
            page = doc.load_page(page_num)

            # Convertir página a imagen con DPI ajustable
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("png")

            # Convertir a base64
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            images_data.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "high"  # Agregado para mejor calidad de análisis
                }
            })

    finally:
        doc.close()  # Importante cerrar el documento

    # Construir mensajes
    content = [{"type": "text", "text": f"Analiza el siguiente PDF '{nombre_archivo}' página por página:"}]
    content.extend(images_data)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content}
    ]

    # Configurar API key
    api_key = openai_api_key or env("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No se ha proporcionado clave OpenAI ni encontrada en configuración/env.")

    client = openai.OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4000,  # Agregado límite razonable
            temperature=0.1  # Para respuestas más consistentes
        )

        return {
            "respuesta_llm": response.choices[0].message.content,
            "total_pages_processed": total_pages,
            "model_used": model
        }

    except openai.OpenAIError as e:
        return {
            "error": f"Error de OpenAI: {str(e)}",
            "total_pages_processed": 0
        }
    except Exception as e:
        return {
            "error": f"Error inesperado: {str(e)}",
            "total_pages_processed": 0
        }
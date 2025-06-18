import re
import json


def cleanJson(text):
    """
    Extrae y decodifica el primer JSON válido encontrado dentro de un string.
    Soporta tanto objetos {} como listas [] anidadas.

    Parámetros:
        text (str): Texto a analizar.
    Devuelve:
        dict/list/None: El JSON decodificado, o None si no se encontró nada válido.
    """
    # Busca bloques que parecen JSON válidos (objetos o listas, incluso anidados)
    for x in re.findall(
            r'(\{(?:[^{}]|\{[^{}]*\})*\}|\[(?:[^\[\]]|\[[^\[\]]*\])*\])',
            text,
            re.DOTALL
    ):
        try:
            return json.loads(x)
        except Exception:
            pass

    # Intenta extraer desde la primera llave/ corchete hasta la última
    try:
        return json.loads(text[text.find('{'):text.rfind('}') + 1])
    except Exception:
        pass
    try:
        return json.loads(text[text.find('['):text.rfind(']') + 1])
    except Exception:
        pass

    return None

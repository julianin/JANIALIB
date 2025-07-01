import re
import json
import ast

def cleanJSON(text, indent=2, debug=False):
    """
    Extrae el primer bloque JSON válido, lo limpia, corrige errores típicos, y lo devuelve
    como string JSON formateado. Soporta outputs de LLM con basura, markdown, comillas mal puestas,
    true/false/null, varios bloques juntos, etc.
    Si debug=True, retorna también el fragmento corregido que se intentó parsear.
    """

    if not isinstance(text, str):
        return None

    original = text

    text = re.sub(r"^\s*(```\s*json|```|json|output:|respuesta:?|responde en json|aquí tienes.*?:)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(```)+\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^#+.*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(//|#).*\n?', '', text, flags=re.MULTILINE)

    pre_idx = min([i for i in [text.find("{"), text.find("[")] if i >= 0], default=None)
    if pre_idx is not None and pre_idx > 0:
        text = text[pre_idx:]

    post_idx = max([text.rfind("}"), text.rfind("]")])
    if post_idx != -1 and post_idx < len(text)-1:
        text = text[:post_idx+1]

    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\xA0-\uFFFF]", "", text)

    pattern = r'(\{(?:[^{}]|(?R))*\}|\[(?:[^\[\]]|(?R))*\])'
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        for start, end in [('{', '}'), ('[', ']')]:
            s = text.find(start)
            e = text.rfind(end)
            if s != -1 and e != -1 and e > s:
                matches = [text[s:e+1]]

    # Probar parseos tolerantes y variantes de cada bloque
    tried = []
    for candidate in matches:
        candidate = candidate.strip()
        for fix in [lambda x: x,
                    lambda x: x.replace("'", '"'),
                    lambda x: re.sub(r",(\s*[\}\]])", r"\1", x),
                    lambda x: re.sub(r"(['\"])?([a-zA-Z0-9_]+)(['\"])?:", r'"\2":', x),
                    lambda x: x.replace("True", "true").replace("False", "false").replace("None", "null")
                    ]:
            block = fix(candidate)
            block_ast = block.replace("true", "True").replace("false", "False").replace("null", "None")
            try:
                obj = json.loads(block)
                formatted = json.dumps(obj, ensure_ascii=False, indent=indent)
                if debug:
                    return formatted, block
                else:
                    return formatted
            except Exception:
                pass
            try:
                obj = ast.literal_eval(block_ast)
                if isinstance(obj, (dict, list)):
                    formatted = json.dumps(obj, ensure_ascii=False, indent=indent)
                    if debug:
                        return formatted, block
                    else:
                        return formatted
            except Exception:
                tried.append(block[:200])
    try:
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False, indent=indent)
    except Exception:
        pass

    if debug:
        return None, tried
    return None

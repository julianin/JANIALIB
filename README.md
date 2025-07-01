# JANIA LIB

Librería multipropósito en Python: utilidades para procesamiento de configuración, extracción de datos de PDF usando LLM y limpieza avanzada de JSON.

## Instalación

Requiere Python 3.7 o superior.

```
pip install jania
```

O bien, clona el repositorio:

```
git clone https://github.com/julianin/JANIALIB.git
cd JANIALIB
pip install .
```

## Dependencias

* toml
* PyYAML
* python-dotenv
* pdf2image
* Pillow
* openai (opcional, solo si usas funciones LLM)

## Funcionalidades principales

### 1. Limpieza y formateo de JSON: `cleanJSON`

Función para extraer, limpiar y formatear el primer bloque JSON válido de un texto (útil para outputs de LLM o APIs poco estrictas).

**Uso básico:**

```python
from jania import cleanJSON
json_str = cleanJSON(raw_text)
```

**Parámetros:**

* `text`: (str) Texto de entrada donde buscar JSON.
* `indent`: (int, opcional) Espacios de indentación para el JSON de salida. Default: 2.
* `debug`: (bool, opcional) Si es True, también devuelve el fragmento corregido que se intentó parsear. Default: False.

**Devuelve:**

* Un string con el JSON limpio y formateado, o None si no se encuentra JSON válido.

**Ejemplo:**

```python
json_str = cleanJSON('output: {a:1, b:2, c:true}')
print(json_str)
# {
#   "a": 1,
#   "b": 2,
#   "c": true
# }
```

### 2. Gestión de configuración: `env` y `env_config`

Utilidades para leer configuración multi-formato y variables de entorno, con prioridad flexible.

#### `env_config(filename)`

Carga un archivo de configuración extra (acepta .py, .toml, .yaml/.yml, .json, .txt). El archivo se usará como fuente prioritaria tras el entorno.

**Uso:**

```python
from jania import env, env_config
env_config('settings.toml')
```

**Parámetro:**

* `filename`: Ruta al archivo de configuración.

#### `env(key, fallback=None)`

Busca el valor de una clave de configuración (str) siguiendo este orden:

1. Variable de entorno
2. Config cargado por `env_config()`
3. config.py
4. settings.toml
5. settings.yaml
6. `fallback` (valor por defecto)

**Uso:**

```python
valor = env('API_KEY', fallback='1234')
```

**Devuelve:**

* Valor de la clave (str) o `fallback` si no se encuentra.

### 3. Extracción y análisis de PDF con LLM: `extractPDF`

Convierte cada página de un PDF a imagen y consulta un modelo LLM Vision (por defecto: gpt-4-vision-preview) para analizarlo.

**Uso típico:**

```python
from jania import extractPDF
with open('documento.pdf', 'rb') as f:
    respuesta = extractPDF(
        prompt='Resume el documento:',
        archivo=f,
        nombre_archivo='documento.pdf',
        model='gpt-4-vision-preview',  # opcional
        max_images=10,                 # opcional
        openai_api_key=None            # opcional
    )
```

**Parámetros:**

* `prompt`: (str) Instrucción para el LLM.
* `archivo`: (BinaryIO) Archivo PDF abierto en modo binario.
* `nombre_archivo`: (str) Nombre del archivo (para almacenamiento temporal).
* `model`: (str, opcional) Modelo de OpenAI Vision a usar. Default: "gpt-4-vision-preview".
* `max_images`: (int, opcional) Número máximo de páginas/imágenes a procesar. Default: 10.
* `openai_api_key`: (str, opcional) API Key de OpenAI. Si no se da, se busca en configuración/env.

**Devuelve:**

* Diccionario con la respuesta del LLM bajo la clave `respuesta_llm`.

**Notas:**

* Requiere la librería `openai` y una API Key válida.
* Si no tienes la clave, colócala en una variable de entorno `OPENAI_API_KEY` o config compatible.

---

## Ejemplo de uso general

```python
from jania import env, env_config
from jania import cleanJSON
from jania import extractPDF

# Configuración
env_config('settings.toml')
clave = env('OPENAI_API_KEY')

# Limpieza de JSON
json_str = cleanJSON('Responde en json: {"x":1, "y":true, "z":null}')

# Análisis de PDF
with open('ejemplo.pdf', 'rb') as f:
    resultado = extractPDF('Describe cada página', f, 'ejemplo.pdf', openai_api_key=clave)
    print(resultado['respuesta_llm'])
```

---

## Licencia

MIT

## Autor

Julian Ania

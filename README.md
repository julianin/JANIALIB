# jania

Librería Python ultra-minimalista para gestión de configuración:
prioridad automática entre entorno, archivos y valores por defecto.

## Instalación

```bash
pip install jania
```

## Uso

```python
from jania import env, env_config

# Cargar un archivo de configuración custom (opcional)
env_config("settings.toml")

# Obtener variable con prioridad: entorno > archivo custom > config.py > settings.toml > settings.yaml > fallback
valor = env("ENDPOINT_OUT_MSG", "http://localhost:5000")
```

## Soporta archivos de configuración en formato:
- Python (`config.py`)
- TOML (`settings.toml`)
- YAML (`settings.yaml`)
- JSON (`config.json`)
- TXT (`key=valor` por línea)

Puedes añadir nuevos formatos fácilmente.

---

## Ejemplo completo

Supón que tienes los siguientes archivos en tu proyecto:

**config.py**
```python
ENDPOINT_OUT_MSG = "http://localhost:8000/msg"
```

**settings.toml**
```toml
ENDPOINT_OUT_MSG = "http://localhost:8080/msg"
```

**settings.yaml**
```yaml
ENDPOINT_OUT_MSG: "http://localhost:8090/msg"
```

Entonces:

```python
from jania import env

print(env("ENDPOINT_OUT_MSG", "http://localhost:5000"))
```

Imprimirá (según prioridad de archivos y entorno).

---

## Licencia

MIT License © 2025 Julian Ania


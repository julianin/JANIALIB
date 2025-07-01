import os
import toml
import yaml
import json
from dotenv import load_dotenv
load_dotenv()

_custom_config = None
_custom_config_dict = {}

def env_config(filename):
    """
    Carga un archivo de configuración adicional (py, toml, yaml, json, txt)
    que se usará como prioridad después del entorno.
    """
    global _custom_config, _custom_config_dict
    if filename.endswith('.py'):
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_config", filename)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _custom_config = mod
        _custom_config_dict = {}
    else:
        _custom_config = None
        with open(filename, encoding="utf-8") as f:
            if filename.endswith('.toml'):
                _custom_config_dict = toml.load(f)
            elif filename.endswith(('.yaml', '.yml')):
                _custom_config_dict = yaml.safe_load(f)
            elif filename.endswith('.json'):
                _custom_config_dict = json.load(f)
            elif filename.endswith('.txt'):
                # Formato: clave=valor (una por línea)
                _custom_config_dict = dict(
                    line.strip().split('=', 1) for line in f if '=' in line)
            else:
                raise Exception('Formato no soportado para config extra')

def env(key, fallback=None):
    """
    Busca una variable de configuración, en el siguiente orden:
      1. Variable de entorno
      2. Config cargado por env_config()
      3. config.py
      4. settings.toml
      5. settings.yaml
      6. fallback
    """
    # 1. Entorno
    val = os.getenv(key)
    if val is not None:
        return val

    # 2. Config custom si existe
    if _custom_config:
        val = getattr(_custom_config, key, None)
        if val is not None:
            return val
    elif _custom_config_dict:
        val = _custom_config_dict.get(key)
        if val is not None:
            return val

    # 3. config.py
    try:
        import config
        val = getattr(config, key, None)
        if val is not None:
            return val
    except ImportError:
        pass

    # 4. settings.toml
    try:
        with open("settings.toml", encoding="utf-8") as f:
            toml_dict = toml.load(f)
        val = toml_dict.get(key)
        if val is not None:
            return val
    except Exception:
        pass

    # 5. settings.yaml
    try:
        with open("settings.yaml", encoding="utf-8") as f:
            yaml_dict = yaml.safe_load(f)
        val = yaml_dict.get(key)
        if val is not None:
            return val
    except Exception:
        pass

    # 6. Fallback
    return fallback

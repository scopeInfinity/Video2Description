"""
Configuration Parser for V2D
"""

import json
import threading
import os

lock = threading.Lock()

def get_config():
    with lock:
        if hasattr(get_config, "config"):
            return get_config.config

        fname = os.environ.get("V2D_CONFIG_FILE", "config.json")
        with open(fname, "r") as fin:
            get_config.config = json.load(fin)
        return get_config.config

def clear():
    with lock:
        if hasattr(get_config, "config"):
            delattr(get_config, "config")

def get_app_config():
    return get_config()["app"]

def get_rpc_config():
    return get_config()["rpc"]

def get_vpreprocess_config():
    return get_config()["vpreprocess"]

def get_vocab_config():
    return get_config()["vocab"]

def get_tests_config():
    return get_config()["tests"]

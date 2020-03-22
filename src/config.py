import json
import threading
import os

config = None
lock = threading.Lock()

def clear():
	global config
	with lock:
		config = None

def getConfig():
	global config
	with lock:
		if config is not None:
			return config

		file = os.environ.get("V2D_CONFIG_FILE", "config.json")
		with open(file, "r") as f:
			config = json.load(f)
		return config

def getAppConfig():
	return getConfig()["app"]

def getRpcConfig():
	return getConfig()["rpc"]

def getVPreprocessConfig():
	return getConfig()["vpreprocess"]

def getVocabConfig():
    return getConfig()["vocab"]

def getTestsConfig():
	return getConfig()["tests"]

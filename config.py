import json

CONFIG_FILE = "config.json"
config = None

def getConfig():
	global config
	if config is not None:
		return config

	with open(CONFIG_FILE, "r") as f:
		config = json.load(f)
	return config

def getAppConfig():
	return getConfig()["app"]

def getRpcConfig():
	return getConfig()["rpc"]
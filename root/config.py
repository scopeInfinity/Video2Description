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

def getVPreprocessConfig():
	return getConfig()["vpreprocess"]

<<<<<<< HEAD:root/config.py
def getTestsConfig():
	return getConfig()["tests"]
=======
def getVocabConfig():
    return getConfig()["vocab"]
>>>>>>> 4a3f3825571d16516e1b0a5809f178572c68d191:config.py

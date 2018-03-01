from flask import Flask, render_template, request
from copy import deepcopy
import os
app = Flask(__name__)

navigation = [("./","Predict",False),("./get_ids","Get ID's",False)]
prefix = '/home/gagan.cs14/btp_VideoCaption'

def getactivenav(index):
	nav = deepcopy(navigation)
	nav[index] = (nav[index][0], nav[index][1], True)
	return nav

def get_train_ids():
	command = "python %s/VideoDataset/videohandler.py -strain" % prefix
	return os.popen(command).read()

def get_test_ids():
	command = "python %s/VideoDataset/videohandler.py -stest" % prefix
	return os.popen(command).read()

def get_val_ids():
	command = "python %s/VideoDataset/videohandler.py -sval" % prefix
	return os.popen(command).read()

def predict_ids(ids):
	command = "python %s/parser.py server -pids %s" % (prefix, ids)
	return os.popen(command).read()

def predict_fnames(fnames):
	command = "python %s/parser.py server -pfs %s" % (prefix, fnames)
	return os.popen(command).read()

@app.route("/")
def main():
	content = dict()
	if request.args.get('ids'):
		content['ids'] = request.args.get('ids')
		content['data_ids'] = predict_ids(request.args.get('ids'))
	elif request.args.get('fnames'):
		content['fnames'] = request.args.get('fnames')
		content['data_fnames'] = predict_ids(request.args.get('data_fnames'))

	return render_template('index.html', navigation = getactivenav(0), content = content)

@app.route("/get_ids")
def get_ids():
	content['train_ids'] = get_train_ids()
	content['val_ids'] = get_val_ids()
	content['test_ids'] = get_test_ids()

	return render_template('get_ids.html', navigation=getactivenav(1), content = content)

if __name__ == "__main__":
	app.run()
import argparse
import json
import os
import random
import re
import traceback

from copy import deepcopy
from flask import Flask, render_template, request, send_from_directory
from waitress import serve

from common.config import get_app_config
from common.rpc import get_rpc
from common.status import ModelWeightsStatus

app = Flask(__name__)
config = get_app_config()

PREDICT_MODE_ONLY = config["PREDICT_MODE_ONLY"]
PREFIX = config["PREFIX"]
app.config['MAX_CONTENT_LENGTH'] =  config["MAX_CONTENT_LENGTH"]
app.config['UPLOAD_FOLDER'] = config["UPLOAD_FOLDER"]
app.config['VIDEOS_FOLDER'] = config["VIDEOS_FOLDER"]

navigation = [("./","Predict",False)]

if not PREDICT_MODE_ONLY:
	navigation.extend([("./get_ids","Get ID's",False),("./play","Play Videos",False)])

	# Don't even define the methods! And definately need to improve this.
	def get_train_ids():
		command = "python %s/backend/videohandler.py -strain" % PREFIX
		return os.popen(command).read()

	def get_test_ids():
		command = "python %s/backend/videohandler.py -stest" % PREFIX
		return os.popen(command).read()

	def get_val_ids():
		command = "python %s/backend/videohandler.py -sval" % PREFIX
		return os.popen(command).read()

	def get_all_ids():
		command = "python %s/backend/videohandler.py -sval -stest -strain" % PREFIX
		return os.popen(command).read()

	def predict_ids(ids):
		proxy = get_rpc()
		return proxy.predict_ids(ids)

	@app.route("/play")
	def play():
		return render_template('play.html', navigation = getactivenav(2))

	@app.route("/get_ids")
	def get_ids():
		content = dict()
		content['ids'] = get_all_ids()
		return render_template('get_ids.html', navigation=getactivenav(1), content = content).replace("]","]<br/><br/>")

	@app.route("/predict")
	def predict_page(fnames = None):
		if request.args.get('fnames'):
			return computeAndRenderPredictionFnames(re.sub("[^0-9 ]", "", request.args.get('fnames')))
		if (not PREDICT_MODE_ONLY) and request.args.get('ids'):
			return computeAndRenderPredictionIDs(ids = re.sub("[^0-9 ]", "", request.args.get('ids')))
		return "Invalid Request"
	
	@app.route('/download', methods=['GET'])
	def download_file():
	  _id = request.args.get('id')
	  if _id  and unicode(_id).isnumeric():
	    return send_from_directory(app.config['VIDEOS_FOLDER'],str(_id)+".mp4")
	    return "File Not Exists"
	  return "Invalid Request"

def predict_fnames(fnames):
    proxy = get_rpc()
    return proxy.predict_fnames(fnames)

def model_weights_notify():
    proxy = get_rpc()
    try:
        status = proxy.get_weights_status()
        if status == str(ModelWeightsStatus.SUCCESS):
            return None
        return status
    except Exception as e:
        print("model_weights_notify failed: %s" % e)
        return "Failed to communicate."

def getactivenav(index):
    nav = deepcopy(navigation)
    nav[index] = (nav[index][0], nav[index][1], True)
    return nav

@app.route("/model_weights_status")
def model_weights_status():
    return model_weights_notify() or "[SUCCESS]"

@app.route("/")
def home():
    weights_notify = model_weights_notify()
    if PREDICT_MODE_ONLY:
        return render_template(
            'publicindex.html',
            weights_notify = weights_notify)
    else:
        return render_template('index.html', navigation = getactivenav(0))

def computeAndRenderPredictionIDs(ids):
	content = dict()
	content['ids'] = ids
	content['data_ids'] = predict_ids(ids)
	return render_template('predict.html', content = content)

def computeAndRenderPredictionFnames(fnames):
	content = dict()
	content['fnames'] = fnames
	content['data_fnames'] = predict_fnames(fnames)
	return render_template('predict.html', content = content)

# http://flask.pocoo.org/docs/0.12/patterns/fileuploads/
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['mp4']

def error(msg):
    return json.dumps({'error':msg})

def success(data):
    return json.dumps({'success':data})

@app.route('/upload', methods=['POST'])
def upload_file():
    print(request.files)
    if request.method != "POST":
        return error("Only POST requests are expected!")
    if "file" not in request.files:
        return error("No filess found!")
    file = request.files['file']
    if not file:
        return error("No file found!")
    if file.filename == '':
        return error("No filename found!")
    if not allowed_file(file.filename):
        return error("Only *.mp4 video files are supported at this moment!")
    filename = str(random.randint(0,1000000)) + ".mp4"
    filename = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    try:
        file.save(filename)
        print("File uploaded: %s" %  filename)
        output = json.loads(predict_fnames([filename]))
    except Exception as e:
        print(traceback.format_exc())
        return error("Request Failed! Exception caught while generating caption.")
    finally:
        os.unlink(filename)
    return success(output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type = int, default=5000)
    args = parser.parse_args()
    serve(app, host='0.0.0.0', port=args.port)

if __name__ == "__main__":
    main()
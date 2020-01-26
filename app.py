from flask import Flask, render_template, request, send_from_directory
from copy import deepcopy
import os, random, re
from rpc import get_rpc
from config import getAppConfig

app = Flask(__name__)
config = getAppConfig()

PREDICT_MODE_ONLY = config["PREDICT_MODE_ONLY"]
PREFIX = config["PREFIX"]
app.config['MAX_CONTENT_LENGTH'] =  config["MAX_CONTENT_LENGTH"]
app.config['UPLOAD_FOLDER'] = config["UPLOAD_FOLDER"]
app.config['VIDEOS_FOLDER'] = config["VIDEOS_FOLDER"]

navigation = [("./","Predict",False)]

if not PREDICT_MODE_ONLY:
	navigation.extend([("./get_ids","Get ID's",False),("./play","Play Videos",False)])

	# Don't even define the methods!
	def get_train_ids():
		command = "python %s/VideoDataset/videohandler.py -strain" % PREFIX
		return os.popen(command).read()

	def get_test_ids():
		command = "python %s/VideoDataset/videohandler.py -stest" % PREFIX
		return os.popen(command).read()

	def get_val_ids():
		command = "python %s/VideoDataset/videohandler.py -sval" % PREFIX
		return os.popen(command).read()

	def get_all_ids():
		command = "python %s/VideoDataset/videohandler.py -sval -stest -strain" % PREFIX
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

def getactivenav(index):
	nav = deepcopy(navigation)
	nav[index] = (nav[index][0], nav[index][1], True)
	return nav

@app.route("/")
def main():
	if PREDICT_MODE_ONLY:
		return render_template('publicindex.html', navigation = getactivenav(0))
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

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        print "POST"
        # check if the post request has the file part
        if 'file' not in request.files:
            return "File not Found"
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = str(random.randint(0,1000000)) + ".mp4"
            filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filename)
            print "Uploaded To %s" %  filename
            output = computeAndRenderPredictionFnames([filename])
            os.unlink(filename)
            return output
    return "File Upload Failed"

if __name__ == "__main__":
	app.run(host='0.0.0.0')

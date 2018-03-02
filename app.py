from flask import Flask, render_template, request
from copy import deepcopy
import os, random
app = Flask(__name__)
prefix = '/home/gagan.cs14/btp_VideoCaption'

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = prefix + "/Uploads"

navigation = [("./","Predict",False),("./get_ids","Get ID's",False)]

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

def get_all_ids():
	command = "python %s/VideoDataset/videohandler.py -sval -stest -strain" % prefix
	return os.popen(command).read()

def predict_ids(ids):
	command = "python %s/parser.py server -pids %s" % (prefix, ids)
	return os.popen(command).read()

def predict_fnames(fnames):
	command = "python %s/parser.py server -pfs %s" % (prefix, fnames)
	return os.popen(command).read()

@app.route("/")
def main():
	return render_template('index.html', navigation = getactivenav(0))

def predict(fnames = None, ids = None):
	assert (fnames is None) ^ (ids is None)
	content = dict()
	if ids is not None:
		content['ids'] = request.args.get('ids')
		content['data_ids'] = predict_ids(ids)
	elif fnames is not None:
		content['fnames'] = fnames
		content['data_fnames'] = predict_fnames(fnames)
	return render_template('predict.html', content = content)

@app.route("/predict")
def predict_page(fnames = None):
	if request.args.get('fnames'):
		return predict(fnames = request.args.get('fnames'))
	if request.args.get('ids'):
		return predict(ids = request.args.get('ids'))
	return "Invalid Request"

@app.route("/get_ids")
def get_ids():
	content = dict()
	content['ids'] = get_all_ids()
	return render_template('get_ids.html', navigation=getactivenav(1), content = content).replace("]","]<br/><br/>")

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
            output = predict(fnames = filename)
            os.unlink(filename)
            return output
    return "File Upload Failed"

if __name__ == "__main__":
	app.run(host='0.0.0.0')

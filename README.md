# Video Captioning
Generate caption for the given video clip

Branch : [VideoCaption](https://github.com/scopeInfinity/Video2Description/tree/VideoCaption)

Status : Ongoing

### Setup
* Clone repository to directory named `btp_<branch_name>`
  * `git clone https://github.com/scopeInfinity/Video2Description.git btp_VideoCaption`
  * Path for code repository is hardcoded in first few lines of `vpreprocess.py`
* Setup anaconda environment, use `environment.yml`
  * Install keras with tensorflow backend.
* Install ffmpeg
  * Configure, build and install ffmpeg from source with shared libraries 
```bash
git clone 'https://github.com/FFmpeg/FFmpeg.git'
cd FFmpeg
./configure --prefix=~/V2D_local/ --enable-shared
make
make install
```
  * Set environment variable
    * Reference [https://conda.io/docs/user-guide/tasks/manage-environments.html#saving-environment-variables](https://conda.io/docs/user-guide/tasks/manage-environments.html#saving-environment-variables)
```
export PATH=/home/gagan.cs14/V2D_local/bin:$PATH
export LD_LIBRARY_PATH=$/home/gagan.cs14/V2D_local/lib/:$LD_LIBRARY_PATH
```
* Install opencv
```
conda install opencv -c conda-forge
```


### Data Directory
File | Reference
--- | --- 
*/path/to/data_dir/VideoDataset/videodatainfo_2017.json* | http://ms-multimedia-challenge.com/2017/dataset
*/path/to/data_dir/VideoDataset/videos/[0-9]+.mp4* | Download videos based on above dataset
*/path/to/data_dir/glove/glove.6B.300d.txt* | https://nlp.stanford.edu/projects/glove/
*/path/to/data_dir/VideoDataset/cache_40_224x224/[0-9]+.npy* | Video cached files will be created on fly

### Working Directory
File | Content
--- | --- 
*/path/to/working_dir/glove.dat* | Pickle Dumped Glove Embedding
*/path/to/working_dir/vocab.dat* | Pickle Dumped Vocabulary Words
  
### Download Dataset
* Execute `python videohandler.py` from *VideoDataset* Directory
  
  
### Training Methods

* Try Iterative Learning
* Try Random Learning  

### Evaluation

#### Prerequisite
```bash
cd /path/to/eval_dir/
git clone 'https://github.com/tylin/coco-caption.git' cococaption
ln /path/to/working_dir/cocoeval.py cococaption/
```
#### Evaluate
```bash
# One can do changes in parser.py for numbers of test examples to be considered in evaluation
python parser.py predict save_all_test
python /path/to/eval_dir/cocoeval.py <results file>.txt
```

#### Sample Evaluation while training

Commit | Training | Total | CIDEr | Bleu_4 | ROUGE_L | METEOR | Model Filename 
--- | --- | --- | --- | --- | --- | --- | --- 
6a63470 | 10 epochs | 1.1642 | 0.1580 | 0.3090 | 0.4917 | 0.2055 | CAttention_ResNet_D512L512_G128G64_D1024D0.20BN_BDGRU1024_D0.2L1024DVS_model.dat_4990_loss_2.484_Cider0.360_Blue0.369_Rouge0.580_Meteor0.256
5920de9 | 17 epochs | 1.1599 | 0.1654 | 0.3022 | 0.4849 | 0.2074 | ResNet_D512L512_G128G64_D1024D0.20BN_BDLSTM1024_D0.2L1024DVS_model.dat_4987_loss_2.203_Cider0.342_Blue0.353_Rouge0.572_Meteor0.256
f5c22f7 | 17 epochs | 1.1559 | 0.1680 | 0.3000 | 0.4832 | 0.2047 | ResNet_D512L512_G128G64_D1024D0.20BN_BDGRU1024_D0.2L1024DVS_model.dat_4983_loss_2.350_Cider0.355_Blue0.353_Rouge0.571_Meteor0.247_TOTAL_1.558_BEST
bd072ac | 11 CPUhrs with Multiprocessing (16 epochs)  |  1.0736 | 0.1528 | 0.2597 | 0.4674 | 0.1936 | ResNet_D512L512_D1024D0.20BN_BDGRU1024_D0.2L1024DVS_model.dat_4986_loss_2.306_Cider0.347_Blue0.328_Rouge0.560_Meteor0.246 
3ccf5d5 | 15 CPUhrs |  1.0307 | 0.1258 | 0.2535 | 0.4619 | 0.1895 | res_mcnn_rand_b100_s500_model.dat_model1_3ccf5d5 

Major changes for model are described at the end.

Pre-trained Models : https://drive.google.com/open?id=1gexBRQfrjfcs7N5UI5NtlLiIR_xa69tK

### Web Server

- From a high RAM ram server **(S)** execute (Within conda environment)
```bash
python parser.py server -s
```
- Edit `rpc.py`
  - Change **`SERVER_IP`** to IP of server **(S)**
- Edit `app.py` and modify following according to need
  - prefix # Line 5, working directory path
  - `app.config['VIDEOS_FOLDER']`
  - `app.config['UPLOAD_FOLDER']`
- Execute `python app.py` from webserver (No need for conda environment)
- Open `http://webserver:5000/` to open Web Server for testing (under default configuration)

### Specifications

##### Commit: 3ccf5d5
- ResNet over LSTM for feature extraction
- Word by Word generation based on last prediction for Sentence Generation using LSTM
- Random Dataset Learning of training data
- Vocab Size 9448
- Glove of 300 Dimension

##### Commit: bd072ac
- ResNet over BiDirection GRU for feature extraction
- Sequential Learning of training data
- Batch Normalization + Few more tweaks in Model
- Bleu, CIDEr, Rouge, Meteor score generation for validation
- Multiprocessing keras

# Image Captioning
Generate caption for the given images

Branch : [onehot_gen](https://github.com/scopeInfinity/Video2Description/tree/onehot_gen)

Commit : [898f15778d40b67f333df0a0e744a4af0b04b16c](https://github.com/scopeInfinity/Video2Description/commit/898f15778d40b67f333df0a0e744a4af0b04b16c)

Trained Model : https://drive.google.com/open?id=1qzMCAbh_tW3SjMMVSPS4Ikt6hDnGfhEN

Categorical Crossentropy Loss : 0.58


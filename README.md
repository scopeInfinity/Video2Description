# Video Captioning
Generate caption for the given video clip

Branch : [VideoCaption](https://github.com/scopeInfinity/Video2Description/tree/VideoCaption)

Status : Ongoing

### Setup

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
*/path/to/data_dir/glove/glove.6B.100d.txt* | https://nlp.stanford.edu/projects/glove/

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

Commit | CIDEr | Bleu_4 | Bleu_3 | Bleu_2 | Bleu_1 | ROUGE_L | METEOR | Model Filename
--- | --- | --- | --- | --- | --- | --- | --- | ---
56e5a31 | 0.0903 | 0.2190 | 0.3276 | 0.4520 | 0.6043 | 0.4438 | 0.1647 | res_mcnn_rand_b100_s70_model.dat_4698_loss_1.2381912469863892 



# Image Captioning
Generate caption for the given images

Branch : [onehot_gen](https://github.com/scopeInfinity/Video2Description/tree/onehot_gen)

Commit : [898f15778d40b67f333df0a0e744a4af0b04b16c](https://github.com/scopeInfinity/Video2Description/commit/898f15778d40b67f333df0a0e744a4af0b04b16c)

Trained Model : https://drive.google.com/open?id=1qzMCAbh_tW3SjMMVSPS4Ikt6hDnGfhEN

Categorical Crossentropy Loss : 0.58


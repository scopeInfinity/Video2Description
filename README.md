# Video Captioning
Generate caption for the given video clip

Branch : [VideoCaption](https://github.com/scopeInfinity/Video2Description/tree/VideoCaption)

Status : Ongoing

### Setup

* Setup anaconda environment, use `environment.yml`
  * Install keras with tensorflow backend.
* Install OpenCV with ffmpeg support
  * Configure, build and install ffmpeg from source with shared libraries 
```bash
git clone 'https://github.com/FFmpeg/FFmpeg.git'
cd FFmpeg
./configure --prefix=~/V2D_local/ --enable-shared
make
make install
```
  * https://github.com/menpo/conda-opencv


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
  
  
  


# Image Captioning
Generate caption for the given images

Branch : [onehot_gen](https://github.com/scopeInfinity/Video2Description/tree/onehot_gen)

Commit : [898f15778d40b67f333df0a0e744a4af0b04b16c](https://github.com/scopeInfinity/Video2Description/commit/898f15778d40b67f333df0a0e744a4af0b04b16c)

Trained Model : https://drive.google.com/open?id=1qzMCAbh_tW3SjMMVSPS4Ikt6hDnGfhEN

Categorical Crossentropy Loss : 0.58


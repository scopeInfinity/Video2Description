FROM continuumio/miniconda:4.7.12-alpine
WORKDIR /code
ENV PATH /opt/conda/bin/:$PATH
COPY environment.yml environment.yml
RUN conda install -c anaconda git
RUN conda create -n env python=2.7
RUN conda install -c anaconda keras==2.0.8
RUN conda install -c conda-forge tensorflow==1.2.1
RUN echo "source activate env" > ~/.bashrc
RUN conda activate .
RUN git clone 'https://github.com/FFmpeg/FFmpeg.git'
# TODO can  we continue from inside the env
RUN cd FFmpeg
RUN echo "Building and Installing FFMpeg"
RUN ./configure --enable-shared
RUN make
RUN make install
RUN conda install opencv -c conda-forge
COPY root/ src/
RUN echo DONE!!!
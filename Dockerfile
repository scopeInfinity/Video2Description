FROM ubuntu:latest as v2d_env
RUN apt-get update
RUN apt-get install -y libsamplerate0 curl libsndfile1 pkg-config nasm wget zip
RUN useradd -m -s /bin/bash si
USER si

# Installing miniconda
RUN wget -N https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh -O /tmp/Miniconda2-latest-Linux-x86_64.sh
RUN bash /tmp/Miniconda2-latest-Linux-x86_64.sh -b
RUN rm /tmp/Miniconda2-latest-Linux-x86_64.sh
USER root
RUN ln -s /home/si/miniconda2/bin/conda /usr/bin/
USER si

# glove
# https://nlp.stanford.edu/projects/glove/
RUN mkdir /home/si/glove
WORKDIR /home/si/glove
RUN wget http://nlp.stanford.edu/data/glove.6B.zip && \
    unzip glove.6B.zip glove.6B.300d.txt && \
    rm glove.6B.zip

# ffmpeg build and install
WORKDIR /tmp
RUN wget https://github.com/FFmpeg/FFmpeg/archive/master.zip -O ffmpeg.zip
RUN unzip ffmpeg.zip
RUN rm ffmpeg.zip
WORKDIR /tmp/FFmpeg-master/
RUN ./configure --enable-shared
RUN make -j32
USER root
RUN make install
USER si
RUN rm -r /tmp/FFmpeg-master/

# coco-caption
WORKDIR /home/si
RUN wget -N 'https://github.com/tylin/coco-caption/archive/master.zip' -O coco.zip && \
    unzip coco.zip && \
    mv coco-caption-master coco-caption && \
    rm coco.zip

# Create conda environment
# Note: ffmpeg with --enable-shared should be before installing opencv
RUN mkdir /home/si/v2d/
WORKDIR /home/si/v2d/
COPY --chown=si:si environment.yml /home/si/v2d/
RUN conda env create -f environment.yml
RUN conda init bash

# Push V2D in the container
FROM v2d_env as v2d
COPY --chown=si:si root/ /home/si/v2d/src/
COPY --chown=si:si root/config_docker.json /home/si/v2d/src/config.json
WORKDIR /home/si/v2d/src

# Prepares cache
FROM v2d as v2d_deploy
COPY --chown=si:si models/ /home/si/v2d/models/
WORKDIR /home/si/v2d/models/
RUN wget -q -N 'https://github.com/scopeInfinity/Video2Description/releases/download/models/ResNet_D512L512_G128G64_D1024D0.20BN_BDLSTM1024_D0.2L1024DVS_model.dat_4987_loss_2.203_Cider0.342_Blue0.353_Rouge0.572_Meteor0.256'
RUN echo "Available Models:"
RUN ls -1 /home/si/v2d/models

WORKDIR /home/si/v2d/src/
RUN conda run -n V2D python parser.py server --init-only -m /home/si/v2d/models/ResNet_D512L512_G128G64_D1024D0.20BN_BDLSTM1024_D0.2L1024DVS_model.dat_4987_loss_2.203_Cider0.342_Blue0.353_Rouge0.572_Meteor0.256
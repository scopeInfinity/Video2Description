FROM ubuntu:latest as base
RUN apt-get update
RUN apt-get install -y libsndfile1 pkg-config nasm wget zip
RUN useradd -m -s /bin/bash si
USER si

# Installing miniconda
RUN wget -N https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh -O /tmp/Miniconda2-latest-Linux-x86_64.sh
RUN bash /tmp/Miniconda2-latest-Linux-x86_64.sh -b
RUN rm /tmp/Miniconda2-latest-Linux-x86_64.sh
USER root
RUN ln -s /home/si/miniconda2/bin/conda /usr/bin/
USER si

# ffmpeg build and install
RUN wget https://github.com/FFmpeg/FFmpeg/archive/master.zip -O /tmp/ffmpeg.zip
WORKDIR /tmp
RUN unzip ffmpeg.zip
WORKDIR /tmp/FFmpeg-master/
RUN ./configure --enable-shared
RUN make -j16
USER root
RUN make install
USER si
RUN rm -r /tmp/FFmpeg-master/
RUN rm -r /tmp/ffmpeg.zip

# Create conda environment
# Note: ffmpeg with --enable-shared is before installing opencv
RUN mkdir /home/si/v2d/
WORKDIR /home/si/v2d/
COPY --chown=si:si environment.yml /home/si/v2d/
RUN conda env create -f environment.yml
RUN conda init bash

# coco-caption
WORKDIR /home/si
RUN wget 'https://github.com/tylin/coco-caption/archive/master.zip' -O coco.zip
RUN unzip coco.zip
RUN mv coco-caption-master coco-caption
RUN rm coco.zip

# glove
RUN mkdir /home/si/glove
WORKDIR /home/si/glove
RUN wget http://nlp.stanford.edu/data/glove.6B.zip
RUN unzip glove.6B.zip glove.6B.300d.txt
RUN rm glove.6B.zip

# Push V2D in the container
FROM base
COPY --chown=si:si root/ /home/si/v2d/src/
COPY --chown=si:si models/ /home/si/v2d/models/
COPY --chown=si:si root/config_docker.json /home/si/v2d/src/config.json

RUN echo "Available Models:"
RUN ls -1 /home/si/v2d/models

# Turning up
WORKDIR /home/si/v2d/src
ENTRYPOINT conda run -n V2D python parser.py server -s -m /project/v2d/models/ResNet_D512L512_G128G64_D1024D0.20BN_BDLSTM1024_D0.2L1024DVS_model.dat_4987_loss_2.203_Cider0.342_Blue0.353_Rouge0.572_Meteor0.256
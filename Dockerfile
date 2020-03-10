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
RUN wget -N 'https://github.com/tylin/coco-caption/archive/master.zip' -O coco.zip
RUN unzip coco.zip
RUN mv coco-caption-master coco-caption
RUN rm coco.zip

# glove
# https://nlp.stanford.edu/projects/glove/
RUN mkdir /home/si/glove
WORKDIR /home/si/glove
RUN wget -N https://docs.google.com/uc?export=download&id=1NzOLm3mT0gJk0Y3IUnWpQFQFDL3zDYj0 -O glove.6B.300d.txt

# models
COPY --chown=si:si models/ /home/si/v2d/models/
RUN wget -N https://docs.google.com/uc?export=download&id=1aNXsT64tsza8vCqtoIZ4maQZ8Wbib6e0 -O ResNet_D512L512_G128G64_D1024D0.20BN_BDLSTM1024_D0.2L1024DVS_model.dat_4987_loss_2.203_Cider0.342_Blue0.353_Rouge0.572_Meteor0.256
RUN echo "Available Models:"
RUN ls -1 /home/si/v2d/models

# Push V2D in the container
FROM base
COPY --chown=si:si root/ /home/si/v2d/src/
COPY --chown=si:si root/config_docker.json /home/si/v2d/src/config.json

# Turning up
WORKDIR /home/si/v2d/src
ENTRYPOINT conda run -n V2D python parser.py server -s -m /project/v2d/models/ResNet_D512L512_G128G64_D1024D0.20BN_BDLSTM1024_D0.2L1024DVS_model.dat_4987_loss_2.203_Cider0.342_Blue0.353_Rouge0.572_Meteor0.256

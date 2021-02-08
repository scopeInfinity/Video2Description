FROM ubuntu:xenial as my_base
RUN apt-get update
RUN apt-get install -y libsamplerate0 curl libsndfile1 pkg-config nasm wget zip

FROM my_base as ffmpeg_builder
WORKDIR /tmp
RUN wget https://github.com/FFmpeg/FFmpeg/archive/master.zip -O ffmpeg.zip
RUN unzip ffmpeg.zip
RUN rm ffmpeg.zip
WORKDIR /tmp/FFmpeg-master/
RUN ./configure --enable-shared
RUN make -j32


FROM my_base as glove_builder
WORKDIR /tmp
# https://nlp.stanford.edu/projects/glove/
RUN wget http://nlp.stanford.edu/data/glove.6B.zip && \
    unzip glove.6B.zip glove.6B.300d.txt && \
    rm glove.6B.zip


FROM my_base as deploy
# FROM conda/miniconda2
RUN apt-get update
RUN apt-get install -y libsamplerate0 curl libsndfile1 pkg-config nasm wget zip
RUN useradd -m -s /bin/bash si
RUN mkdir /var/log/v2d
RUN chown si:si /var/log/v2d
RUN chmod 700 /var/log/v2d
USER si

# Installing miniconda
RUN wget -N https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh -O /tmp/Miniconda2-latest-Linux-x86_64.sh
RUN bash /tmp/Miniconda2-latest-Linux-x86_64.sh -b
RUN rm /tmp/Miniconda2-latest-Linux-x86_64.sh
USER root
RUN ln -s /home/si/miniconda2/bin/conda /usr/bin/
USER si

# glove
RUN mkdir -p /home/si/v2d/dataset
WORKDIR /home/si/v2d/dataset
COPY --from=glove_builder /tmp/glove.6B.300d.txt /home/si/v2d/dataset/glove.6B.300d.txt

# ffmpeg build and install
COPY --from=ffmpeg_builder /tmp/FFmpeg-master/ /tmp/FFmpeg-master/
WORKDIR /tmp/FFmpeg-master/
USER root
RUN make install
USER si
RUN echo 'export LD_LIBRARY_PATH=/usr/local/lib' >> /home/si/.bashrc

# coco-caption
WORKDIR /home/si
RUN wget -N 'https://github.com/tylin/coco-caption/archive/master.zip' -O coco.zip && \
    unzip coco.zip && \
    mv coco-caption-master coco-caption && \
    rm coco.zip

# Create conda environment
# Note: ffmpeg with --enable-shared should be before installing opencv
WORKDIR /home/si/v2d/
COPY --chown=si:si environment.yml /home/si/v2d/
RUN conda env create -f environment.yml
RUN conda init bash
RUN echo "conda activate V2D" >> /home/si/.bashrc

# Prepare basic files
ENV V2D_CONFIG_FILE=config_docker.json
RUN mkdir -p /home/si/v2d/dataset
RUN mkdir -p /home/si/v2d/dataset_cache
RUN mkdir -p /home/si/v2d/models
RUN mkdir -p /tmp/v2d/app/uploads
COPY --chown=si:si dataset/videodatainfo_2017.json /home/si/v2d/dataset/
COPY --chown=si:si dataset/test_videodatainfo_2017.json /home/si/v2d/dataset/
COPY --chown=si:si src/ /home/si/v2d/src/
WORKDIR /home/si/v2d/src

# Prepares cache for pretrained model
COPY --chown=si:si models/ /home/si/v2d/models/
WORKDIR /home/si/v2d/models/
RUN wget -q -N 'https://github.com/scopeInfinity/Video2Description/releases/download/models/ResNet_D512L512_G128G64_D1024D0.20BN_BDGRU1024_D0.2L1024DVS_model.dat_4983_loss_2.350_Cider0.355_Blue0.353_Rouge0.571_Meteor0.247_TOTAL_1.558_BEST'
RUN echo "Available Models:"
RUN ls -1 /home/si/v2d/models

WORKDIR /home/si/v2d/src/
RUN conda run -n V2D python -m backend.parser server --init-only -m /home/si/v2d/models/ResNet_D512L512_G128G64_D1024D0.20BN_BDGRU1024_D0.2L1024DVS_model.dat_4983_loss_2.350_Cider0.355_Blue0.353_Rouge0.571_Meteor0.247_TOTAL_1.558_BEST

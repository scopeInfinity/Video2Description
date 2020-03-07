FROM continuumio/miniconda:4.7.12-alpine as base1
COPY --chown=anaconda:anaconda environment.yml /project/v2d/
WORKDIR /project/v2d/
ENV PATH /opt/conda/bin/:$PATH
RUN conda env create -f environment.yml
RUN ["conda", "install", "-n", "V2D", "-c", "menpo", "opencv"]

# FROM base1 as base2
# COPY --chown=anaconda:anaconda FFmpeg/ /project/ffmpeg/
# WORKDIR /project/ffmpeg/
# RUN ["conda", "run", "-n", "V2D", "/bin/sh", "-c", "./configure", "--enable-shared"]
# RUN ["conda", "run", "-n", "V2D", "/bin/sh", "-c", "make", "-j4"]
# RUN ["conda", "run", "-n", "V2D", "/bin/sh", "-c", "make", "install"]
# RUN ["conda", "run", "-n", "V2D", "/bin/sh", "-c", "make", "install"]

FROM base1
WORKDIR /project/v2d/src/
COPY --chown=anaconda:anaconda root/ /project/v2d/src/
COPY --chown=anaconda:anaconda models/ /project/v2d/models/
COPY --chown=anaconda:anaconda root/config_docker.json /project/v2d/src/config.json
RUN ls -l /project/v2d/models/
ENTRYPOINT ["conda", "run", "-n", "V2D", "python", "parser.py", "server", "-s", "-m" , "/project/v2d/models/CAttention_ResNet_D512L512_G128G64_D1024D0.20BN_BDGRU1024_D0.2L1024DVS_model.dat_4990_loss_2.484_Cider0.360_Blue0.369_Rouge0.580_Meteor0.256.256__good"]

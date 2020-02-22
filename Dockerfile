FROM continuumio/miniconda:4.7.12-alpine as base1
COPY --chown=anaconda:anaconda root/ /project/v2d/src/
WORKDIR /project/v2d/src/
ENV PATH /opt/conda/bin/:$PATH
RUN conda env create -f environment.yml

FROM base1 as base2
COPY --chown=anaconda:anaconda FFmpeg/ /project/ffmpeg/
WORKDIR /project/ffmpeg/
RUN ./configure --enable-shared
RUN ./make -j4
RUN make install

# skipping base2
FROM base1
WORKDIR /project/v2d/src/
SHELL ["conda", "run", "-n", "V2D"]
RUN echo "Models found:"
RUN ls -l /project/v2d/models/
ENTRYPOINT ["python", "parser.py", "server", "-s", "-m" , "/project/v2d/models/CAttention_ResNet_D512L512_G128G64_D1024D0.20BN_BDGRU1024_D0.2L1024DVS_model.dat_4990_loss_2.484_Cider0.360_Blue0.369_Rouge0.580_Meteor0.256.256__good"]
RUN echo DONE!!!
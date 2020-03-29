FROM python:2 as frontend
RUN pip install enum34 flask waitress

RUN useradd -m -s /bin/bash si
RUN mkdir -p /home/si/v2d/uploads
RUN chown si:si /home/si/v2d/uploads
USER si

# Prepare basic files
ENV V2D_CONFIG_FILE=config_docker.json
RUN mkdir -p /tmp/v2d/app/uploads
COPY --chown=si:si src/frontend /home/si/v2d/src/frontend/
COPY --chown=si:si src/common /home/si/v2d/src/common/
COPY --chown=si:si src/*.json /home/si/v2d/src/
COPY --chown=si:si src/__init__.py /home/si/v2d/src/__init__.py
WORKDIR /home/si/v2d/src
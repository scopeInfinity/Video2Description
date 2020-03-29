FROM python:2 as frontend
RUN pip install enum34 flask waitress

RUN useradd -m -s /bin/bash si
RUN mkdir /var/log/v2d
RUN chown si:si /var/log/v2d
RUN chmod 700 /var/log/v2d
USER si

# Prepare basic files
ENV V2D_CONFIG_FILE=config_docker.json
RUN mkdir -p /tmp/v2d/app/uploads
COPY --chown=si:si src/frontend /home/si/v2d/src/frontend/
COPY --chown=si:si src/common /home/si/v2d/src/common/
COPY --chown=si:si src/*.json /home/si/v2d/src/
COPY --chown=si:si src/__init__.py /home/si/v2d/src/__init__.py
WORKDIR /home/si/v2d/src
FROM  tensorflow/tensorflow:latest-gpu

LABEL maintainer=achillesrasquinha@gmail.com

ENV GIXPERT_PATH=/usr/local/src/gixpert

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        git \
        curl \
    && mkdir -p $GIXPERT_PATH

COPY . $GIXPERT_PATH
COPY ./docker/entrypoint.sh /entrypoint.sh

WORKDIR $GIXPERT_PATH

RUN pip install -r ./requirements.txt && \
    python setup.py install

ENTRYPOINT ["/entrypoint.sh"]

CMD ["gixpert"]
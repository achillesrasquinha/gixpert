FROM  nvidia/cuda:11.4.1-cudnn8-devel-ubuntu18.04

LABEL maintainer=achillesrasquinha@gmail.com

ENV GIXPERT_PATH=/usr/local/src/gixpert

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        git \
        curl \
        python3-dev \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py \
    && mkdir -p $GIXPERT_PATH

COPY . $GIXPERT_PATH
COPY ./docker/entrypoint.sh /entrypoint.sh

WORKDIR $GIXPERT_PATH

RUN pip install -r ./requirements.txt && \
    python3 setup.py install

ENTRYPOINT ["/entrypoint.sh"]

CMD ["gixpert"]
FROM  nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

LABEL maintainer=achillesrasquinha@gmail.com

ENV GIXPERT_PATH=/usr/local/src/gixpert

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        git \
        curl \
        python3-dev \
    && ln -s python3 python \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python get-pip.py \
    && rm get-pip.py \
    && mkdir -p $GIXPERT_PATH

COPY . $GIXPERT_PATH
COPY ./docker/entrypoint.sh /entrypoint.sh

WORKDIR $GIXPERT_PATH

RUN pip install -r ./requirements.txt && \
    python setup.py install

ENTRYPOINT ["/entrypoint.sh"]

CMD ["gixpert"]
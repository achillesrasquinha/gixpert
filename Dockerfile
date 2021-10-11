

FROM  python:3.7-alpine

LABEL maintainer=achillesrasquinha@gmail.com

ENV GIXPERT_PATH=/usr/local/src/gixpert

RUN apk add --no-cache \
        bash \
        git \
    && mkdir -p $GIXPERT_PATH

COPY . $GIXPERT_PATH
COPY ./docker/entrypoint.sh /entrypoint.sh

WORKDIR $GIXPERT_PATH

RUN pip install -r ./requirements.txt && \
    python setup.py install

ENTRYPOINT ["/entrypoint.sh"]

CMD ["gixpert"]
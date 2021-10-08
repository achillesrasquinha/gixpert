

FROM  python:3.7-alpine

LABEL maintainer=achillesrasquinha@gmail.com

ENV GIXPERT_PATH=/usr/local/src/gixpert

RUN apk add --no-cache \
        bash \
        git \
    && mkdir -p $GIXPERT_PATH

COPY . $GIXPERT_PATH
COPY ./docker/entrypoint.sh /entrypoint.sh

RUN pip install $GIXPERT_PATH

WORKDIR $GIXPERT_PATH

ENTRYPOINT ["/entrypoint.sh"]

CMD ["gixpert"]
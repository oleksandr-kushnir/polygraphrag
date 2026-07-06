FROM pgvector/pgvector:pg16

RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-server-dev-16 \
    flex \
    bison \
    curl \
    && rm -rf /var/lib/apt/lists/*

ARG AGE_VERSION=1.5.0
RUN curl -fL \
    "https://github.com/apache/age/releases/download/PG16%2Fv${AGE_VERSION}-rc0/apache-age-${AGE_VERSION}-src.tar.gz" \
    -o /tmp/age.tar.gz \
    && mkdir -p /tmp/age-src \
    && tar -xzf /tmp/age.tar.gz -C /tmp/age-src --strip-components=1 \
    && cd /tmp/age-src \
    && make install \
    && rm -rf /tmp/age.tar.gz /tmp/age-src \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

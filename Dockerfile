# Dockerfile

FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-1

WORKDIR /root

ENV PROJECT_ID=alekseyv-scalableai-dev
ENV GOOGLE_APPLICATION_CREDENTIALS=/root/service_account_key.json
COPY service_account_key.json /root/

RUN pip install nbdev

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
RUN git clone https://github.com/vlasenkoalexey/gcp_runner
RUN pip install -e gcp_runner

RUN mkdir /root/models
RUN mkdir /root/criteo_nbdev
COPY criteo_nbdev/* /root/criteo_nbdev/
# Dockerfile

FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-1

WORKDIR /root

RUN git clone https://github.com/vlasenkoalexey/gcp_runner
RUN pip install -e gcp_runner

RUN mkdir /root/models
RUN mkdir /root/criteo_nbdev
COPY criteo_nbdev/* /root/criteo_nbdev/
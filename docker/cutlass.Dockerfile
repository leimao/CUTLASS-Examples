FROM nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04

ARG CUTLASS_VERSION=3.5.1
ARG CMAKE_VERSION=3.28.0
ARG NUM_JOBS=8

ENV DEBIAN_FRONTEND=noninteractive

# Install package dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        locales \
        locales-all \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        wget \
        git && \
    apt-get clean

# System locale
# Important for UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# Install CMake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    bash cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm -rf /tmp/*

RUN cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip && \
    pip install --upgrade pip setuptools wheel

# Download CUTLASS
RUN cd /tmp && \
    wget https://github.com/NVIDIA/cutlass/archive/refs/tags/v${CUTLASS_VERSION}.tar.gz && \
    tar -xvf v${CUTLASS_VERSION}.tar.gz && \
    # CUTLASS is header-only
    cp -r cutlass-${CUTLASS_VERSION}/include/* /usr/local/include/ && \
    cp -r /tmp/cutlass-${CUTLASS_VERSION}/tools/util/include/cutlass/* /usr/local/include/cutlass/ && \
    rm -rf /tmp/*
FROM nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04

ARG CMAKE_VERSION=4.2.0
ARG GOOGLETEST_VERSION=1.17.0
ARG NUM_JOBS=8

ARG PYTHON_VENV_PATH=/python/venv

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
        python3-full \
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

# Install GoogleTest
RUN cd /tmp && \
    wget https://github.com/google/googletest/archive/refs/tags/v${GOOGLETEST_VERSION}.tar.gz && \
    tar -xzf v${GOOGLETEST_VERSION}.tar.gz && \
    cd googletest-${GOOGLETEST_VERSION} && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j${NUM_JOBS} && \
    make install && \
    rm -rf /tmp/*

# Install QT6 and its dependencies for Nsight Compute GUI
# https://leimao.github.io/blog/Docker-Nsight-Compute/
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        dbus \
        fontconfig \
        gnupg \
        libfreetype6 \
        libglib2.0-0 \
        libnss3 \
        libsqlite3-0 \
        libx11-xcb1 \
        libxcb-glx0 \
        libxcb-xkb1 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxi6 \
        libxml2 \
        libxrandr2 \
        libxrender1 \
        libxtst6 \
        libxkbfile-dev \
        openssh-client \
        xcb \
        xkb-data \
        libxcb-cursor0 \
        qt6-base-dev && \
    apt-get clean

RUN mkdir -p ${PYTHON_VENV_PATH} && \
    python3 -m venv ${PYTHON_VENV_PATH}

ENV PATH=${PYTHON_VENV_PATH}/bin:$PATH

RUN cd ${PYTHON_VENV_PATH}/bin && \
    pip install --upgrade pip setuptools wheel

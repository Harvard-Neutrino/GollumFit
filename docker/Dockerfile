# GollumFit container build (NeoDANSA fork).
# Build from the REPOSITORY ROOT so the build context contains the source +
# vendored submodules:
#   docker build -f docker/Dockerfile -t gollumfit:neodansa .
#
# This Dockerfile was updated from the upstream (stale) version which referenced
# the old `Fit/` + `sources/` layout. Current layout: src/ + include/ + python/
# + vendor/ submodules + root Makefile/configure.

FROM ubuntu:20.04

LABEL maintainer="NeoDANSA"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    gfortran \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    git \
    openssh-client \
    pkg-config \
    libgsl-dev \
    libboost-all-dev \
    libboost-python-dev \
    libhdf5-dev \
    hdf5-tools \
    cmake \
    libcfitsio-dev \
    libblas-dev \
    liblapack-dev \
    libsuitesparse-dev \
    python3-sphinx \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && pip3 install --upgrade pip \
    && pip3 install numpy h5py scipy pandas matplotlib pybind11

ENV CXX=g++
ENV CC=gcc
ENV GOLLUMSPACE=/gollumspace
ENV GOLLUMBUILDPATH=/usr/local
ENV PREFIX=/usr/local
ENV SROOT=/usr
ENV PATH=$GOLLUMBUILDPATH/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:/usr/local/lib64:$LD_LIBRARY_PATH
ENV C_INCLUDE_PATH=/usr/local/include:/usr/include/hdf5/serial:$C_INCLUDE_PATH
ENV CPLUS_INCLUDE_PATH=/usr/include/hdf5/serial:/usr/local/include:$CPLUS_INCLUDE_PATH
ENV PKG_CONFIG_PATH=/usr/lib/pkgconfig:/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
ENV HDF5_DISABLE_VERSION_CHECK=1

RUN mkdir -p $GOLLUMSPACE $GOLLUMSPACE/lib \
    && mkdir -p $PREFIX/bin $PREFIX/include $PREFIX/lib $PREFIX/lib64

# ---- Dependencies FIRST (cache-friendly: independent of GollumFit source, so
#      editing src/include/python does NOT trigger a full dependency rebuild) ----
COPY vendor $GOLLUMSPACE/GollumFit/vendor

WORKDIR $GOLLUMSPACE/GollumFit/vendor/SQuIDS
RUN ./configure --prefix=/usr && make -j2 && make install

# HDF5 libs live in an arch-specific serial/ subdir on Ubuntu 20.04; expose them
# to the linker so `-lhdf5`/`-lhdf5_hl` resolve in all subsequent builds.
RUN ARCH=$(uname -m) \
    && ln -sf /usr/lib/${ARCH}-linux-gnu/hdf5/serial/libhdf5.so /usr/lib/libhdf5.so \
    && ln -sf /usr/lib/${ARCH}-linux-gnu/hdf5/serial/libhdf5_hl.so /usr/lib/libhdf5_hl.so

WORKDIR $GOLLUMSPACE/GollumFit/vendor/nuSQuIDS
RUN ./configure --prefix=/usr && make -j2 && make install

WORKDIR $GOLLUMSPACE/GollumFit/vendor/PhysTools
RUN mkdir -p build build/lbfgsb \
    && ARCH=$(uname -m) \
    && ./configure --prefix=/usr --with-boost-incdir=/usr/include/ --with-boost-libdir=/usr/lib/${ARCH}-linux-gnu \
    && make -j2 \
    && make install

WORKDIR $GOLLUMSPACE
RUN git clone https://github.com/cnweaver/photospline.git photospline \
    && mkdir -p photospline/build
WORKDIR $GOLLUMSPACE/photospline/build
RUN cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr ../ \
    && make -j2 \
    && make install

WORKDIR $GOLLUMSPACE/GollumFit/vendor/LeptonWeighter
RUN ARCH=$(uname -m) \
    && ./configure --prefix=/usr --with-boost-incdir=/usr/include/ --with-boost-libdir=/usr/lib/${ARCH}-linux-gnu \
    && make -j2 \
    && make install

# ---- Diver (differential-evolution optimizer; required by the Python bindings) ----
WORKDIR $GOLLUMSPACE
RUN git clone https://github.com/diveropt/Diver.git Diver \
    && cd Diver \
    && make libdiver.so \
    && find . -name 'libdiver.so' -exec cp {} /usr/lib/ \; \
    && find . -name 'diver.hpp' -exec cp {} /usr/include/ \;
ENV DIVER_DIR=$GOLLUMSPACE/Diver

# ---- GollumFit source LAST (edits here rebuild only the lib + bindings) ----
COPY configure Makefile $GOLLUMSPACE/GollumFit/
COPY src $GOLLUMSPACE/GollumFit/src
COPY include $GOLLUMSPACE/GollumFit/include
COPY python $GOLLUMSPACE/GollumFit/python

WORKDIR $GOLLUMSPACE/GollumFit
RUN ./configure && make -j2 && make install
WORKDIR $GOLLUMSPACE/GollumFit/python
RUN python3 setup.py install --prefix=$PREFIX \
    && python3 -c "import GollumFitPy; print('GollumFitPy import OK')"

WORKDIR /workspace

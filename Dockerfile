FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y git nano cmake wget libglew-dev g++ build-essential vim \
    libgtk2.0-dev pkg-config libgl1-mesa-glx libgl1-mesa-dri git wget \
    libblas-dev liblapack-dev && \
    mkdir /app && mkdir /dpds && cd /dpds

RUN cd /dpds && \
    wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && \
    tar -xzf eigen-3.4.0.tar.gz && \
    rm eigen-3.4.0.tar.gz && \
    cd eigen-3.4.0 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make install

RUN cd /dpds && \
    wget https://github.com/stevenlovegrove/Pangolin/archive/refs/tags/v0.8.tar.gz && \
    tar -xzf v0.8.tar.gz && \
    rm v0.8.tar.gz && \
    cd Pangolin-0.8/ && \
    mkdir build && cd build && \
    cmake ..  -DCMAKE_INCLUDE_PATH=/usr/local/include/eigen3 && \
    cmake --build .

RUN cd /dpds && \
    git clone https://github.com/opencv/opencv.git && \
    git clone https://github.com/opencv/opencv_contrib.git && \
    cd opencv/ && git checkout 4.8.0 && \
    cd ../opencv_contrib/ && git checkout 4.8.0 && \
    cd /dpds/opencv && \
    mkdir build && cd build && \
    cmake -D WITH_GTK=ON -D WITH_CUDA=ON -D OPENCV_EXTRA_MODULES_PATH=/dpds/opencv_contrib/modules .. && \
    make -j2 && \
    make install && \
    ldconfig

RUN cd /dpds && \
    git clone https://github.com/nubificus/orb-slam2_cuda.git 

RUN cd /dpds/orb-slam2_cuda/Clustering/ORB-SLAM2/ && \
    chmod +x build.sh &&\
    ./build.sh

RUN cd /dpds/orb-slam2_cuda/NoClustering/ORB-SLAM2/ && \
    chmod +x build.sh &&\
    ./build.sh

RUN apt-get update && apt-get install -y pip && \
    pip install evo

# CMD ["./executable"]
# CMD ["/bin/sh", "-ec", "while :; do echo '.'; sleep 5 ; done"]

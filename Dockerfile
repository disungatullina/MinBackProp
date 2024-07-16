FROM ubuntu:20.04

# Updates
RUN apt-get update --fix-missing -y
RUN apt-get upgrade -y

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y wget unzip curl zlib1g zlib1g-dev gcc g++ build-essential libcurl4-openssl-dev git python3 python3-setuptools python3-dev python3-pip libblas-dev liblapack-dev ffmpeg libsm6 libxext6 libgflags-dev libopencv-dev libeigen3-dev vim python3-h5py pkg-config libhdf5-dev

RUN apt-get upgrade -y cmake qt5-default

RUN python3 -m pip install --upgrade pip setuptools wheel cmake
RUN python3 -m pip install numpy matplotlib tqdm kornia kornia_moons tensorboardX scikit-learn
RUN python3 -m pip install torch==1.12.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

WORKDIR /var/www/

RUN git clone https://github.com/gflags/gflags.git
WORKDIR gflags
RUN mkdir build_ 
WORKDIR build_ 
RUN cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_CXX_STANDARD=17
RUN make -j2
RUN make -j2 install
WORKDIR /var/www/

RUN git clone https://github.com/google/glog.git
WORKDIR glog
RUN mkdir build
WORKDIR build 
RUN cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_CXX_STANDARD=17
RUN make -j2
RUN make -j2 install
WORKDIR /var/www/

RUN git clone https://github.com/opencv/opencv.git
WORKDIR opencv
RUN git checkout 3.4.2
RUN mkdir build 
WORKDIR build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D BUILD_EXAMPLES=ON ..
RUN make -j4
RUN make -j4 install
WORKDIR /var/www/
RUN sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
RUN ldconfig

WORKDIR /var/www/
RUN git clone https://github.com/disungatullina/magsac.git --recursive
WORKDIR magsac
RUN mkdir build 
WORKDIR build 
RUN cmake ..
RUN make
WORKDIR ..
RUN python3 ./setup.py install
RUN python3 -m pip install .
WORKDIR ..

RUN apt-get install qt5-default -y
RUN python3 -m pip install --upgrade h5py

RUN git clone https://github.com/disungatullina/MinBackProp.git --recurse-submodules -j8
WORKDIR MinBackProp
RUN wget https://cmp.felk.cvut.cz/~weitong/nabla_ransac/diff_ransac_data.zip
RUN unzip diff_ransac_data.zip
RUN rm diff_ransac_data.zip
RUN find /usr/local/lib/ -type f -name "cv2*.so"
RUN ln -s /usr/local/lib/python3.8/site-packages/cv2.cpython-38-aarch64-linux-gnu.so cv2.so
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
CMD [ "/bin/bash", "-c", "export"]
CMD [ "/bin/bash"]
WORKDIR /var/www/MinBackProp

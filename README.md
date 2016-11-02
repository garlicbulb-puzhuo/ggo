
### Benchmark

@(Playground)
### Install Cloudera docker image
https://www.cloudera.com/documentation/enterprise/5-6-x/topics/quickstart_docker_container.htm
### Install Docker
### Install Python 2.7
There are 2 ways to install python 2.7 on your local mac.
#### Install Python 2.7 with yum
Install python 2.7
```
$ yum install -y centos-release-SCL
$ yum install -y python27
```
Enable python 2.7
```
$ source /opt/rh/python27/enable
```
#### Install Python 2.7  from scratch
Follow steps on this tutorial: https://github.com/h2oai/h2o-2/wiki/installing-python-2.7-on-centos-6.3.-follow-this-sequence-exactly-for-centos-machine-only

One caveat is that you need to configure python building environment as below
```
./configure --enable-shared \
            --prefix=/usr/local \
            LDFLAGS="-Wl,--rpath=/usr/local/lib"
```
### Install opencv
#### Install opencv on Mac OS
http://www.mobileway.net/2015/02/14/install-opencv-for-python-on-mac-os-x/
#### Install opencv on CentOs
##### Install opencv on CentOs - NOT WORKING
```
$ sudo yum install opencv
```
##### Install opencv on CentOs - Build from Source
http://techieroop.com/install-opencv-in-centos/
Install dependencies
```
$ yum install cmake
$ yum install python-devel nump
$ yum install gcc gcc-c++
```
```
$ yum install gtk2-devel
$ yum install libdc1394-devel
$ yum install libv4l-devel
$ yum install ffmpeg-devel
$ yum install gstreamer-plugins-base-devel
```
```
$ yum install libpng-devel libjpeg-turbo-devel jasper-devel openexr-devel libtiff-devel libwebp-devel
```
- Download opencv from github
```
$ yum install git
$ mkdir opencv-build
$ cd opencv-build
$ git clone https://github.com/opencv/opencv.git
$ cd opencv
$ git checkout tags/3.1.0
```
Create a new directory build to compile opencv from source.
```
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
$ make
$ sudo make install
```
Move opencv module from in defult python path
```
$ cp /usr/local/lib/python2.7/site-packages/cv2.so /usr/lib/python2.7/site-packages
```
Verify Installation
```
$ python
>>> import cv2
>>> print cv2.__version__
```
### Deep Learning Packages
Install Theano
http://deeplearning.net/software/theano/install.html
Install python package locally, for example, elephas
```
$ python setup.py install
```
### MISC
Impersonate cloudera user
```
$ sudo -i -u cloudera
```
Upgrade python packages installed by pip
```
$ pip install pip-review
$ pip-review --local --interactive
```


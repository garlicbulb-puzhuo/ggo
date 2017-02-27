This document explains how to install deep learning libraries on aws or mac.

# Tensorflow
Installing tensorflow is quite difficult because it requires installation from source. Also, it depends on CUDA versions. And, it takes several hours. 
 
This installation works as of Feb 2017, but it may be outdated very soon. We will update this document as needed.
 
NOTE: we may use existing public AMIs (e.g., [this one](https://github.com/ritchieng/tensorflow-aws-ami)) in the future as it's painful to install tensorflow.

- Install CUDA Toolkit 8.0. This may require a reboot.
    ```
    mkdir ~/setup
    cd ~/setup
    wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64-deb
    sudo dpkg -i cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64-deb
    sudo apt-get update
    sudo apt-get install cuda
    # sudo shutdown -r now
    ```

- Install cuDNN 5.1. Note that, you will have to download the file and install manually. This may require a reboot.
    ```
    cd ~/setup
    # this doesn’t work. Download manually and upload it to aws
    # wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod_20161129/8.0/cudnn-8.0-linux-x64-v5.1-tgz
    # wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod_20161129/8.0/libcudnn5-dev_5.1.10-1+cuda8.0_amd64-deb
    sudo tar -xvf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
    # sudo shutdown -r now
    ```

- Install nvidia tools. This may require a reboot.
    ```
    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt-get update
    sudo apt-get install nvidia-367
    # sudo shutdown -r now
    ```

- Test nvidia installation.
    ```
    nvidia-smi
    ```

- Install bazel. This is to compile tensorflow.
    ```
    sudo apt-get install software-properties-common swig
    sudo add-apt-repository ppa:webupd8team/java
    sudo apt-get update
    sudo apt-get install oracle-java8-installer
    echo "deb http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
    curl https://storage.googleapis.com/bazel-apt/doc/apt-key.pub.gpg | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install bazel
    ```

- Install tensorflow.
    ```
    cd ~/setup
    git clone https://github.com/tensorflow/tensorflow
    cd tensorflow
    TF_UNOFFICIAL_SETTING=1 ./configure
    # WARN: read this link: http://stackoverflow.com/questions/33651810/the-minimum-required-cuda-capability-is-3-5 … change the minimum cuda compute requirement to 3.0?
    # WARN: follow warnings in this link: https://alliseesolutions.wordpress.com/2016/09/08/install-gpu-tensorflow-from-sources-w-ubuntu-16-04-and-cuda-8-0-rc/
    bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    # Find the exact filename in /tmp/tensorflow_pkg/ and execute the following line.
    # pip install /tmp/tensorflow_pkg/tensorflow…
    ```

- Verify tensorflow installation
    ```
    cd ~
    git clone https://github.com/aymericdamien/TensorFlow-Examples
    cd ~/TensorFlow-Examples/examples/1_Introduction
    python helloworld.py
    # The following example will fail because it uses old tensorflow library.
    python basic_operations.py
    ```

# Theano, Keras
```
$ pip install theano
$ pip install keras
```
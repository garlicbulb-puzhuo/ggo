#!/usr/bin/env bash

cd /home/ubuntu/Developer/ggo

# added by Anaconda2 2.4.1 installer
export PATH="/home/ubuntu/anaconda2/bin:$PATH"
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/home/ubuntu/anaconda2/lib/only_python_so:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/ubuntu/cudnn-install/lib64:$LD_LIBRARY_PATH"
export CPATH="/home/ubuntu/cudnn-install/include:$CPATH"
export LIBRARY_PATH="/home/ubuntu/cudnn-install/lib64:$LIBRARY_PATH"


. /home/ubuntu/torch/install/bin/torch-activate

master_ec2_host=`cat ~/.master_ec2_host`
THEANO_FLAGS="floatX=float32,device=gpu" python -m scripts.ec2.ec2_launcher --dburl ${master_ec2_host}


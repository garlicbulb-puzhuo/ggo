#!/usr/bin/env bash

cd /home/ubuntu/Developer/ggo
master_ec2_host=`cat ~/.master_ec2_host`
THEANO_FLAGS="floatX=float32,device=gpu" python -m scripts.ec2.ec2_launcher --dburl ${master_ec2_host}


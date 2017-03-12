#!/usr/bin/env bash

WORKING_DIR=/home/ubuntu/Developer/ggo
. ${WORKING_DIR}/bin/app-env.sh

cd ${WORKING_DIR}
master_ec2_host=`cat ~/.master_ec2_host`
THEANO_FLAGS="floatX=float32,device=gpu" python -m scripts.ec2.ec2_launcher --dburl ${master_ec2_host}


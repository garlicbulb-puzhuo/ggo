#!/usr/bin/env bash

if curl -s http://169.254.169.254/latest/meta-data/spot/termination-time | grep -q .*T.*Z
then
    echo ec2 intance is about to be shut down
    terminate
fi

function terminate {
    master_ec2_host=`cat ~/master_ec2_host`
    cd /home/ubuntu/Developer/ggo
    python -m scripts.ec2.ec2_terminator --dburl ${master_ec2_host}
}

#!/usr/bin/env bash

WORKING_DIR=/home/ubuntu/Developer/ggo
. ${WORKING_DIR}/bin/app-env.sh

function schedule_task {
    if curl -s http://169.254.169.254/latest/meta-data/spot/termination-time | grep -q .*T.*Z
    then
        now=$(date +"%D %T")
        echo "$now ec2 instance is about to be shut down on ${HOSTNAME}"

        # Kill the training program if any
        TRAINING_PID=$(ps -ef | grep scripts.ec2.ec2_launcher | grep -v grep | awk '{print $2}')
        if [ -z ${TRAINING_PID} ]; then
            echo "$now no training program is running"
        else
            echo kill -9 ${TRAINING_PID}
            kill -9 ${TRAINING_PID}
        fi

        terminate
        exit
    fi
}


function terminate {
    now=$(date +"%D %T")
    master_ec2_host=`cat ~/.master_ec2_host`
    cd ${WORKING_DIR}

    echo "$now terminating tasks on ${HOSTNAME}"
    python -m scripts.ec2.ec2_terminator --dburl ${master_ec2_host}

    now=$(date +"%T")
    echo "$now finished terminating tasks"
}

while true; do
    schedule_task
    sleep 5
done

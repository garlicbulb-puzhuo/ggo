#!/usr/bin/env bash

cd /home/ubuntu/Developer/ggo
git pull --rebase

/home/ubuntu/Developer/ggo/bin/bootstrap_ec2.sh > ~/Developer/application_logs/bootstrap_ec2.log 2>&1
nohup /home/ubuntu/Developer/ggo/bin/terminate_ec2.sh > ~/Developer/application_logs/terminate_ec2.log 2>&1 &
nohup /home/ubuntu/Developer/ggo/bin/schedule_task_ec2.sh > ~/Developer/application_logs/schedule_task_ec2.log 2>&1 &

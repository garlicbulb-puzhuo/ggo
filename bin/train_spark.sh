#!/usr/bin/env bash

set -e
set -u

# current program name
progname="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"

working_dir=
training_file=

function usage {
    echo "usage: ${progname} -w directory -t directory [-h]"
    echo "  -w  Directory to store weights"
    echo "  -t  File of training data"
    echo "  [-h] Display usage"
    exit 1
}

while getopts ":t:w:h" opt; do
    case $opt in
        w)
            working_dir=$OPTARG
            working_dir_arg=1
            ;;
        t)
            training_file=$OPTARG
            training_file_arg=1
            ;;
        h)
            usage
            ;;
        \?)
            echo "${progname}: Invalid option: -$OPTARG"
            usage
            ;;
        :)
            echo "${progname}: Option -$OPTARG requires an argument"
            usage
            ;;
    esac
done

if [ $# -eq 0 ]; then
    echo "${progname}: No command line argument provided"
    usage
fi

if [ ${working_dir_arg} -eq 0 ]; then
    echo "${progname}: Missing working output directory argument"
    usage
fi

if [ ${training_file_arg} -eq 0 ]; then
    echo "${progname}: Missing training data directory argument"
    usage
fi

BASEDIR=$(dirname $0)/..
MAIN_TRAIN_SCRIPT=${BASEDIR}/scripts/train.py

cd ${working_dir}

echo "Start training"

exec_env="export SPARK_CONF_DIR=${SPARK_CONF_DIR}"

su -s /bin/bash ${SVC_USER} -c "$exec_env & spark-submit --num-executors 10 \
    --master yarn-client --driver-memory 10G --executor-memory 5G \
    --principal ${KERBEROS_PRINCIPAL} --keytab ${KERBEROS_KEYAB} --proxy-user ${USER} \
    --conf spark.akka.frameSize=1024 \
    --conf spark.executorEnv.HOME=${HOME} \
    ${MAIN_TRAIN_SCRIPT} \
    --train_imgs_path ${training_file} --train --train_mode spark --config_file ${BASEDIR}/config/config.ini
    > ${working_dir}/train.log 2>&1"

echo "Training done"

echo "Start pulling results"

regex="application_[0-9]+_[0-9]+"
application_id=$(grep -Ei " $regex " ${working_dir}/train.log | head -1 | grep -oEi $regex)

su -s /bin/bash ${SVC_USER} -c "hdfs dfs -cat /var/log/hadoop-yarn/apps/${USER}/logs/${application_id}/* \
    | ${BASEDIR}/application_log.sh > ${working_dir}/results.csv"


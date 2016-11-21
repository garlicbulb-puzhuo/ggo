#!/usr/bin/env bash

set -e
set -u

# current program name
progname="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"

working_dir=
training_file=
config_file=

function usage {
    echo "usage: ${progname} -w directory -t directory [-h]"
    echo "  -w  Directory to store weights"
    echo "  -t  File of training data"
    echo "  -c  Configuration file"
    echo "  [-h] Display usage"
    exit 1
}

while getopts ":w:t:c:h" opt; do
    case $opt in
        w)
            working_dir=$OPTARG
            working_dir_arg=1
            ;;
        t)
            training_file=$OPTARG
            training_file_arg=1
            ;;
        c)
            config_file=$OPTARG
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
CONFIGURATION_FILE=${config_file:=${BASEDIR}/config/config.ini}

cd ${working_dir}

echo "Start training"

exec_env="export SPARK_CONF_DIR=${SPARK_CONF_DIR}"

SPARK_COMMAND="$exec_env && spark-submit --num-executors 10 \
    --master yarn-client --driver-memory 10G --executor-memory 6G \
    --principal ${KERBEROS_PRINCIPAL} --keytab ${KERBEROS_KEYTAB} --proxy-user ${PROXY_USER} \
    --conf spark.akka.frameSize=1024 \
    --conf spark.network.timeout=600s \
    --conf spark.executorEnv.HOME=${SPARK_ENV_HOME} \
    ${MAIN_TRAIN_SCRIPT} \
    --train_imgs_path ${training_file} --train --train_mode spark --config_file ${CONFIGURATION_FILE} \
    >${working_dir}/train.log 2>&1"

echo "Spark Command: $SPARK_COMMAND"

eval ${SPARK_COMMAND}

echo "Training done"

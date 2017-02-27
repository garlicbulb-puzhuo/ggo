#!/bin/bash

# current program name
progname="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"

input_dir=
input_dir_arg=0

output_dir=
output_dir_arg=0

log_dir=
log_dir_arg=0

processor=""
model_file=""

function usage {
    echo "usage: ${progname} -p processor [-m model weights file] -i input_directory -o output_directory -l log_directory [-h]"
    echo "  -i  root directory for the partitioned input files"
    echo "  -o  root directory for the partitioned output files"
    echo "  -p  processor, valid values are [preprocess, predict]"
    echo "  -m  model file, this parameter is optional and needs to be set if processor is predict"
    echo "  -l  root directory for the application logs"
    echo "  [-h] Display usage"
    exit 1
}

while getopts ":i:o:h" opt; do
    case $opt in
        i)
            input_dir=$OPTARG
            input_dir_arg=1
            ;;
        o)
            output_dir=$OPTARG
            output_dir_arg=1
            ;;
        l)
            log_dir=$OPTARG
            log_dir_arg=1
            ;;
        p)
            if [[ "$OPTARG" != "preprocess" && "$OPTARG" != "predict" ]]; then
              echo "${progname}: $OPTARG is an invalid processor"
              usage
            fi
            processor=$OPTARG
            ;;
        m)
            model_file=$OPTARG
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

if [ ${input_dir_arg} -eq 0 ]; then
    echo "${progname}: Missing input directory argument"
    usage
fi

if [ ${output_dir_arg} -eq 0 ]; then
    echo "${progname}: Missing output directory argument"
    usage
fi

if [ ${log_dir_arg} -eq 0 ]; then
    echo "${progname}: Missing log directory argument"
    usage
fi

if [ ${processor} != "" ]; then
    echo "${progname}: Missing processor"
    usage
fi

if [ ${processor} == "predict" && ${model_file} == "" ]; then
    echo "${progname}: Missing model file if processor is predict"
    usage
fi

files=($(ls ${input_dir}))
len=${#files[*]}

i=0
while [ ${i} -lt ${len} ];
do
    input_batch_dir=${files[$i]}
    batch=$(basename ${input_batch_dir})
    mkdir -p ${output_dir}/{batch}
    mkdir -p ${log_dir}/${batch}

    if [ ${processor} == "preprocessor" ]; then
        echo python -m scripts.luna.watershed --input_dir ${input_batch_dir} --output_path ${output_dir}/${batch}
        nohup python -m scripts.luna.watershed --input_dir ${input_batch_dir} --output_path ${output_dir}/${batch} > ${log_dir}/${batch}/job.log 2>&1 &
    elif [ ${processor} == "predict" ]; then
        echo python -m scripts.kaggle.nodule_detection --model_id 2 --model_weights ${model_file} --input_path ${input_batch_dir} --output_path ${output_dir}/${batch}
        nohup python -m scripts.kaggle.nodule_detection --model_id 2 --model_weights ${model_file} --input_path ${input_batch_dir} --output_path ${output_dir}/${batch} > ${log_dir}/${batch}/job.log 2>&1 &
    fi
done

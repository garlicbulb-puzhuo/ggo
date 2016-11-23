#!/usr/bin/env bash

set -e
set -u

# current program name
progname="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"

working_dir=
working_dir_arg=0

function usage {
    echo "usage: ${progname} -w directory [-h]"
    echo "  -w  Working directory"
    echo "  [-h] Display usage"
    exit 1
}

while getopts ":w:h" opt; do
    case $opt in
        w)
            working_dir=$OPTARG
            working_dir_arg=1
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

echo "Start pulling results"

regex="application_[0-9]+_[0-9]+"
application_id=$(grep -Ei " $regex " ${working_dir}/train.log | head -1 | grep -oEi $regex)
echo "Found application ${application_id}"

hdfs dfs -cat /var/log/hadoop-yarn/apps/${USER}/logs/${application_id}/* \
    | grep -a "history and metadata values" \
    | awk -F: '{print $2}' \
    | sed -e 's/[[]//g' -e 's/[]]//g' \
    >${working_dir}/results.csv

echo "Done pulling results"

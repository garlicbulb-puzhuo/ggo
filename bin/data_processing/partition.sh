#!/bin/bash

# current program name
progname="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"

input_dir=
input_dir_arg=0

output_dir=
output_dir_arg=0

batch_size=50

function usage {
    echo "usage: ${progname} -i input_directory -o output_directory [-h]"
    echo "  -i  input directory"
    echo "  -o  output directory for the partitioned output"
    echo "  [-h] Display usage"
    exit 1
}

while getopts ":i:o:b:h" opt; do
    case $opt in
        i)
            input_dir=$OPTARG
            input_dir_arg=1
            ;;
        o)
            output_dir=$OPTARG
            output_dir_arg=1
            ;;
        b)
            batch_size=$OPTARG
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


files=($(ls ${input_dir}))
len=${#files[*]}

echo "Total number of input files: ${len}"
echo "Partitioning data into batches of size ${batch_size}"

i=0
while [ ${i} -lt ${len} ];
do
   n=$((i % ${batch_size}))
   batch=$(( i / ${batch_size} ))
   if [ ${n} -eq 0 ];
   then
       echo "Generate batch ${batch}"
       mkdir -p ${output_dir}/${batch}
   fi

   echo mv ${input_dir}/${files[$i]} ${output_dir}/${batch}
   mv ${input_dir}/${files[$i]} ${output_dir}/${batch}
   let i++
done


#!/bin/bash

working_dir=/home/csong/kaggle_data
model_weights=${working_dir}/model.weights.final.hdf5
for i in {0..20}
do
    mkdir -p ${working_dir}/predict_output/${i}
    mkdir -p ${working_dir}/predict_log/${i}
    
    echo python -m scripts.kaggle.nodule_detection --model_id 2 --model_weights ${model_weights} --input_path ${working_dir}/stage2_output/${i} --output_path ${working_dir}/predict_output/${i}
    nohup python -m scripts.kaggle.nodule_detection --model_id 2 --model_weights ${model_weights} --input_path ${working_dir}/stage2_output/${i} --output_path ${working_dir}/predict_output/${i} > ${working_dir}/predict_log/${i}/job.log 2>&1 &
done

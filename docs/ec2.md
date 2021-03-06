## Run GPU based training on an ec2 spot instance
- Create task directory on s3 and its configurations. The typical structure for a task looks like the following
    ```
     mnt/s3/ec2/test_run_1/
     ├── config
     │   └── task_config.ini
     ├── config.ini
     └── model
    ```
- Manually insert the task in the master ec2 mysql backend
    ```
    mysql -h ec2-54-153-80-106.us-west-1.compute.amazonaws.com -u kaggle -p
    
    mysql> use ggo;
    mysql> begin;
    mysql> insert into task (working_dir, worker) values ('ec2/test_run_1', 'host');
    mysql> commit;
    ```
- Select `700.v4` as the source AMI
- Make sure to check `Persistent request` 
- On the ec2 launch page, in `Advanced details` section, put the following bash script into `User data` when launching an ec2 instance.

    ```
    #!/bin/bash
    su -s /bin/bash ubuntu -c "~/Developer/ggo/bin/ec2_startup.sh > ~/Developer/application_logs/ec2_startup.log 2>&1"
    ```

- If you want to tear down the instance completely, cancel your spot request

## Run CPU based data preprocessing on ec2 instance
- Select `200.v4` as the source AMI 
- On the ec2 launch page, in `Advanced details` section, put the following bash script into `User data` when launching an ec2 instance.

    ```
    #!/bin/bash
    cd /home/ubuntu/Developer/ggo
    git pull --rebase
    su -s /bin/bash ubuntu -c "bin/bootstrap_ec2.sh > ~/Developer/application_logs/bootstrap_ec2.log 2>&1
    ```

- ssh onto the ec2 instance, s3 is mounted at `~/mnt/s3`
- Change working directory

    ```
    $ cd /home/ubuntu/Developer/ggo
    ```

- Find the directory on s3 where the input data is stored, say the input directory is `~/mnt/s3/kaggle/input`
- Partition the data set into multiple sub-sets, each of a smaller size, by running the following script

    ```
    $ mkdir -p ~/mnt/s3/kaggle/partitioned_input
    $ bin/data_processing/partition.sh -i ~/mnt/s3/kaggle/input -o ~/mnt/s3/kaggle/partitioned_input [-b batch_size]
    ```

    Please be noted that if you don't set `batch_size`, the default value is `50`.
    After this step, the image files in `~/mnt/s3/kaggle/input` will be partitioned and moved into `~/mnt/s3/kaggle/partitioned_input`.
- To run watershed preprocessing scripts in the background, run the following

    ```
    $ bin/data_processing/process.sh -p preprocess -i ~/mnt/s3/kaggle/partitioned_input -o ~/mnt/s3/kaggle/watershed_output -l /tmp/preprocesing_logs 
    ```

- To run nodule detection scripts in the background, run the following

    ```
    $ bin/data_processing/process.sh -m model_weights.hd5f -p predict -i ~/mnt/s3/kaggle/watershed_output  -o ~/mnt/s3/kaggle/prediction_output -l /tmp/prediction_logs 
    ```
    
## MISC
### install MySQL-python

```
sudo apt-get build-dep python-mysqldb
pip install MySQL-python
```

## Spot instance tips
### Run spot instance termination checker
When running a model in a spot instance, run a spot instance termination checker in a new tmux window. Here's an example command:
```
$ python -m scripts.ec2.ec2_terminate_checker -u 'sangmin\'s spot instance running model 1 with tensorflow'
```

## Run GPU based training on an ec2 spot instance
- Manually insert the task in the master ec2 mysql backend
- Select `cs231n.v2` as the source AMI
- Make sure to check `Persistent request` 
- Put the following bash script into `user data` when launching an ec2 instance

    ```
    #!/bin/bash
    su -s /bin/bash ubuntu -c "~/Developer/ggo/bin/ec2_startup.sh > ~/Developer/application_logs/ec2_startup.log 2>&1"
    ```

- If you want to tear down the instance completely, cancel your spot request

## Run CPU based data preprocessing on ec2 instance
- Select `200.v1` as the source AMI 
- Put the following bash script into `user data` when launching an ec2 instance

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
- Launch watershed preprocessing scripts in the background

    ```
    $ bin/data_processing/process_imgs.sh -i ~/mnt/s3/kaggle/partitioned_input -o ~/mnt/s3/kaggle/output -l /tmp/preprocesing_logs 
    ```

## MISC
### install MySQL-python

```
sudo apt-get build-dep python-mysqldb
pip install MySQL-python
```

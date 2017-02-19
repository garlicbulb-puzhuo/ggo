### install MySQL-python
```
sudo apt-get build-dep python-mysqldb
pip install MySQL-python
```

### Put the following script into `user data` when launching an ec2 instance 
```
#!/bin/bash
cd /home/ubuntu/Developer/ggo
git pull --rebase
su -s /bin/bash ubuntu -c "bin/bootstrap_ec2.sh > ~/Developer/application_logs/bootstrap_ec2.log 2>&1 "
su -s /bin/bash ubuntu -c "nohup bin/schedule_stask_ec2.sh > ~/Developer/application_logs/schedule_stask_ec2.log 2>&1 &"
su -s /bin/bash ubuntu -c "nohup bin/terminate_ec2.sh > ~/Developer/application_logs/terminate_ec2.log 2>&1 &"
```



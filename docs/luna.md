### Train luna data
- Modify the configuration file as needed. The default configuration file for luna16 is in `config/luna/config.ini`.
- Start training in standalone mode
```
$ cd ~/Developer/ggo
$ python -m scripts.luna.LUNA_train -t <train_path> -v <val_path> -c config/luna/model_1_theano.ini
```

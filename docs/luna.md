### Train luna data
- Modify the configuration file as needed. The default configuration file for luna16 is in `config/luna/config.ini`.
- Start training in standalone mode
```
$ cd ~/Developer/ggo
$ python -m scripts.luna.LUNA_train --train_path <train_path> --val_path <val_path> --config_file config/luna/config.ini
```

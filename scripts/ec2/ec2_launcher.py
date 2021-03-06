#!/usr/bin/env python2.7

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..luna.LUNA_train import main
from task import Task
from task import Base

import logging
import sys
import ConfigParser
import os
import argparse
import socket

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging_handler_out = logging.StreamHandler(sys.stdout)
logger.addHandler(logging_handler_out)


def get_parser():
    parser = argparse.ArgumentParser(description='launch task from ec2')
    parser.add_argument('-d', '--dburl', nargs='?', required=True,
                        help='the mysql db url')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    engine = create_engine(args.dburl)

    session = sessionmaker()
    session.configure(bind=engine)
    Base.metadata.create_all(engine)

    s = session()
    rs = s.query(Task).all()

    working_dir = None
    for task in rs:
        if task.state == 'pending' or task.state == 'starting':
            # start processing the task
            task.state = 'processing'
            task.worker = socket.gethostname()

            working_dir = task.working_dir

            s.add(task)
            s.commit()
            break

    if working_dir is None:
        logger.info('No pending task to be processed. Exit the program.')
        sys.exit(0)
    else:
        logger.info('Found one pending task in %s.' % working_dir)

    mount_path = '/home/ubuntu/mnt/s3'
    working_dir = os.path.join(mount_path, working_dir)

    # read config file from the working directory
    config = ConfigParser.ConfigParser()
    task_config_file = os.path.join(working_dir, 'config.ini')

    logger.info('Reading config file from %s...' % task_config_file)
    config.read(task_config_file)
    data_config = dict(config.items('config'))

    # prepare for arguments
    train_path = data_config.get('train_path')
    if not os.path.isabs(train_path):
        train_path = os.path.join(working_dir, train_path)

    val_path = data_config.get('val_path')
    if not os.path.isabs(val_path):
        val_path = os.path.join(working_dir, val_path)

    config_file = data_config.get('config_file')
    config_file = os.path.join(working_dir, config_file)

    prog_args = ['--train-path', train_path, '--val-path', val_path, '--config-file', config_file]
    logger.info('Preparing for main program arguments [%s]' % ','.join(map(str, prog_args)))

    # launch main program
    main(prog_args)


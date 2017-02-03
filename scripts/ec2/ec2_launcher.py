#!/usr/bin/env python2.7

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


from task import Task
from task import Base
from ..train import main

import logging
import sys
import ConfigParser
import os
import argparse

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging_handler_out = logging.StreamHandler(sys.stdout)
logger.addHandler(logging_handler_out)


def get_parser():
    parser = argparse.ArgumentParser(description='lauch task from ec2')
    parser.add_argument('--dburl', metavar='dburl', nargs='?',
                        help='the mysql db url')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if not args.dburl:
        parser.error('Required to set --dburl')

    engine = create_engine(args.dburl)

    session = sessionmaker()
    session.configure(bind=engine)
    Base.metadata.create_all(engine)

    s = session()
    rs = s.query(Task).all()

    for task in rs:
        if task.state == 'pending':
            # start processing the task
            task.state = 'processing'

            working_dir = task.working_dir

            s.add(task)
            s.commit()
            break

    if working_dir is None:
        logger.info('No pending task to be processed. Exit the program.')
        sys.exit(0)
    else:
        logger.info('Found one pending task in %s.' % working_dir)

    mount_path = '~/mnt/s3'
    working_dir = os.path.join(mount_path, working_dir)

    # read config file from the working directory
    config = ConfigParser.ConfigParser()
    task_config_file = os.path.join(mount_path, 'config.ini')
    config.read(task_config_file)
    data_config = dict(config.items('config'))

    # prepare for arguments
    train_imgs_path=data_config.get('train_imgs_path')
    config_file=data_config.get('config_file')
    prog_args = ['--train', '--train_imgs_path', train_imgs_path, '--train_mode', 'standalone', '--config_file', config_file]
    logger.info('Preparing for main program arguments [%s]' % ','.join(map(str, prog_args)))

    # launch main program
    main(prog_args)


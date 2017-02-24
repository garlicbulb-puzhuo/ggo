#!/usr/bin/env python2.7

import logging
import socket
import argparse
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from task import Task
from task import Base

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging_handler_out = logging.StreamHandler(sys.stdout)
logger.addHandler(logging_handler_out)


def get_parser():
    parser = argparse.ArgumentParser(description='launch task from ec2')
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
    worker_host = socket.gethostname()

    for task in rs:
        if task.state == 'processing' and task.worker == worker_host:
            # stop processing on the task
            task.state = 'pending'
            task.worker = 'NA'
            s.add(task)
            logger.info('Terminate the task [%d] on worker [%s]' % (task.id, worker_host))

    s.commit()



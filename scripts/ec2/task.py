#!/usr/bin/env python2.7


from sqlalchemy import Column, DateTime, String, Integer, func
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class Task(Base):
    __tablename__ = 'task'
    id = Column(Integer, primary_key=True)
    working_dir = Column(String)
    state = Column(String, default='starting')
    worker = Column(String)
    created_time = Column(DateTime, default=func.now())
    last_updated = Column(DateTime, default=func.now())



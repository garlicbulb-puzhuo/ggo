#!/usr/bin/env bash

set -e

BASEDIR=$(dirname $0)/..
DEVELOPER_ROOT_DIR=${BASEDIR}/..

source ${DEVELOPER_ROOT_DIR}/env/bin/activate

cd ${DEVELOPER_ROOT_DIR}/elephas
git pull --rebase
pip uninstall elephas
python setup.py install

deactivate

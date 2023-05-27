#!/bin/sh

set -ex

apt update
apt install python3.6
apt install python3.6-distutils

update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
echo 2 | update-alternatives --config python3

wget https://bootstrap.pypa.io/pip/3.6/get-pip.py -O get-pip.py
python get-pip.py
rm -f get-pip.py

python -m pip install --upgrade pip setuptools wheel
python -m pip install virtualenv

echo "done"

#!/bin/sh 

apt update
apt install python3.6
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
echo 2 | update-alternatives --config python3
wget https://bootstrap.pypa.io/pip/3.6/get-pip.py
python get-pip.py
python -m pip install--upgrade pip setuptools wheel



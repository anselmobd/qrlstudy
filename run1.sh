#!/bin/bash

./qgye.py a 30000 1000 -i1000
./qgye.py b 30000 1000 -i1000
./qgye.py c 30000 1000 -i1000
./qgye.py d 30000 1000 -i1000
./qgye.py e 30000 1000 -i1000
./qgye.py f 30000 1000 -i1000
./qgye.py g 30000 1000 -i1000

find ./qtb/qgye1/ -name "*.qtb" -exec ./qeval.py {} 100 100 -q \; | tee ./qtb/qgye1/qeval.txt

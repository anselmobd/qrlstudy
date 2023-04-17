#!/bin/bash

./qgye.py a 15000 1000 -i100
./qgye.py b 15000 1000 -i100
./qgye.py c 15000 1000 -i100
./qgye.py d 15000 1000 -i100
./qgye.py e 15000 1000 -i100
./qgye.py f 15000 1000 -i100
./qgye.py g 15000 1000 -i100

find ./qtb/qgye2/ -name "*.qtb" -exec ./qeval.py {} 100 100 -q \; | tee ./qtb/qgye2/qeval.txt

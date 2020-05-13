#! /usr/bin/env bash

jupyter nbconvert --to script notebooks/*.ipynb --output-dir=scripts

cd scripts
for SCRIPT in *.py
do
    ipython $SCRIPT --matplotlib inline
done
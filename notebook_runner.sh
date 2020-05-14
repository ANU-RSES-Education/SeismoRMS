#! /usr/bin/env bash

jupyter nbconvert --to script notebooks/SeismoSocialDistancing.ipynb --output-dir=scripts

cd scripts
ipython SeismoSocialDistancing.py --matplotlib inline

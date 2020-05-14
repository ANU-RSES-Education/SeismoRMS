#! /usr/bin/env bash

jupyter nbconvert --to script notebooks/*.ipynb --output-dir=scripts

cd scripts
ipython SeismoSocialDistancing.ipynb --matplotlib inline

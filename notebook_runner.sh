#! /usr/bin/env bash

mkdir -p workdir
jupyter nbconvert --to script notebooks/SeismoSocialDistancing.ipynb --output-dir=workdir
cp python/seismosocialdistancing_core.py workdir/seismosocialdistancing_core.py

cd workdir
ipython SeismoSocialDistancing.py --matplotlib inline

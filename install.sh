#!/usr/bin/env bash
pip install torch==0.4 torchvision
pip install scipy
pip install sklearn
pip install tensorboardX
cd tools/apex
pip install -v --no-cache-dir .
cd ../..

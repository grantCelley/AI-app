#!/bin/bash

if [ ! -d labelenv ]; then
    virtualenv -p python3 labelenv
    source labelenv/bin/activate
    pip install -r label_requirements.txt
else
    source labelenv/bin/activate
fi

labelImg data/*
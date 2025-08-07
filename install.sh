#!/bin/bash

cd calvin_env
pip install -e .

cd ../calvin_models
pip install -e .

# Install MulticoreTSNE separately
pip install git+https://github.com/DmitryUlyanov/Multicore-TSNE
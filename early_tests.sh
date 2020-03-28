#!/bin/bash

# Execute light tests which can be called before setting up environment to save time and resources.
cd src/
python -m unittest tests.env.test_config
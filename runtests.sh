#!/bin/bash

# Run pytest for all supported python versions locally in an isolated venv
# before submitting to GitHub/Travis-CI.

# Loop over Python 3.4, 3.5, 3.6
for ((i = 4; i<7; i++)); do

  # Print info
  echo " "
  echo "                   **************  PYTHON 3."${i}"  **************"
  echo " "

  # Create venv
  conda create -y -n test_3${i} python=3.${i} numpy scipy numexpr python-dateutil setuptools pytest pytest-cov mkl=2017.0.4 &> /dev/null

  # Activate venv
  source activate test_3${i}
  # Run tests
  pytest tests/ --cov=empymod
  # De-activate venv
  source deactivate test_3${i}
  # Remove venv
  conda remove -y -n test_3${i} --all &> /dev/null
  # Got to home directory

done

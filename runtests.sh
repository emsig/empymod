#!/bin/bash

# Help text
usage="
$(basename "$0") [-h] [-c -v]

Run pytest for empymod locally in an isolated venv before submitting to
GitHub/Travis-CI; by default for all supported python versions of empymod.

where:
    -h : show this help text
    -v : Python 3.x version, e.g. '-v 5' for Python 3.5
         Default: 4 5 6
    -c : Anaconda channel, e.g. '-c conda-forge' to use conda-forge
         Default: defaults
    -p : If provided prints output of conda.
"

# Set default values
CHANNEL=defaults
PYTHON3VERSION="4 5 6"
PRINT=0

# Get Optional Input
while getopts "hpv:c:" opt; do
  case $opt in
    h) echo "$usage"
       exit
       ;;
    v) PYTHON3VERSION=$OPTARG
       ;;
    c) CHANNEL=$OPTARG
       ;;
    p) PRINT=1
       ;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done

# Loop over Python versions
for i in ${PYTHON3VERSION[@]}; do

  # Print info
  echo " "
  STR="  PYTHON 3."${i}"  **  Channel "$CHANNEL"  "
  LENGTH=$(( ($(tput cols) - ${#STR}) / 2 - 2 ))
  printf "  "
  printf '\e[1m\e[34m%*s' "${LENGTH}" '' | tr ' ' -
  if [ $((${#STR}%2)) -ne 0 ];
  then
      printf "-"
  fi
  printf "${STR}"
  printf '%*s\n' "${LENGTH}" '' | tr ' ' -
  printf "\e[0m\n"

  # Create venv
  if [ "$PRINT" == "1" ]; then
    conda create -y -n test_3${i} python=3.${i}
  else
    conda create -y -n test_3${i} python=3.${i} &> /dev/null
  fi

  # Activate venv
  source activate test_3${i}

  # Install with CHANNEL
  if [ "$PRINT" == "1" ]; then
    conda install -y -c $CHANNEL numpy scipy numexpr python-dateutil \
        setuptools pytest pytest-cov
  else
    conda install -y -c $CHANNEL numpy scipy numexpr python-dateutil \
        setuptools pytest pytest-cov &> /dev/null
  fi

  # Run tests
  pytest tests/ --cov=empymod

  # De-activate venv
  source deactivate test_3${i}

  # Remove venv
  if [ "$PRINT" == "1" ]; then
    conda remove -y -n test_3${i} --all
  else
    conda remove -y -n test_3${i} --all &> /dev/null
  fi

done

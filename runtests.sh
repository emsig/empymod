#!/bin/bash

# Help text
usage="
$(basename "$0") [-hcpmdw] [-v VERSION(S)]

Run pytest for empymod locally in an isolated virtual environment before
submitting to GitHub; by default for all supported python versions of empymod.

where:
    -h : Show this help text.
    -v : Python 3.x version, e.g. '-v 7' for Python 3.7. Default: '6 7 8'.
    -p : Print output of conda.
    -m : Run tests without matplotlib/IPython.
    -d : Delete environments after tests
    -w : Disable warnings

"

# Set default values
PYTHON3VERSION="6 7 8"
PRINT="/dev/null"
PCKGS="numpy scipy numba pytest pytest-cov scooby pytest-flake8 pip"
MPLIPY="matplotlib IPython"
STR2="WITH matplotlib/IPython  "
PROPS="--mpl --flake8"
INST="pytest-mpl"
SD="_soft-dep"
WARN=""

# Get Optional Input
while getopts "hv:pmdw" opt; do

  case $opt in
    h) echo "$usage"
       exit
       ;;
    v) PYTHON3VERSION=$OPTARG
       ;;
    p) PRINT="/dev/tty"
       ;;
    m) MPLIPY=""
       STR2="NO matplotlib/IPython  "
       PROPS="--flake8"
       INST="pytest-flake8"
       SD=""
       ;;
    d) DELETE=true
       ;;
    w) WARN="--disable-warnings"
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

  # Environment name
  NAME=test_empymod_3${i}_${SD}

  # Print info
  STR="  PYTHON 3.${i}  **  $STR2"
  LENGTH=$(( ($(tput cols) - ${#STR}) / 2 - 2 ))
  printf "\n  "
  printf '\e[1m\e[34m%*s' "${LENGTH}" '' | tr ' ' -
  if [ $((${#STR}%2)) -ne 0 ];
  then
      printf "-"
  fi
  printf "${STR}"
  printf '%*s\n' "${LENGTH}" '' | tr ' ' -
  printf "\e[0m\n"

  # Create virtual environment
  if [ ! -d "$HOME/anaconda3/envs/$NAME" ]; then
    conda create -y -n $NAME -c conda-forge python=3.${i} $PCKGS $MPLIPY &> $PRINT
  fi

  # Activate virtual environment
  source activate $NAME

  # Install flake8
  if [ ! -d "$HOME/anaconda3/envs"+$NAME ]; then
    pip install $INST &> $PRINT
  fi

  # Run tests
  cp tests/matplotlibrc .
  pytest --cov=empymod $PROPS $WARN
  rm matplotlibrc
  coverage html

  # De-activate virtual environment
  conda deactivate

  # Remove virtual environment
  if [ "$DELETE" = true ] ; then
    conda remove -y -n $NAME --all &> $PRINT
  fi

done

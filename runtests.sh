#!/bin/bash

# Help text
usage="
$(basename "$0") [-hcpndw] [-v VERSION(S)]

Run pytest for empymod locally in an isolated venv before submitting to
GitHub/Travis-CI; by default for all supported python versions of empymod.

where:
    -h : Show this help text.
    -v : Python 3.x version, e.g. '-v 6' for Python 3.6. Default: '5 6 7'.
    -c : Use channel 'conda-forge' instead of channel 'defaults'.
    -p : Print output of conda.
    -n : Run tests without numexpr/matplotlib/IPython.
    -d : Delete environments after tests
    -w : Disable warnings

"

# Set default values
CHAN=defaults
PYTHON3VERSION="5 6 7"
PRINT="/dev/null"
PCKGS="numpy scipy pytest pytest-cov"
NMXPR="numexpr matplotlib IPython"
STR2="**  WITH numexpr/matplotlib/IPython  "
PROPS="--mpl --flake8"
INST="pytest-flake8 pytest-mpl scooby"
SD="_soft-dep"
WARN=""

# Get Optional Input
while getopts "hv:cpndw" opt; do

  case $opt in
    h) echo "$usage"
       exit
       ;;
    v) PYTHON3VERSION=$OPTARG
       ;;
    c) CHAN=conda-forge
       ;;
    p) PRINT="/dev/tty"
       ;;
    n) NMXPR=""
       STR2="**  NO numexpr/matplotlib/IPython  "
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
  NAME=test_3${i}_${CHAN}${SD}

  # Print info
  STR="  PYTHON 3.${i}  **  Channel $CHAN  $STR2"
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

  # Create venv, with channel CHAN
  if [ ! -d "$HOME/anaconda3/envs/$NAME" ]; then
    conda create -y -n $NAME -c $CHAN python=3.${i} $PCKGS $NMXPR &> $PRINT
  fi

  # Activate venv
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

  # De-activate venv
  conda deactivate

  # Remove venv
  if [ "$DELETE" = true ] ; then
    conda remove -y -n $NAME --all &> $PRINT
  fi

done

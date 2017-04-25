# 10-703 Homework 3 Programming

## Collaborators
Frederick Lee (frederil)

Paul Michel (pmichel1)

Hai Pham (htpham)

Richard Zhu (rzhu1)

## Setup

To install the packages using pip and a virtualenv run the following
commands:

```
cd deeprl_hw3_src
virtualenv hw3-env
source hw3-env/bin/activate
pip install -U -r requirements.txt
```

Make sure you are on Python 3.5.2.

## Outputs

For Question #2 and the extra credit for DAGGER, run

```
python run_imitation.py
```

which should output all necessary information in `imitation_output.txt`, `dagger_data.csv`, and `DAGGER Plot.png`.

Similarly for Question #3, run the following

```
python run_reinforce.py
```

to get outputs in `reinforce_data.csv`, `reinforce_output.txt`, and `REINFORCE Plot.png`.
# Learning Flexible and Reusable Locomotion Primitives for a Microrobot
This is the repository for the paper "Learning Flexible and Reusable Locomotion Primitives for a Microrobot". More information can be found on our website [here](https://sites.google.com/view/learning-locomotion-primitives/).
Included are demos for running the experiments laid out in the paper.

## Installation
Tested and maintained for Python 2.7.12/3.5.2.
### External Dependencies
Before installing the repo, there are two dependencies that need to be set up manually.
* [V-REP](http://www.coppeliarobotics.com/downloads.html), an open-source robotics simulator used to run the experiments (the limited version works if you can't access the educational pro version).
* [Opto](https://github.com/robertocalandra/opto), a package that implements several of the optimization algorithms used.

### Installing Using Pip
To install the remaining dependencies, we recommend cloning into the repo and installing the libraries using pip:
```
git clone https://github.com/bhyang/microrobot-locomotion.git
cd microrobot-locomotion
pip install -r requirements.txt
```

## Simulator Setup
Before running any of the experiments, make sure V-REP is open (see the V-REP documentation for troubleshooting issues with installation/booting). Scenes are automatically loaded and can be found in `scenes/`. The default simulator settings should work fine, but check that the following settings are correct:
* Physics engine: Bullet 2.78
* Time step: 50 ms

## Running Experiments
To test the single-objective optimization for walking speed only, run:
```
python normal.py
```
To run the multi-objective optimization taking into account walking speed and energy efficiency, run:
```
python moo.py
```
To run the multi-objective optimization for unbounded gait discovery, run:
```
python discovery.py
```
To run the inclination optimization, run:
```
python incline.py
```
To run the turning optimization, run:
```
python turning.py
```
## Citation

If you find this code useful, please support us by citing our paper:

Yang, B.; Wang, G.; Calandra, R.; Contreras, D.; Levine, S. & Pister, K. Learning Flexible and Reusable Locomotion Primitives for a Microrobot IEEE Robotics and Automation Letters (RA-L), 2018
```
@Article{Yang2018,
  Title                    = {Learning Flexible and Reusable Locomotion Primitives for a Microrobot},
  Author                   = {Brian Yang and Grant Wang and Roberto Calandra and Daniel Contreras and Sergey Levine and Kristofer Pister},
  Journal                  = {IEEE Robotics and Automation Letters (RA-L)},
  Year                     = {2018},
  Doi                      = {10.1109/LRA.2018.2806083},
}
```

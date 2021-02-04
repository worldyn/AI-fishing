
## Scenario: 
A sea full of jellyfishes, one king fish and one diver.

## Objective:
Capture the king fish. The diver has to reach the king fish while avoiding the jellyfishes. 

## Specifications:
Finds the best policy (Q-learning e-greedy) to reach the king fish 
without touching the jellyfishes. Touching a jellyfish means a discount in the reward while reaching the 
king fish means a positive reward (usually).
The possible actions of the diver are: up, down, left, right.

# Instalation

##  Ubuntu 16.04 / 18.04 / 20.04

### Requirements:
- Anaconda 3 (recommended), you could also use virtualenv as done in the previous assignments.
- a python 3.6/3.7 blank environment (minimum)

# Install packages`
4) Install the required packages thorough pip
```
pip install -r requirements.txt
```

##  Instructions with virtualenv (from requirements.txt)

0) Open a terminal and unzip the exercise 
1) Move to the repository path in your system 
```
cd [path_to_the_repository]
```
2) Install virtualenv and load your python virtual environment
```
sudo pip install virtualenvwrapper
. /usr/local/bin/virtualenvwrapper.sh
mkvirtualenv -p /usr/bin/python3.6 fishingderby
```
3) Install the required packages thorough pip
```
pip install -r requirements.txt
```

## To run
1) To run your agent execute the following.
```
python main.py settings.yml
```
It will run the agent with the settings in settings.yml


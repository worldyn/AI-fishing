#!/usr/bin/env python3
from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import sys
import os
from hmm import HMM
from random import uniform
import json
import csv

def printerr(s):
  s += '\n'
  os.write(2,str.encode(s))

class PlayerControllerHMM(PlayerControllerHMMAbstract):
  # testcomment
  def init_parameters(self):
    """
    In this function you should initialize the parameters you will need,
    such as the initialization of models, or fishes, among others.
    """

    self.num_fishes = 7
    N = 14 #self.num_fishes # num of fish species
    M = 8 # num of directions to go 
    self.hmm_models = []
    # will be filled with up to 180 observations for each 70 fishes
    self.observe = [] 
    self.latest_guess = 0
    self.num_guesses = 0
    # num every model is trained
    self.num_train_models = zerosvect(self.num_fishes)     
    printerr("INITIALIZATION")

    for m in range(self.num_fishes):
      A0 = zeros(N,N)
      B0 = zeros(N,M)
      PI0 = zerosvect(N)

      # all states should transition to hidden state m
      for i in range(N):
        for j in range(N):
          if j == m:
            A0[i][j] = 0.48
          else:
            A0[i][j] = 0.04
      #printerr("AAAAAAAA0 " + str(A0))
      # always start at state m
      for i in range(N):
        if i == m:
          PI0[i] = 0.48
        else:
          PI0[i] = 0.04

      '''
      # put row stoch into PI0
      s = 0
      for i in range(N):
        r = uniform(0,1)
        PI0[i] = r
        s += r
      for i in range(N):
        PI0[i] /= s

      # put row stoch into A0
      for i in range(N):
        s = 0
        for j in range(N):
          r = uniform(0,1)
          A0[i][j] = r
          s += r
        for j in range(N):
          A0[i][j] /= s
      '''

      # put row stoch into B0
      for i in range(N):
        s = 0
        for j in range(M):
          r = uniform(0,1)
          B0[i][j] = r
          s += r
        for j in range(M):
          B0[i][j] /= s

      #print("PI0 for model " + str(m) + " : " + str(PI0))
      self.hmm_models.append(HMM(A0,B0,PI0,N,M))
      # end loop

  def guess(self, step, observations):
    """
    This method gets called on every iteration, providing observations.
    Here the player should process and store this information,
    and optionally make a guess by returning a tuple containing the fish index and the guess.
    :param step: iteration number
    :param observations: a list of N_FISH observations, encoded as integers
    :return: None or a tuple (fish_id, fish_type)
    """
    
    printerr("step = " + str(step))
    #printerr("observations (T=" + str(len(observations)) +") : " + str(observations))
    # This code would make a random guess on each step:
    #return (step % N_FISH, random.randint(0, N_SPECIES - 1))

    maxprob = float('-inf')
    guess = 0

    self.observe.append(observations)
    T = len(self.observe)

    # chance is prob better first round than shitty init model
    #if step < 4:
    #  return (step % N_FISH, random.randint(0, N_SPECIES - 1))
    
    O = [ self.observe[t][step % N_FISH] for t in range(T) ]

    for m in range(self.num_fishes):
      #self.hmm_models[m].T = T
      #self.hmm_models[m].O = observations
      prob = self.hmm_models[m].obsSeqProb(O, T)  
      #printerr("PROB for model " + str(m) + ": " + str(prob))
      #printerr("MODEL: " + str(self.hmm_models[m].PI[m]))
      if prob > maxprob:
        maxprob = prob
        guess = m
    
    # guess is most likely state for last hidden state
    self.latest_guess = guess
    self.num_guesses += 1
    printerr("HMM GUESS " + str(guess))

    if T == 70:
      printerr("--------- END RESULTS OF MODELS -------")
      for m in range(7):
        h = self.hmm_models[m]
        printerr("model " + str(m) + " was trained " \
          + str(self.num_train_models[m]) + " times")
        printerr("model " + str(m) + " : " + str(h))


    return (step % N_FISH, guess)

  def reveal(self, correct, fish_id, true_type):
    """
    This methods gets called whenever a guess was made.
    It informs the player about the guess result
    and reveals the correct type of that fish.
    :param correct: tells if the guess was correct
    :param fish_id: fish's index
    :param true_type: the correct type of the fish
    :return:
    """
    #printerr("Fish # " + str(fish_id) + "; GUESS: " + \
    #  str(correct) + "; real type: " + str(true_type))
    T = len(self.observe)
    #if (T > 10 and T % 5 == 0) or T > 50:
    if T > 0:
      O = [self.observe[i][fish_id] for i in range(len(self.observe))]

      printerr("Training fish #"+str(fish_id)+" on model " + str(true_type))
      self.hmm_models[true_type].train_bw(O,T, 1e5)
      self.num_train_models[true_type] += 1
      
      # save model to file
      # e.g: n7_m2_PARAMETER
      h = self.hmm_models[true_type]
      #printerr("NEW MODEL: " + str(h))
      #printerr("Trained model " + str(true_type) + ": " \
        #+ str(self.num_train_models[true_type]) + " times")

      #for j in range(self.N):
      #  t = self.A[0][j] == 0


      path = 'params/'
      prefix = path + 'n' + str(h.N) + '_m' + str(true_type) + '_' 
      #with open('testA.csv','w') as f:
      #  writer = csv.writer(f)
      #  writer.writerows(h.A)

      #with open(prefix + 'A', 'w') as f:
      #with open('testA', 'w') as f:
      #  f.write(json.dumps(self.hmm_models[true_type].A)
      #with open(prefix + 'B', 'w') as f:
      #with open('testB', 'w') as f:
      #  f.write(json.dumps(self.hmm_models[true_type].B)
      #with open(prefix + 'PI', 'w') as f:
      #with open('testPI', 'w') as f:
      #  f.write(json.dumps(self.hmm_models[true_type].PI)

      if T == 70:
        printerr("# trainings on models:")
        for m in range(7):
          printerr("model " + str(m) + " was trained " \
            + str(self.num_train_models[m]) + " times")

def zeros(rows,cols):
  return [ [0 for j in range(cols)] for i in range(rows) ]

def zerosvect(n):
  return [0 for i in range(n)]

def zerostensor(a,b,c):
  return [ [ [ 0 for k in range(c) ] for j in range(b)] for i in range(a) ]

def scalarmult(vector, a):
  return map(lambda x: x * a, vector)

# matrix multiplication, returns new matrix object (lists of lists)
def mult(X, Y):
  rowsX = len(X)
  colsX = len(X[0])
  rowsY = len(Y)
  colsY = len(Y[0])
  
  if colsX != rowsY:
    print("iiiiiiii")
    raise Exception("ERROR: colsX != rowsY")

  Z = zeros(rowsX, colsY)
  for i in range(rowsX):
    for j in range(colsY):
      s = 0
      for k in range(colsX):
        s += X[i][k] * Y[k][j]
      Z[i][j] = s

  return Z


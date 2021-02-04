import fileinput
import sys
from math import log, exp
from random import randint, uniform
#import numpy as np
#import matplotlib.pyplot as plt

class HMM:
  # PI = initial state distribution, vector size N
  # N = #states
  # M = #emissions
  def __init__(self, A, B, PI, N, M, T=0, O=[]):
    # No error checking
    self.A = A
    self.B = B
    self.PI = PI
    self.N = N
    self.M = M
    self.T = T
    self.O = O
  
  def __str__(self):
    s = "N = " + str(self.N) + " ; M = " + str(self.M) + "\n"
    s += "A = " + str(self.A) + "\n"
    s += "B = " + str(self.B) + "\n"
    s += "PI = " + str(self.PI) + "\n"
    return s
  def listA(self):
    s = ''
    for i in range(self.N):
      s += ' '.join(map(str,self.A[i]))
      s += ' '
    return s
    #return [ self.A[i][j] for j in range(self.N) for i in range(self.N) ]

  def listB(self):
    return [ self.B[i][j] for j in range(self.M) for i in range(self.N) ]

  # returns most likely LAST state from O,T
  # uses viterbi
  def lastProbableState(self, O, T):
    delta, deltaidx = self.deltas(O,T)

    # set last state as argmax of delta for time T-1
    maxval = float('-inf')
    last_max_state = 0
    for j in range(self.N):
      if delta[T-1][j] > maxval:
        maxval = delta[T-1][j]
        last_max_val = j
    return last_max_val

  # returns vector with most likely states for O, T
  # uses viterbi algo
  def stateSeqProb(self, O, T):
    delta, deltaidx = self.deltas(O,T)
    X = zerosvect(T)

    # set last state as argmax of delta for time T-1
    maxval = float('-inf')
    for j in range(self.N):
      if delta[T-1][j] > maxval:
        maxval = delta[T-1][j]
        X[T-1] = j

    #X[T-1] = maxstate
    # backtrack to other states
    for t in range(T-2, -1, -1):
      X[t] = deltaidx[t+1][X[t+1]]

    return X

  def deltas(self, O, T):
    #delta = zeros(self.N, T)
    #deltaidx = zeros(self.N, T)
    delta = zeros(T, self.N)
    deltaidx = zeros(T, self.N)

    # let probabilities 0...1 go from 1 to 2 instead to avoid zero in log
    for i in range(self.N):
      #delta[i][0] = logp(self.B[i][O[0]]) + logp(self.PI[i])
      delta[0][i] = logp(self.B[i][O[0]] * self.PI[i])
      #delta[i][0] = self.B[i][O[0]] * self.PI[i]
    
    for t in range(1,T):
      for i in range(self.N):
        delta[t][i] = float('-inf')
        #maxval = float('-inf')
        #maxstate = 0
        for j in range(self.N):
          #print("t == " + str(t))
          #print("i == " + str(i))
          #print("j == " + str(j) + "\n")
          #curr = logp(self.A[j][i]) + delta[j][t-1] + logp(self.B[i][O[t]])
          curr = logp(self.A[j][i]) + delta[t-1][j] + logp(self.B[i][O[t]])
          if curr > delta[t][i]:
            delta[t][i] = curr
            deltaidx[t][i] = j
    return delta,deltaidx

  def obsSeqProb(self, O, T):
    alphas,consts = self.forward2(O,T)
    p = 0
    for i in range(T):
      p -= logp(consts[i])
    return p 

  def train_bw(self, O, T, max_iters=150):
    iters = 0
    oldlogprob = float('-inf')
    logprob = 0
    #x_cords = np.arange(0,max_iters)
    #y_cords = 
    #print("O in train_bw:" + str(O))
    while True:
      # compute alphas=[NxT], consts=[T]
      alphas, consts = self.forward(O, T)
      # compute betas=[NxT]
      betas = self.backward(O, T, consts)
      # compute digamma=[txNxN], gamma=[txN]
      digammas, gammas = self.gammas(O, T, alphas, betas, consts)

      # re-estimate pi
      for i in range(self.N):
        self.PI[i] = gammas[0][i]
        if self.PI[i] < 0.005:
          self.PI[i] = 0.005

      # re-estimate A
      for i in range(self.N):
        denom = 0
        for t in range(T-1):
          denom += gammas[t][i]
        for j in range(self.N):
          numer = 0
          for t in range(T-1):
            numer += digammas[t][i][j]
          self.A[i][j] = numer / (denom + 1e-3)
          if self.A[i][j] < 0.005:
            self.A[i][j] = 0.005

      t = True
      for j in range(self.N):
        t = self.A[0][j] == 0
      if t:
        # WHOLE ROW IS ZERO!!!!
        
        #print("consts=" + str(consts))
        '''
        print("alphas=" + str(alphas))
        print("betas=" + str(betas))
        print("digammas=" + str(digammas))
        print("gammas=" + str(gammas))
        '''
        #while True:
        #  pass

      # normalize A
      '''
      for i in range(self.N):
        s = 0
        for j in range(self.N):
          #if self.A[i][j] < 1e-2:
            #self.A[i][j] = 1e-2
          s += self.A[i][j]
        if s == 0:
          print("A with zero: " + str(self.A))
        for j in range(self.N):
          self.A[i][j] /= s
      '''

      # re-estimate B
      for i in range(self.N):
        denom = 0
        for t in range(T):
          denom += gammas[t][i]
        for j in range(self.M):
          numer = 0
          for t in range(T):
            if O[t] == j:
              numer += gammas[t][i]
          self.B[i][j] = numer / (denom + 1e-3)
          if self.B[i][j] < 0.005:
            self.B[i][j] = 0.005
      
      # compute logged P(Observations | model)
      logprob = 0
      for t in range(T):
        logprob -= logp(consts[t])

      # check convergence
      iters += 1
      diff = oldlogprob - logprob
      #if iters >= max_iters or oldlogprob > logprob:
      #if oldlogprob > logprob:
      if diff*diff < 0.00001: #or iters >= max_iters:
        return
      else:
        oldlogprob = logprob

    # O = observations
  # T = num observations
  # alphas = [NxT]
  # betas = [TxN]
  # consts = [T]
  def gammas(self, O, T, alphas, betas, consts):
    digammas = zerostensor(T, self.N, self.N)
    gammas = zeros(T, self.N)
    #numer = float(0) # top
    #denom = float(0) # bottom

    for t in range(T-1):
      for i in range(self.N):
        gammas[t][i] = 0
        for j in range(self.N):
          digammas[t][i][j] = alphas[t][i] * self.A[i][j] \
            * self.B[j][O[t+1]] * betas[t+1][j] 
          #digammas[t][i][j] = logp(alphas[t][i]) + logp(self.A[i][j]) \
          #  + logp(self.B[j][O[t+1]]) + logp(betas[t+1][j])
          if digammas[t][i][j] < 0.005:
            digammas[t][i][j] = 0.005
          gammas[t][i] += digammas[t][i][j]
          if gammas[t][i] < 0.005:
            gammas[t][i] = 0.005
    # special case gammas for T-1
    for i in range (self.N):
      gammas[T-1][i] = alphas[T-1][i]

    return digammas, gammas

  def backward(self, O, T, consts):
    betas = zeros(T, self.N) 
    # initial betas scaled by consts[T-1]
    for i in range(self.N):
      betas[T-1][i] = consts[T-1]
    
    #a_scaling = zerosvect(self.N)
    # t = T-2 to 0
    for t in range(T-2,-1,-1):
      for i in range(self.N):
        betas[t][i] = 0
        for j in range(self.N):
          betas[t][i] += self.A[i][j] * self.B[j][O[t+1]] * betas[t+1][j]
        '''
          a_scaling[j] = logp(self.A[i][j]) + logp(self.B[j][O[t+1]]) \
            + logp(betas[t+1][j])
        b_scaling = max(a_scaling)
        s = 0
        for j in range(self.N):
          s += exp(a_scaling[j] - b_scaling)
        logbeta = b_scaling + logp(s)
        betas[t][i] = exp(logbeta)
        '''
        # scale betas with same factor as alphas
        betas[t][i] = consts[t] * betas[t][i]
    return betas

  def forward(self, O, T):
    alpha = zeros(T,self.N) # matrix w/ alphas for each time
    norm_consts = zerosvect(T) # vector with normalization constants

    # for case t = 0
    norm_consts[0] = 0 # start at zero to sum the terms
    for i in range(self.N):
      alpha[0][i] = self.PI[i] * self.B[i][O[0]]
      #print("ZZZERO alpha[0]["+str(i)+"], PI["+str(i)+"]="+str(self.PI[i])+" , B["+str(i)+"][O[0]] = " + str(self.B[i][O[0]]))
      norm_consts[0] += alpha[0][i]
    norm_consts[0] = 1 / (norm_consts[0]+1e-3)
    #print("NNNNNNNNNN norm_consts[0] = " + str(norm_consts[0]))

    # normalize alpha
    for i in range(self.N):
      alpha[0][i] = norm_consts[0] * alpha[0][i]
    
    # for case t > 0
    for t in range(1,T):
      norm_consts[t] = 0 # start at zero in order to sum the terms
      for i in range(self.N):
        alpha[t][i] = 0 # start at 0 for summation
        for j in range(self.N):
          alpha[t][i] += alpha[t-1][j] * self.A[j][i]
        alpha[t][i] *= self.B[i][O[t]]
        norm_consts[t] += alpha[t][i]
      norm_consts[t] = 1 / (norm_consts[t] + 1e-3)
      for i in range(self.N):
        alpha[t][i] *= norm_consts[t]

    return alpha, norm_consts


  # iterative forward algo with stamp's algo to avoid underflow
  # returns alphas and consts for normalization
  def forward2(self, O, T):
    alpha = zeros(self.N,T) # matrix w/ alphas for each time
    norm_consts = zerosvect(T) # vector with normalization constants

    # for case t = 0
    norm_consts[0] = 0 # start at zero to sum the terms
    for i in range(self.N):
      alpha[i][0] = self.PI[i] * self.B[i][O[0]]
      norm_consts[0] += alpha[i][0]
    norm_consts[0] = 1 / (norm_consts[0] + 1e-3)

    # normalize alpha
    for i in range(self.N):
      alpha[i][0] = norm_consts[0] * alpha[i][0]
    
    # for case t > 0
    for t in range(1,T):
      norm_consts[t] = 0 # start at zero in order to sum the terms
      for i in range(self.N):
        alpha[i][t] = 0 # start at 0 for summation
        for j in range(self.N):
          alpha[i][t] += alpha[j][t-1] * self.A[j][i]
        alpha[i][t] *= self.B[i][O[t]]
        norm_consts[t] += alpha[i][t]
      norm_consts[t] = 1 / (norm_consts[t] + 1e-3)
      for i in range(self.N):
        alpha[i][t] *= norm_consts[t]

    return alpha, norm_consts
    #return alpha

# return logged probability or Non if 0
def logp(prob):
  try:
    return log(prob)
  except:
    return float('-inf')

# returns index for maximum value in a list
def argmax(A):
  maxidx = 0
  maxval = A[0]
  for i in range(len(A)):
    val = A[i]
    if val > maxval:
      maxval = val
      maxidx = i
  return maxidx

# returns a new matrix object with zeros (lists of lists)
def zeros(rows, cols):
 return [ [0 for j in range(cols)] for i in range(rows) ]

# return a new vector object with N zeros
def zerosvect(n):
 return [0 for i in range(n)]

# return a x b x c tensor
def zerostensor(a,b,c):
 return [ [ [ 0 for k in range(c) ] for j in range(b)] for i in range(a) ]

# vector v = [..].
def scalarmult(vector, a):
  return map(lambda x: x * a, vector) 

# matrix multiplication, returns new matrix object (lists of lists)
def mult(X, Y):
  rowsX = len(X)  
  colsX = len(X[0])
  rowsY = len(Y)  
  colsY = len(Y[0])

  #print("mult colsX = " + str(colsX) + "; rowsY = " + str(rowsY))

  if colsX != rowsY: 
    #print("iiiiiiii")
    raise Exception("ERROR: colsX != rowsY")

  Z = zeros(rowsX, colsY)
  for i in range(rowsX):
    for j in range(colsY):
      s = 0
      for k in range(colsX):
        s += X[i][k] * Y[k][j]
      Z[i][j] = s

    return Z

# return 1xM matrix with distribution of emissions for the next step
# A trans.mat, B emission mat, Q state prob. distribution
def nextEmissionDistribution(A, B, Q):
  return mult(mult(Q,A),B)
  

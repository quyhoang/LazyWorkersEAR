"""
16Jan20
Genetic Algorithm
The swam laziness is in walking behavior
The threshold is for walking. A random number is used to compare with the walking threshold. Agents always report if detect algae.
If the threshold is to0 high, the agents do not move -> fail.
If the threshold is too low, agents always move and lose energy for walking.
There will be no upper limit for energy (the upper limit is very high) - later we can introduce the effect of an upper limit

Two fitness function
Final 
Threshold sequence is sorted.
"""

# from setting import* #parameters
# Libraries
# import pdb #debug with set_trace()
# import copy #copy list
# import csv #export data
import concurrent.futures #parallel computing

#Calculation
from math import *
import numpy as np
import statistics as stat

import random
import time
start = time.perf_counter()

random.seed(time.time())

#Visualization
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-pastel')
#%matplotlib qt #this is needed to show animation. Otherwise, we need to change the setting
# if sypder is not set to show animation, copy the previous line to the console first then run this code

#Utilize GPU
# from numba import jit, cuda

#Parallel processing with multi-core CPU
# import multiprocessing as mp
# pool = mp.Pool(mp.cpu_count())

# Multiagent framework
from mesa import Agent, Model
#from mesa.time import RandomActivation
from mesa.time import SimultaneousActivation #Using random activation 
from mesa.space import MultiGrid #allow multiple robot on a cell
#from mesa.space import SingleGrid



#%% GENERAL SETTING
# =============================================================================
#
#random.seed(a=None, version=2)
#random.seed(1)
#np.random.seed(1)


# =============================================================================
# SWARM and SPACE
# =============================================================================
swarmSize = 100 #number of individuals in each swarm
megaSwarmSize = 20 #number of swarms

universalHeight = 15
universalWidth = 15

 

# =============================================================================
# ENERGY
# =============================================================================
EupThreshold = 2500000 #Max energy - Eharvest
Ethreshold = 0 #Elow threshold

Eharvest = 15
Edetect = 0 #detect algae
Ereport = 15 #debugging
Emove = 20 #walking

initialEnergy = 1000

energyProbability = 1
EnergyAvailablePercentage = 500 #80%


# =============================================================================
# FOR EVALUATION
# =============================================================================
algaeAppearanceRate = 0.01 #this decides how long the operational time is
maxLifeTimeAlgae = 200 #if algae live longer than this, the robot system fails

bestFitQueueLength = 10 #best fitnesses are saved in a queue
nTrials = 500 #number of trials to get averate effective operational time
nTrials2 = 1

maxStep = 100_000 #total steps, used in cumulative fitness functions

fitness = open("./data/fitnessUniversal.txt",'a')
meanFitness = open("./data/meanFitnessUniversal.txt",'a')
threshold = open("./data/thresholdUniversal.txt",'a')
meanThreshold = open("./data/meanThresholdUniversal.txt",'a')

print(f'Condition: Swarm size = {swarmSize}, Mega Swarm Size = {megaSwarmSize}', file = meanThreshold)
print(f'Grid Size = {universalHeight}*{universalWidth}', file = meanThreshold)
print(f'Allowed lifetime of algae = {maxLifeTimeAlgae}, Enargy Availability = {EnergyAvailablePercentage/10}%', file = meanThreshold)

#%% SWARM CLASS
# =============================================================================

class MoniModel(Model):   
    def __init__(self, N, width = 10, height = 10):
        self.num_agents = N
        self.width = width
        self.height = height
        self.grid = MultiGrid(height, width, False) #non toroidal grid
        self.schedule = SimultaneousActivation(self) #all active agents move together
        self.moveStimulus = random.randint(0,maxLifeTimeAlgae)
        

        self.threshold = [0 for _ in range(N)] #response threshold of N agents in the model, variable. This will not be used
        #only for initialization
        
        self.abCount = 0 #initial abnormality count
        self.detectedAb = 0
        
        # Create map of abnormalities
        self.anomalyMap = np.zeros((height,width))  # a 2D array represent the grid
        self.fail = 0 #fail = 1 when there are algae exceed maximum allowed lifetime 
        
        
        # Create agents
        for i in range(self.num_agents):
            #create and add agent with id number i to the scheduler
            a = MoniAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
            
        
    def updateAnomaly(self):
        '''
        Update the map that contain information of algae (stimulus value)
        '''
        for i in range(self.height):
            for j in range(self.width):#                pdb.set_trace()            
                if self.anomalyMap[i,j] > 0:
                    self.anomalyMap[i,j] += 1
                    if self.anomalyMap[i,j] > maxLifeTimeAlgae:      
                        self.fail = 1
                elif random.random() < algaeAppearanceRate: #rate of appearance of algae ----------------------------------------------------------------------------------------------
                    self.anomalyMap[i,j] += 1
                    
        if np.amax(self.anomalyMap) < maxLifeTimeAlgae:
            #after the algae is cleaned. This is used for the cumulative fitness function.
            self.fail = 0
            # print(self.fail)

    def show(self):
        '''
        Show response threshold of all agents in the swarm
        '''
        print(self.threshold)
    
    def showEnergy(self):
        for agent in self.schedule.agents:
            print(agent.threshold,"->", agent.energy)
            print(agent.varyThreshold,"->", agent.energy)
            
    def step(self):
        self.moveStimulus = random.randint(0,maxLifeTimeAlgae)
        self.updateAnomaly()
        self.schedule.step() #step of each agent, combined
        
    
    def run_model(self, n):
        '''
        run the model in n step
        '''
        self.resetModel()
        for i in range(n):
            self.step()
            #print(self.schedule.steps)
            
            
    def fitness(self):
        self.resetModel() #reset step count, reposition agents
        while self.fail == 0:
            self.step()
            #print(self.schedule.steps)
        return self.schedule.steps
        #the returned value is the operational time (from beginning until fail)
    
    def fitness2(self):
        self.resetModel() #reset step count, reposition agents
        goodTime = 0
        #cumulative function, count the time with no bad algae
        while self.schedule.steps < maxStep:
            self.step()
            if self.fail == 0:
                goodTime += 1
        return goodTime
    
    def realFitness2(self):
        realFitness2 = 0
        for i in range(nTrials2):
            realFitness2 += self.fitness2()
        realFitness2 = realFitness2/nTrials2
        return realFitness2
        
        
            
        
    def realFitness(self): #real fitness is average of fitness over many trials (law of large number)
        realFitness = 0
        for i in range(nTrials):
            # self.resetModel()
            realFitness += self.fitness()
        realFitness = realFitness/nTrials
        return realFitness
    
    
 
        
    def resetModel(self):  
        self.threshold = sorted(self.threshold)
        #reinitialize agent position
        for i in range(self.num_agents):
            #create and add agent with id number i to the scheduler
            a = MoniAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
        
        #reset step number and fail flag
        self.schedule.steps = 0
        self.fail = 0
        
        #reset anomalyMap
        for i in range(self.height):
            for j in range(self.width):
                self.anomalyMap[i,j] = 0
                
        
    def copyModel(self):
        targetModel = MoniModel(self.num_agents, self.width, self.height)
        targetModel.threshold = self.threshold[:] #slice to copy the thresholds to target model
        targetModel.resetModel() #set threshold value for agents in the copied model
  
# =============================================================================
#         for agent in targetModel.schedule.agents:
#             for originalAgent in self.schedule.agents:
#                 if (agent.unique_id == originalAgent.unique_id):
#                     agent.threshold = originalAgent.threshold
# =============================================================================
                    
                    
        return targetModel          
                
 
# =============================================================================
# Agent class               
# =============================================================================
    
class MoniAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.threshold = self.model.threshold[unique_id]  
        self.varyThreshold = self.threshold
        self.energy = initialEnergy #initial energy
        self.nextPos = (0,0) #initial position

         
    def printAgent(self):
        print('pos:',self.pos[0],self.pos[1])
        
         
    def copyAgent(self):
        targetAgent = MoniAgent(self.unique_id,self.model)
        targetAgent.threshold = self.threshold
        return targetAgent
        
    def move(self):    
        def new_pos(self):
            possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False)
            return self.random.choice(possible_steps)
         
        self.nextPos = new_pos(self)
        
            
    def energyAvailable(self):
        #binary model
        if self.model.schedule.steps%1000 < EnergyAvailablePercentage:  #in 1000 steps
            if random.random() < energyProbability:
                return 1
        return 0
        
    def step(self): 
        #harvest energy
        if self.energyAvailable(): #Duration in which an agent could get energy
            self.energy += Eharvest #energy harvested in this step
            self.varyThreshold = self.threshold
        else:
            self.varyThreshold = 0
            #when energy is not available, lazy agents tend to move more
# =============================================================================
#             if self.energy > EupThreshold:
#                 self.energy = EupThreshold     
# =============================================================================
                
        if self.energy > Edetect:      
            self.energy -= Edetect #spend energy to detect algae
            ##algae with stimulus higher than agent response threshold exist
            if (self.model.anomalyMap[self.pos[0],self.pos[1]]): #if there are algae   
                if self.energy > Ereport:
                    self.energy -= Ereport #report to base station
                    self.model.anomalyMap[self.pos[0],self.pos[1]] = 0 #algae is removed
                
        if self.energy > Emove and self.model.anomalyMap[self.pos[0],self.pos[1]] == 0:
            #if the location is cleared of algae and it can move
            # if random.randint(0,maxLifeTimeAlgae) > self.threshold:
            if self.model.moveStimulus > self.varyThreshold: #same value for the whole swarm
            #if the random number is higher than the lazy tendency of the agent, it moves    
                self.energy -= Emove
                self.move()
    
    #next step after staged change
    def advance(self):
        #although this is simultaneous activation, actually in the step stage, agents are activated randomly.
        #they only move together.
        self.model.grid.move_agent(self, self.nextPos)
        

    
def swarmCrossover(*args): #generate new swarm
    '''
    Generate new swarm from parent swarms
    After parent swarms are decided, random pairs from both parents will be chosen to make new agents in the new swarm
    until the number of offsprings reaches swarm population.
    Checked
    ''' 
    offspringSwarm = MoniModel(swarmSize,universalWidth,universalHeight)
    parent1 = args[0].threshold
    parent2 = args[1].threshold
    
    for i in range(swarmSize):
        k = random.random()
        if k < 0.49:
            offspringSwarm.threshold[i] = parent1[i]
        elif k < 0.98:
            offspringSwarm.threshold[i] = parent2[i]
        else: 
            offspringSwarm.threshold[i] = random.randint(0,swarmSize)

#    offspringSwarm.show()
    return offspringSwarm
    

class MegaModel:
    def __init__(self, size):
        self.generation = 0 #initial generation count
        self.size = size
        self.megaSwarm = []
        self.sortedFitness = [0 for _ in range(self.size)] #later this is used to hold fitness of corresponding swarm
        self.bestFitQueue = [0 for _ in range (bestFitQueueLength)] #hold the best operational time of a swarm
        self.bestFitQueuePointer = bestFitQueueLength-1
        
        for _ in range(size):
            model = MoniModel(swarmSize,universalWidth,universalHeight)
            self.megaSwarm.append(model)
            
    def copyMega(self):
        '''
        Create a copy of the mega swarm
        '''
        targetMegaModel = MegaModel(self.size)
        for swarmIndex in range(self.size):
            targetMegaModel.megaSwarm[swarmIndex] = self.megaSwarm[swarmIndex].copyModel()
        return targetMegaModel
    
    def memberFitness(self,index):
        '''
        This return the fitness of a member swarm in a mega swarm
        Used to compute fitnesses in parallel
        '''
        return self.megaSwarm[index].realFitness2()
    
    
    def nextGeneration(self):
        '''
        Generate a new meta swarm from previous generation
        '''
        self.generation += 1
        megaSwarmCopy = self.copyMega()
        
        
        fit = [0 for _ in range(self.size)]
        # this list holds the fitness of each swarm in the group in each generation
        for swarmIndex in range(self.size):
            fit[swarmIndex] = self.megaSwarm[swarmIndex].realFitness2() #real fitness instead of fitness, second function
        
# =============================================================================
#         # Parallel processing, not applicable in Windows
#         fit = []
#         with concurrent.futures.ProcessPoolExecutor() as executor:
#             results = executor.map(self.memberFitness,range(self.size))
#         
#         for result in results:
#             fit.append(result)
# =============================================================================
        
        print(fit)
        if self.bestFitQueuePointer == bestFitQueueLength-1:
            self.bestFitQueuePointer = 0
        else:
                self.bestFitQueuePointer += 1
                
        self.bestFitQueue[self.bestFitQueuePointer] = fit[0]
        
        #best performance among all swarms, save to fitness.txt
        meanvalue = stat.mean(fit)
        print(max(fit),"mean",meanvalue)
        print(max(fit),file=fitness)
        print(meanvalue,file = meanFitness)
        
       
        

        megaSwarmCopy.sortedFitness = sorted(range(len(fit)), key=lambda k: fit[k], reverse = True)
        
        #save the best distribution to threshold.txt
        bestSwarmIndex = megaSwarmCopy.sortedFitness[0]
        bestDistribution = [agent.threshold for agent in megaSwarmCopy.megaSwarm[bestSwarmIndex].schedule.agents]
        print(bestDistribution,file=threshold)
        
        #save the thresholds of ALL swarms into a file
        temp = []
        for i in range(self.size):
            temp = temp + self.megaSwarm[i].threshold
        print(f'megaDistGen{self.generation} = {temp}',file = meanThreshold)
            
        
        
        for i in range(megaSwarmSize):
            parent1Index = megaSwarmCopy.sortedFitness[np.random.randint(0,megaSwarmCopy.size/3+1)]
            parent2Index = megaSwarmCopy.sortedFitness[np.random.randint(0,megaSwarmCopy.size/3+1)]
            parent1 = megaSwarmCopy.megaSwarm[parent1Index] 
            parent2 = megaSwarmCopy.megaSwarm[parent2Index]
            self.megaSwarm[i] = swarmCrossover(parent1,parent2)
        
        
    
    def evolve(self, generationCount):
        for _ in range(generationCount):
            self.nextGeneration()
#            if self.terminateCondition():
#                break
            
    def terminateCondition(self): 
        '''
        Condition for termination of evolution process
        '''
        if self.bestFitQueue[bestFitQueueLength-1]:
            if np.std(self.bestFitQueue) < 2: #not much improvement in 10 consecutive runs
                return 1      
        return 0
        
    def geneDecompose(self):
        '''
        Generate a histogram to show composition of value of genes
        '''
        flattenGenes = [agent.threshold for swarm in self.megaSwarm for agent in swarm.schedule.agents]
#        plt.figure()
        plt.hist(flattenGenes)
                
        

#%% EVOLUTION     
# =============================================================================


superSwarm = MegaModel(megaSwarmSize)

#initialize swarms
for i in range(megaSwarmSize): #the threshold is for walking
    # initially all swarms are homogeneous
    # temp1 = random.randint(0,maxLifeTimeAlgae/10-1) 
    temp1 = i
    temp2 = [random.randint(0,30) for _ in range(swarmSize)]
    
    temp3 = [temp1*10 for _ in range(swarmSize)]
    superSwarm.megaSwarm[i].threshold = [(temp3[k]+temp2[k])%maxLifeTimeAlgae for k in range(swarmSize)]
    #evenly distribute the values of genes
    
    superSwarm.megaSwarm[i].resetModel() #this set new thresholds to agents
    
    
# superSwarm.megaSwarm[1].threshold = [150 for _ in range(swarmSize)] #group of swarms
# superSwarm.evolve(200)  #with maximum number of generations
superSwarm.nextGeneration()





#%% Clean up
meanThreshold.close()
fitness.close()
meanFitness.close()
threshold.close()


finish = time.perf_counter()
print(f'Elapsed time: {round(finish-start,2)}s')


#%% Animation
# =============================================================================
# 
# fig = plt.figure()
# def animate(i):
#     superSwarm.nextGeneration()
#     flattenGenes = superSwarm.megaSwarm[superSwarm.sortedFitness[0]].threshold
# #    flattenGenes = [j for swarm in superSwarm.megaSwarm for j in swarm.threshold]
#     plt.cla()
#     plt.hist(flattenGenes)
#     plt.axis([0, swarmSize, 0, swarmSize])
#     plt.xlabel('Threshold values')
#     plt.ylabel('Frequency')
#     
# ani = animation.FuncAnimation(fig, animate, interval=5)
# plt.show()
# plt.close("all")
# =============================================================================





#%% Git
# =============================================================================
# # initialize git repository directly from IPython command line (Spyder 4)
# !git init
# !git add .
# !git commit -m "Completed code"
# !git remote add origin https://github.com/quyhoang/LazyWorkersEAR.git
# !git push -u origin master
# !git revert a5186f7b29de2db3f2b34cfd61e3949e21735a4d #revert to stable version
# =============================================================================

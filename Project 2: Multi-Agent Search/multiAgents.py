# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        #
        #
        #
        #   DEN EXW KANEI ALLAGES STA ALLA ARXEIA APLWS TA EVALA EPEIDH ANAGRAFETAI STHN EKFWNHSH
        #
        #
        #
        #

        # The function that I'm going to use to find the distance
        from util import manhattanDistance

        # Print the above variables to check exactly what do vars contain
        """
        print "\n1---->successorGameState",successorGameState
        print "\n2---->newFood",newFood
        print "\n3---->newScaredTimes",newScaredTimes
        print "\n4---->newGhostStates",newGhostStates
        print "\n5--->newPos",newPos 
        """

        # Initial food distance with float("-Inf"), this helps me from removing the guess what value do I have to put at inital
        fooDist = float("-inf")

        # This variable works like a const
        floatInf = float("-inf")

        # From current game state get the food
        currentFood = currentGameState.getFood()

        # And convert it to list , from T/F
        listFood = currentFood.asList()

        # Convert tuple to list (x,y) -> [x,y]
        listPosition = list(newPos)

        # Action holds the E,N,W,S,Stop (east,north,...) therefore that means that if pacman==stop it just stopped
        if action == 'Stop':
            # So we return the -inf
            return floatInf
        
        # For each Ghost state 
        for each_GhostState in newGhostStates:
            # Find ghost's position
            pos_GhostState = each_GhostState.getPosition()

            # Find ghost's scaredTime
            stime_GhostState = each_GhostState.scaredTimer

            # If ghost's position equals to new Pacman's position  
            if pos_GhostState == tuple(listPosition):

                # And if scared time == 0 which means that power pellet time has been finished 
                if stime_GhostState is 0:

                # Then return -inf
                    return floatInf

        # For each food that exists in list
        for each_Food in listFood:

            # Find manhattanDistance between Pacman and food
            # I put "-" before manhattan, because I want to get the smaller number from all distances, and if "-" wasn't exists we would take the biggest one
            # Therefore if we have 2,3,4,5 -> -2,-3,-4,-5 we are goind to get -2
            current_Distance =  -(manhattanDistance(listPosition, each_Food))
          
            # If current distance is bigger than fooDist then new final distance is current distance
            if current_Distance > fooDist:

                # set new fooDist
                fooDist = current_Distance
          
        return fooDist  


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def min_max_Main_function(depth, gameState, indexOfAgent):

            # Check if agent is >= from total number of agents for example number of agents is 3 but with values 0,1,2 
            if indexOfAgent >= gameState.getNumAgents():

                # if that is true then increase depth
                depth += 1

                # Set agent = Pacman
                indexOfAgent = 0

            # Check if depth is equal to self.depth which means that is equal to minimax action from current gameState self.depth is the total depth, depth is the current depth   
            if depth == self.depth:
                # Return a distastance from gameState
                return self.evaluationFunction(gameState)

            # Check if game state isLose,which means that we haven't died yet, and if game state is win which means if we eat all food    
            elif gameState.isLose() or gameState.isWin():
              return self.evaluationFunction(gameState)
            # If Agent is Pacman    
            elif indexOfAgent is 0:
              # return the max if agent is pacman
                return min_max_function(depth,gameState,indexOfAgent,1)
            else:
              # else if is a ghost return the min
                return min_max_function(depth,gameState,indexOfAgent,0) 


        def min_max_function(depth,gameState,indexOfAgent,minMaxFlag):

            # I followed the pseudocode from AI_hw2_frodistirio.pdf

            # indexOfAgent == 0 -> Pacman , indexOfAgent >=1 Ghosts tip is from above
            # minMaxFlag == 0 -> min , minMaxFlag == 1 -> max
            if minMaxFlag == 0: # MIN

                # I took that from tips above 
                list_LegalActions_Agent = gameState.getLegalActions(indexOfAgent)

                # initial a minim list
                vMin = ["",float("inf")]

                # if we dont' have legal moves 
                if not list_LegalActions_Agent:
                    return self.evaluationFunction(gameState)

                # For each eaction that exists in list of legal legal actions for agent
                for each_LegalAction in list_LegalActions_Agent:

                    # I took that from tips above, returns the successor game state after indexOfAgent takes an each_legalAction
                    succGameState = gameState.generateSuccessor(indexOfAgent, each_LegalAction)

                    # call main minMmax
                    v = min_max_Main_function(depth,succGameState,indexOfAgent + 1)

                    # Store it in a temp variable
                    tempV = v

                    # Check if type of v is list which means that has 2 arguments,the 1st one is the action for example W,E,S,N and the 2nd is the v              
                    if type(v) is list:

                        # Therefore with v[1] we take the v and with v[0] the E,W,S... etc
                        tempV = v[1]

                    # If new v is smaller than vMin, vMin had initialised with infinity           
                    if tempV < vMin[1]:

                        # Therefore set vMin with [action,v] and return it.
                        vMin = [each_LegalAction,tempV]
                        
                return vMin
            
            elif minMaxFlag == 1: # MAX
                # I took that from tips above 
                list_LegalActions_Agent = gameState.getLegalActions(indexOfAgent)

                # initial a maximum list
                vMax = ["",-float("inf")]

                # if we dont' have legal moves
                if not list_LegalActions_Agent:
                    return self.evaluationFunction(gameState)

                # For each eaction that exists in list of legal legal actions for agent
                for each_LegalAction in list_LegalActions_Agent:

                    # I took that from tips above, returns the successor game state after indexOfAgent takes an each_legalAction
                    succGameState = gameState.generateSuccessor(indexOfAgent, each_LegalAction)

                    # call main minMmax              
                    v = min_max_Main_function(depth,succGameState,indexOfAgent + 1)

                    # store v in a temp variable
                    tempV = v

                    # Check if type of v is list which means that has 2 arguments,the 1st one is the action for example W,E,S,N and the 2nd is the v              
                    if type(v) is list:

                        # Therefore with v[1] we take the v and with v[0] the E,W,S... etc
                        tempV = v[1]

                    # If new v is smaller than vMax, vMax had initialised with infinity           
                    if tempV > vMax[1]:

                        # Therefore set vMax with [action,v] and return it.
                        vMax = [each_LegalAction,tempV]
                        
                return vMax

        finalList = min_max_Main_function(0,gameState,0)
        # I return the action for example East, West, ... etc
        return finalList[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # The code is the same as above but the only difference is that I added alpha -beta implementation from berkeley pdf. 

        def min_max_Main_function(depth, gameState, indexOfAgent, alpha, beta):

            # Check if agent is >= from total number of agents
            if indexOfAgent >= gameState.getNumAgents():

                # if that is true then increase depth
                depth += 1

                # Set agent = Pacman
                indexOfAgent = 0

            # Check if depth is equal to self.depth which means that is equal to minimax action from current gameState    
            if depth == self.depth:
                return self.evaluationFunction(gameState)

            # Check if game state isLose,which means that we haven't died yet, and if game state is win which means if we eat all food    
            elif gameState.isLose() or gameState.isWin():
              return self.evaluationFunction(gameState)
            # If Agent is Pacman    
            elif indexOfAgent is 0:
              # return the max if agent is pacman
                return min_max_function(depth,gameState,indexOfAgent,1, alpha, beta)
            else:
              # else if is a ghost return the min
                return min_max_function(depth,gameState,indexOfAgent,0, alpha, beta) 


        def min_max_function(depth,gameState,indexOfAgent,minMaxFlag, alpha, beta):

            # I followed the pseudocode from AI_hw2_frodistirio.pdf

            # indexOfAgent == 0 -> Pacman , indexOfAgent >=1 Ghosts tip is from above
            # minMaxFlag == 0 -> min , minMaxFlag == 1 -> max
            if minMaxFlag == 0: # MIN

                # I took that from tips above 
                list_LegalActions_Agent = gameState.getLegalActions(indexOfAgent)

                # initial a minim list
                vMin = ["",float("inf")]

                # if we dont' have legal moves 
                if not list_LegalActions_Agent:
                    return self.evaluationFunction(gameState)

                # For each eaction that exists in list of legal legal actions for agent
                for each_LegalAction in list_LegalActions_Agent:

                    # I took that from tips above, returns the successor game state after indexOfAgent takes an each_legalAction
                    succGameState = gameState.generateSuccessor(indexOfAgent, each_LegalAction)

                    # call main minMmax
                    v = min_max_Main_function(depth,succGameState,indexOfAgent + 1, alpha, beta)

                    # Store it in a temp variable
                    tempV = v

                    # Check if type of v is list which means that has 2 arguments,the 1st one is the action for example W,E,S,N and the 2nd is the v              
                    if type(v) is list:

                        # Therefore with v[1] we take the v and with v[0] the E,W,S... etc
                        tempV = v[1]

                    # If new v is smaller than vMin, vMin had initialised with infinity           
                    if tempV < vMin[1]:

                      # Therefore set vMin with [action,v] and return it.
                        vMin = [each_LegalAction,tempV]

                    # That part is new one, it was based on berkeley project2 pdf 
                    if alpha > tempV:
                        return [each_LegalAction,tempV]

                    beta = min(beta,tempV)     
                return vMin
            
            elif minMaxFlag == 1: # MAX
                # I took that from tips above 
                list_LegalActions_Agent = gameState.getLegalActions(indexOfAgent)

                # initial a maximum list
                vMax = ["",-float("inf")]

                # if we dont' have legal moves
                if not list_LegalActions_Agent:
                    return self.evaluationFunction(gameState)

                # For each eaction that exists in list of legal legal actions for agent
                for each_LegalAction in list_LegalActions_Agent:

                    # I took that from tips above, returns the successor game state after indexOfAgent takes an each_legalAction
                    succGameState = gameState.generateSuccessor(indexOfAgent, each_LegalAction)

                    # call main minMmax              
                    v = min_max_Main_function(depth,succGameState,indexOfAgent + 1, alpha, beta)

                    # store v in a temp variable
                    tempV = v

                    # Check if type of v is list which means that has 2 arguments,the 1st one is the action for example W,E,S,N and the 2nd is the v              
                    if type(v) is list:

                        # Therefore with v[1] we take the v and with v[0] the E,W,S... etc
                        tempV = v[1]

                    # If new v is smaller than vMax, vMax had initialised with infinity           
                    if tempV > vMax[1]:

                        # Therefore set vMax with [action,v] and return it.
                        vMax = [each_LegalAction,tempV]

                    # That part is new one, it was based on berkeley project2 pdf  
                    if beta < tempV:
                        return [each_LegalAction,tempV]

                    alpha = max(alpha,tempV)    
                return vMax

        finalList = min_max_Main_function(0,gameState,0, -float("inf"), float("inf"))
        # I return the action for example East, West, ... etc
        return finalList[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expect_main_function(depth, gameState, indexOfAgent):

            # Check if agent is >= from total number of agents
            if indexOfAgent >= gameState.getNumAgents():

                # if that is true then increase depth
                depth += 1

                # Set agent = Pacman
                indexOfAgent = 0

            # Check if depth is equal to self.depth which means that is equal to minimax action from current gameState    
            if depth == self.depth:
                return self.evaluationFunction(gameState)

            # Check if game state isLose,which means that we haven't died yet, and if game state is win which means if we eat all food    
            elif gameState.isLose() or gameState.isWin():
              return self.evaluationFunction(gameState)
            # If Agent is Pacman    
            elif indexOfAgent is 0:
              # return the max if agent is pacman
                return expect_max_function(depth,gameState,indexOfAgent,1)
            else:
              # else if is a ghost return the min
                return expect_max_function(depth,gameState,indexOfAgent,0) 


        def expect_max_function(depth,gameState,indexOfAgent,expectMaxFunction):

            # I followed the pseudocode from AI_hw2_frodistirio.pdf

            # indexOfAgent == 0 -> Pacman , indexOfAgent >=1 Ghosts tip is from above
            # expectMaxFunction == 0 -> min , expectMaxFunction == 1 -> max
            if expectMaxFunction == 0: # MIN

                # I took that from tips above 
                list_LegalActions_Agent = gameState.getLegalActions(indexOfAgent)

                # find probability
                prob = 1.0/len(list_LegalActions_Agent)

                # initial a expectmax list , [action,probability]
                exMax = ["",0]                      

                # if we dont' have legal moves 
                if not list_LegalActions_Agent:
                    return self.evaluationFunction(gameState)

                # For each eaction that exists in list of legal legal actions for agent
                for each_LegalAction in list_LegalActions_Agent:

                    # Put at 1st place of list the action
                    exMax[0] = each_LegalAction

                    # I took that from tips above, returns the successor game state after indexOfAgent takes an each_legalAction
                    succGameState = gameState.generateSuccessor(indexOfAgent, each_LegalAction)

                    # call main minMmax
                    v = expect_main_function(depth,succGameState,indexOfAgent + 1)

                    # Store it in a temp variable
                    tempV = v



                    # Check if type of v is list which means that has 2 arguments,the 1st one is the action for example W,E,S,N and the 2nd is the v              
                    if type(v) is list:

                        # Therefore with v[1] we take the v and with v[0] the E,W,S... etc
                        tempV = v[1]


                    # Put at 2nd place of 1st element of list the value multiply with the probability
                    exMax[1] += tempV * prob
                        
                return exMax
            
            elif expectMaxFunction == 1: # MAX
                # I took that from tips above 
                list_LegalActions_Agent = gameState.getLegalActions(indexOfAgent)

                # initial a maximum list
                vMax = ["",-float("inf")]

                # if we dont' have legal moves
                if list_LegalActions_Agent is 0:
                    return self.evaluationFunction(gameState)

                # For each eaction that exists in list of legal legal actions for agent
                for each_LegalAction in list_LegalActions_Agent:

                    # I took that from tips above, returns the successor game state after indexOfAgent takes an each_legalAction
                    succGameState = gameState.generateSuccessor(indexOfAgent, each_LegalAction)

                    # call main minMmax              
                    v = expect_main_function(depth,succGameState,indexOfAgent + 1)

                    # store v in a temp variable
                    tempV = v

                    # Check if type of v is list which means that has 2 arguments,the 1st one is the action for example W,E,S,N and the 2nd is the v              
                    if type(v) is list:

                        # Therefore with v[1] we take the v and with v[0] the E,W,S... etc
                        tempV = v[1]

                    # If new v is smaller than vMax, vMax had initialised with infinity           
                    if tempV > vMax[1]:

                        # Therefore set vMax with [action,v] and return it.
                        vMax = [each_LegalAction,tempV]
                        
                return vMax

        finalList = expect_main_function(0,gameState,0)
        # I return the action for example East, West, ... etc
        return finalList[0]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # list food
    list_Food = []

    # Get food False-True
    position_Food = currentGameState.getFood()

    # get pacman's position    
    position_Pacman = currentGameState.getPacmanPosition()

    # convert pacman position from tuple to list
    list_position_Pacman = list(position_Pacman)

    # convert False True to list
    list_position_Food = position_Food.asList()

    # for each food that exists in list
    for each_food in list_position_Food:

        # find distance of pacman from current food
        distance_Pacman = manhattanDistance(list_position_Pacman,each_food)

        # append it to list and putthe number with minus, because we are going to take ta max which would be the "smaller"
        list_Food.append(-distance_Pacman)
    # if list_food its not true append the zero
    if not list_Food :
        list_Food.append(0)

    # get current score
    currentScore = currentGameState.getScore()
    

    # calculate the sum of these
    final_value = currentScore + max(list_Food)

    # and return it 
    return final_value        

# Abbreviation
better = betterEvaluationFunction


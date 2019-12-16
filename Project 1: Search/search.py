# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
	"""
  # print "Start:", problem.getStartState()
  #  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  #  print "Start's successors:", problem.getSuccessors(problem.getStartState())
    
    "*** YOUR CODE HERE ***"
    # Use a stack as frontier
    stack = util.Stack()

    # A list of visited nodes
    visited_nodes = []	
    
    # The first node is: (location,directions,path)
    First_node = (problem.getStartState(),None,[])

    # Push in PQ, the first node
    stack.push(First_node)

    # Loop
    while True:

    	# Check if prioriy_stack is empty
    	if stack.isEmpty():
    		return []

    	# Take the current element
    	current_node = stack.pop()

    	# Get location
    	current_node_location = current_node[0]

    	# Get directions
    	current_node_directions = current_node[1]

    	# Get path
    	current_node_path = current_node[2]

    	# Check if current node is new
    	if current_node_location not in visited_nodes:

    		# Put it in list, because its a new node
    		visited_nodes.append(current_node_location)

    		# here we have to check if that node is a goal state
    		if problem.isGoalState(current_node_location):

    			# return the current path, which is the goal state path
    			return current_node_path

    		# Else, current node is not goalstate so we have to find the successors to continue
    		succ = problem.getSuccessors(current_node_location)

    		# A list of successors
    		list_succ = list(succ)

    		# For each successor
    		for successor in list_succ:

    			# Current Successor location
    			succ_loc = successor[0]

    			# Current Successor direction
    			succ_directions = successor[1]

    			# Create the new final path, which is the direction of successor plus the current final path
    			new_final_path = current_node_path + [succ_directions]

    			# Check if successor exists in visited nodes
    			if succ_loc not in visited_nodes:

    				# Push it in stack
    				stack.push((succ_loc,succ_directions,new_final_path)) 

    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # First of all check if starting state is the goal state
    if problem.isGoalState(problem.getStartState()):
    	return []

    # Use a Queue as frontier
    queue = util.Queue()

    # A list of visited nodes
    visited_nodes = []	
    
    # The first node is: (location,directions,path)
    First_node = (problem.getStartState(),None,[])

    # Push in PQ, the first node
    queue.push(First_node)

    # Loop
    while True:

    	# Check if prioriy_queue is empty
    	if queue.isEmpty():
    		return []

    	# Take the current element
    	current_node = queue.pop()

    	# Get location
    	current_node_location = current_node[0]

    	# Get directions
    	current_node_directions = current_node[1]

    	# Get path
    	current_node_path = current_node[2]

    	# Check if current node is new
    	if current_node_location not in visited_nodes:

    		# Put it in list, because its a new node
    		visited_nodes.append(current_node_location)

    		# here we have to check if that node is a goal state
    		if problem.isGoalState(current_node_location):

    			# return the current path, which is the goal state path
    			return current_node_path

    		# Else, current node is not goalstate so we have to find the successors to continue
    		succ = problem.getSuccessors(current_node_location)

    		# A list of successors
    		list_succ = list(succ)

    		# For each successor
    		for successor in list_succ:

    			# Current Successor location
    			succ_loc = successor[0]

    			# Current Successor direction
    			succ_directions = successor[1]

    			# Create the new final path, which is the direction of successor plus the current final path
    			new_final_path = current_node_path + [succ_directions]

    			# Check if successor exists in visited nodes
    			if succ_loc not in visited_nodes:

    				# Push it in queue
    				queue.push((succ_loc,succ_directions,new_final_path)) 

    return []				



   




def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # First of all check if starting state is the goal state
    if problem.isGoalState(problem.getStartState()):
    	return []

    # Use a Priority Queue as frontier
    priority_queue = util.PriorityQueue()

    # A list of visited nodes
    visited_nodes = []	
      
    # The first node is:(location,directions,path,cost)
    First_node = ((problem.getStartState(),None),[],0)

    # Push in PQ, the first node with his final distance
    priority_queue.push(First_node,None)

    # loop
    while True:

    	# Check if prioriy_queue is empty
    	if priority_queue.isEmpty():
    		return []

    	# Take the current node
    	current_node = priority_queue.pop()

    	# Get location
    	current_node_location = current_node[0][0]

    	# Get directions
    	current_node_directions = current_node[0][1] 

    	# Get Path
    	current_node_path = current_node[1]

    	# Get cost
    	current_node_cost = current_node[2]

    	# if current_node_location is new node
    	if current_node_location not in visited_nodes:

    		# Put it in list
    		visited_nodes.append(current_node_location)

    		# Find successors
    		succ = problem.getSuccessors(current_node_location)

    		# A list of successors
    		list_succ = list(succ)

    		# Check if current node is goalstate
    		if problem.isGoalState(current_node_location):
    			return current_node_path

    		# For each successor
    		for successor in list_succ:

    			# Current Successor location
    			succ_loc = successor[0]

    			# Current Successor direction
    			succ_directions = successor[1]

    			# Create the new final path, which is the direction of successor plus the current final path
    			new_final_path = current_node_path + [succ_directions]

    			# Current Successor cost
    			new_final_cost = current_node_cost + successor[2]

    			# If current node is a new one
    			if succ_loc not in visited_nodes:

    				# Check if is a goal state
    				if problem.isGoalState(succ_loc):
    					return new_final_path

    				# Create a new node 
    				create_new_node = (successor, new_final_path,new_final_cost)

    				# And put it in pq
    				priority_queue.push(create_new_node,new_final_cost)

    return []
   

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # Use a Priority Queue as frontier
    priority_queue = util.PriorityQueue()

    # A list of visited nodes
    visited_nodes = []	
    
    # Beginning distance
    current_distance = 0

    # Put heuristic in var
    heur = heuristic(problem.getStartState(),problem)

    # Final distance is the heuristic plus the distance of two nodes
    final_distance = current_distance + heur

    # The first node is:(location,directions,cost,path)
    First_node = (problem.getStartState(),None,current_distance,[])

    # Push in PQ, the first node with his final distance
    priority_queue.push(First_node,final_distance)

    # loop
    while True:

    	# Check if prioriy_queue is empty
    	if priority_queue.isEmpty():
    		return []

    	# Take the current node
    	current_node = priority_queue.pop()

    	# Get location
    	current_node_location = current_node[0]

    	# Get directions
    	current_node_directions = current_node[1]

    	# Get cost
    	current_node_cost = current_node[2]

    	# Get Path
    	current_node_path = current_node[3]

    	# if current_node_location is new node
    	if current_node_location not in visited_nodes:

    		# Put it in list
    		visited_nodes.append(current_node_location)

    		# Find successors
    		succ = problem.getSuccessors(current_node_location)

    		# A list of successors
    		list_succ = list(succ)

    		# For every successor
    		for successor in list_succ:

    			# successor location
    			succ_loc = successor[0]

    			# successor directions
    			succ_directions = successor[1]

    			# successor cost
    			succ_cost = successor[2]

    			# If current successor location
    			if succ_loc not in visited_nodes:

    				# Check if is a goal state
    				if (problem.isGoalState(succ_loc)):
    					return current_node_path + [succ_directions]

    				# Calculate current diststance
    				current_distance = current_node_cost + succ_cost

    				# same at heuristic
    				heur = heuristic(succ_loc,problem)
    							
    				# same at final distance 
    				final_distance = current_distance + heur

    				# Create a new node
    				create_new_node = (succ_loc,succ_directions,current_distance,current_node_path + [succ_directions])
    				
    				# Push it at PQ
    				priority_queue.push(create_new_node,final_distance)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util, math,heapq, itertools

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.LastValues = util.Counter() # A Counter is a dict with default 0
        self.bestActions = util.Counter()
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        Vk = []
        for i in range(0, self.iterations):
            Vk.insert(i, util.Counter())
            for current_state in states:

                if current_state == 'TERMINAL_STATE':
                    Vk[i][current_state] = 0
                    continue

                val = []
                possible_actions = self.mdp.getPossibleActions(current_state)
                for action in possible_actions:
                    val.append(self.computeQValueFromValues(current_state, action))

                Vk[i][current_state] = max(val)

            self.values = Vk[i]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        value = 0
        avaliable_scenario = self.mdp.getTransitionStatesAndProbs(state, action)
        for state_prim, probability in avaliable_scenario:
            value += probability * (
                        self.mdp.getReward(state, action, state_prim) + self.discount * self.values[state_prim])

        return value


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)

        values = []
        for action in actions:
            q = self.computeQValueFromValues(state, action)
            values.append(q)

        maximum = max(values)
        maxIdx = values.index(maximum)
        return actions[maxIdx]


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        for i in range(0, self.iterations):

            idx = i % len(states)
            current_state = states[idx]

            if current_state == 'TERMINAL_STATE':
                self.values[current_state] = 0
                continue

            val = []
            possible_actions = self.mdp.getPossibleActions(current_state)
            for action in possible_actions:
                val.append(self.computeQValueFromValues(current_state, action))

            self.values[current_state] = max(val)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.REMOVED = '<removed-task>'  # placeholder for a removed task
        self.counter = itertools.count()  # unique sequence count

        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = util.Counter()
        states = self.mdp.getStates()

        for state in states:
            predecessors[state] = []

        for state in states:
            possible_actions = self.mdp.getPossibleActions(state)
            for action in possible_actions:
                avaliable_scenario = self.mdp.getTransitionStatesAndProbs(state, action)
                for state_prim, probability in avaliable_scenario:
                    if probability > 0:
                        predecessors[state_prim].append(state)

        for state in states:
            if state == 'TERMINAL_STATE':
                continue
            diff = self.find_diff(state)
            self.add_task(state, -diff)

        for i in range(0, self.iterations):
            current_state = self.pop_task()
            if current_state == 'empty':
                break

            if current_state == 'TERMINAL_STATE':
                #self.values[current_state] = 0
                continue

            val = []
            possible_actions = self.mdp.getPossibleActions(current_state)
            for action in possible_actions:
                val.append(self.computeQValueFromValues(current_state, action))

            self.values[current_state] = max(val)

            for pred in predecessors[current_state]:
                diff = self.find_diff(pred)
                if diff > self.theta:
                    self.add_task(pred, -diff)


    def find_diff(self, state):
        values = []
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            q = self.computeQValueFromValues(state, action)
            values.append(q)
        return abs(self.values[state] - max(values))

    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            if self.entry_finder[task][0] <= priority:
                return
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.pq, entry)

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_task(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heapq.heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task
        return 'empty'



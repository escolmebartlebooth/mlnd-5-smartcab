import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class QLearner(object):
    """A Q-learning object"""
    
    def __init__(self):
        self.learningRate = 0.2
        self.discountRate = 0.9
        self.epsilon = 0.5
        self.qTable = {}
        self.previous_state = ()
        self.previous_action = None
        self.previous_reward = 0
    
    def reset():
        self.learningRate = 0.2
        self.discountRate = 0.9
        self.epsilon = 0.5
        # don't reset qtable as it needs to learn...?
        #self.previous_state = ()
        #self.previous_action = None
        #self.previous_reward = 0
    
    def update(self,state,action,reward):
        # first time through
        if len(self.previous_state) <> 0:
            # if the state is not in the qtable, add it to table with zero values
            if self.previous_state not in self.qTable:
                self.qTable[self.previous_state] = {self.previous_action : 0}
            else:
                if self.previous_action not in self.qTable[self.previous_state]:
                    self.qTable[self.previous_state][self.previous_action] = self.previous_reward
                else:
                    # capture q(s,a) from table
                    qsa = self.qTable[self.previous_state][self.previous_action]
                    # if new state isn't in qtable, add it with zero values
                    if state not in self.qTable:
                        mqsa = 0
                    else:
                        maxkey = None
                        mqsa = 0
                        for key in self.qTable[state]:
                            if self.qTable[state][key] > mqsa:
                                maxkey = key
                                mqsa = self.qTable[state][key]
                    # now we have qsa and mqsa we can learn
                    qsa = (1-self.learningRate)*qsa
                    discountedRate = reward + (self.discountRate * mqsa)
                    qsa = qsa + (self.learningRate * discountedRate)
                    # update qTable
                    self.qTable[self.previous_state][self.previous_action] = qsa
        # can update previous state, action, reward
        self.previous_state = state
        self.previous_action = action
        self.previous_reward = reward
                                
                        
               
        # capture the maximum Q(s',a') value
        
        # using the 2 Q values and other params, calc new Q(s,a) using:
        # (1-alpha)*Q(s,a)+alpha * [reward + gamma * Qmax(s',a')]
        # update q table with that value
    
    def get_action(self,state):
        # should select best action based on greedy epsilon etc
        # if random action, choose random else choose best
        if random.randint(0,10) > (self.epsilon*10):
            print 'USING BEST'
            # pick best
            if state in self.qTable:
                print 'state present'
                # get best action if any present
                # do this by iterating the actions dictionary assoicated with the state key and returning max value
                maxkey = None
                maxvalue = 0
                for key in self.qTable[state]:
                    if self.qTable[state][key] > maxvalue:
                        maxkey = key
                        maxvalue = self.qTable[state][key]
                return maxkey
            else:
                # state not in qTable so return None
                return 'random'
        else:
            # pick random triggered by epsilon
            return 'random'
        
        

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        
        # override state variable initialised to None as empty tuple
        self.state = ()
        # create a q-learner
        self.q_learner = QLearner()

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        # reset state variable initialised to empty tuple
        self.state = ()
        # don't reset the qlearner as it learns over each trial
        # self.qLearner.reset
        
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        # initial view of state takes all inputs
        # 26-07: Update to remove right as it doesn't add value
        self.state = (self.next_waypoint,inputs['light'],inputs['left'],inputs['oncoming'])
        
        print self.state
        
        # TODO: Select action according to your policy
        action = None
        action = self.q_learner.get_action(self.state)
        # update action to be a random choice
        if action == 'random':
            # qtable returned random choice, so choose random 
            action = random.choice(self.env.valid_actions)
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        # learn one step in arrears
        self.q_learner.update(self.state,action,reward)
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""
    
    # create common place to set debug values
    dbg_deadline = True
    dbg_update_delay = 0.5
    dbg_display = False
    dbg_trials = 2

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=dbg_deadline)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=dbg_update_delay, display=dbg_display)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=dbg_trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()

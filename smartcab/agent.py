import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class QLearner(object):
    """A Q-learning object"""
    
    def __init__(self):
        # initialise learning parameters
        self.learning_rate = 1
        self.discount_rate = 0.9
        self.epsilon = 1
        
        # initialise q table and state, action, reward trackers
        self.q_table = {}
        self.previous_state = ()
        self.previous_action = None
        self.previous_reward = 0
        
        # initialise step counters for use in decaying parameters
        self.step = 0
        self.trial = 0
        self.results_table = []
    
    def reset(self, deadline, reward):
        self.trial = self.trial + 1
        self.results_table.append([self.trial - 1, deadline, reward ])
        # don't reset qtable as it needs to learn over all trials
        
    
    def update(self,state,action,reward):
        # update learning rate
        self.learning_rate = 1.0 / self.trial
        
        if len(self.previous_state) <> 0:
            # if previous state has been initialised
            if (self.previous_state not in self.q_table):
                # if the state is not in the qtable, add it to table with zero values
                self.q_table[self.previous_state] = {self.previous_action : 0}
            else:
                if (self.previous_action not in self.q_table[self.previous_state]):
                    # if the state is present but the action isn't add the action
                    self.q_table[self.previous_state][self.previous_action] = self.previous_reward
                else:
                    # capture q(s,a) from table
                    qsa = self.q_table[self.previous_state][self.previous_action]
                    if (state not in self.q_table):
                        # if new state isn't in q table, set the q(s',a') to zero
                        mqsa = 0
                    else:
                        # otherwise get the maximal q(s',a')
                        mqsa = 0
                        for key in self.q_table[state]:
                            if (self.q_table[state][key] > mqsa):
                                mqsa = self.q_table[state][key]
                    # now we have qsa and mqsa we can learn
                    # (1-alpha)*Q(s,a)+alpha * [reward + gamma * Qmax(s',a')]
                    qsa = (1-self.learning_rate)*qsa
                    discounted_rate = reward + (self.discount_rate * mqsa)
                    qsa = qsa + (self.learning_rate * discounted_rate)
                    # update q Table with new estimate for q(s,a)
                    self.q_table[self.previous_state][self.previous_action] = qsa
        # update previous state, action, reward for next pass
        self.previous_state = state
        self.previous_action = action
        self.previous_reward = reward
    
    def get_action(self,state):
        # should select best action based on greedy epsilon approach
        # if random action, choose random else choose best
    
        # update epsilon
        self.epsilon = 1.0 / self.trial

        if (random.random() < self.epsilon):
            # pick best
            if (state in self.q_table):
                # get best action if any present
                # do this by iterating the actions dictionary assoicated with the state key and returning max value
                max_key_value = None
                max_qsa_value = 0
                for key in self.q_table[state]:
                    if self.q_table[state][key] > max_qsa_value:
                        max_key_vaue = key
                        max_qsa_value = self.q_table[state][key]
                return max_key_value
            else:
                # state not in q_table so return None
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
        
        # set a run type to allow for different states
        # this should be overridden in run()
        self.run_type = 'random'

        # override state variable initialised to None as empty tuple
        self.state = ()

        # create a q-learner class
        self.q_learner = QLearner()
        self.q_learner.trial = 1
        self.last_reward = 0
        self.last_deadline = 0
        

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        
        # reset state variable initialised to empty tuple for each initial step of a trial
        self.state = ()
        
        # reset the qlearner as it learns over each trial
        self.q_learner.reset(self.last_deadline, self.last_reward)
       
        
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.last_deadline = deadline

        # switch to allow for running in different states
        if self.run_type == 'random':
            # if random run type, all actions chosen randomly so ignore state
            self.state = None
        elif self.run_type == 'way_light_only':
            # if way_light_only then combine traffic light and way point
            self.state = (self.next_waypoint,inputs['light'])
        else:
            # otherwise incorporate the vehicle states at the intersection
            self.state = (self.next_waypoint,inputs['light'],inputs['oncoming'],inputs['left'],inputs['right'])
        
        # TODO: Select action according to your policy
        
        action = None
        if self.run_type == 'random':
            # if run type is random, set action to string random
            action = 'random'
        else:
            # otherwise call the q learner to get the action and increment the step
            self.q_learner.step = self.q_learner.step + 1
            action = self.q_learner.get_action(self.state)
            
        if action == 'random':
            # run type is random or qtable returned random choice, so choose random 
            action = random.choice(self.env.valid_actions)
            
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.last_reward = reward

        # TODO: Learn policy based on state, action, reward

        # learn one step in arrears - assuming not a random run
        if self.run_type != 'random':
            self.q_learner.update(self.state,action,reward)
            
        print "LearningAgent.update(): way_point = {}, deadline = {}, inputs = {}, action = {}, reward = {}".format(self.next_waypoint,deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""
    
    # create common place to set debug values
    dbg_deadline = True
    dbg_update_delay = 0.01
    dbg_display = False
    dbg_trials = 100
    # create switches to run as random, state1, state2
    dbg_runtype = 'way_light_vehicles'

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    # set the run type (random choice, simple state, state with vehicles)
    a.run_type = dbg_runtype
    e.set_primary_agent(a, enforce_deadline=dbg_deadline)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=dbg_update_delay, display=dbg_display)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=dbg_trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    print a.q_learner.q_table
    print a.q_learner.results_table

if __name__ == '__main__':
    run()

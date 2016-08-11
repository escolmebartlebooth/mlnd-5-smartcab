import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

# import numpy and matplotlib for results
import numpy as np
from matplotlib import pyplot as plt

class QLearner(object):
    """A Q-learning object"""
    
    def __init__(self):
        # initialise learning parameters
        self.learning_rate = 1
        self.discount_rate = 0.95
        self.epsilon = 1
        
        # initialise q table and state, action, reward trackers
        self.q_table = {}
        self.previous_state = ()
        self.previous_action = None
        self.previous_reward = 0
        
        # initialise step counters for use in decaying parameters
        self.step = 0
        self.trial = 0
        
         # create a table to log results
        self.results_table = None
        self.negative_rewards = 0
        self.net_rewards = 0
        
        # check first reset
        self.reset_check = False
    
    def reset(self,t):
        if (self.reset_check):
            # update results table
            if (self.results_table == None):
                self.results_table = [[self.trial,t,self.previous_reward,self.negative_rewards,self.net_rewards]]
            else:
                self.results_table.append([self.trial,t,self.previous_reward,self.negative_rewards,self.net_rewards])

        self.trial = self.trial + 1
        
        self.reset_check = True
        
        # reset negative and net rewards counter
        self.negative_rewards = 0
        self.net_rewards = 0
        # don't reset qtable as it needs to learn over all trials
        
    def update(self,state,action,reward):
        # update learning rate
        self.learning_rate = 1.0 / self.step
        
        # if reward is negative increment trial counter
        if (reward <0):
            self.negative_rewards += 1
        
        # increment net rewards for trial
        self.net_rewards = self.net_rewards + reward
        
        # initialise state action pairs if not in qtable
        if (state not in self.q_table):
            self.q_table[state] = {action : 0}
        elif (action not in self.q_table[state]):
            self.q_table[state][action] = 0
        
        if len(self.previous_state) <> 0:
            # if previous state has been initialised
            # can assume states in q table from above with initial values
            # capture q(s,a) from table
            qsa = self.q_table[self.previous_state][self.previous_action]
            
            mqsa = 0
            for key in self.q_table[state]:
                if (self.q_table[state][key] > mqsa):
                    mqsa = self.q_table[state][key]
                    
            # now we have qsa and mqsa we can learn
            # (1-alpha)*Q(s,a)+alpha * [reward + gamma * Qmax(s',a')]
            qsa = (1-self.learning_rate)*qsa
            discounted_rate = self.previous_reward + (self.discount_rate * mqsa)
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
        self.epsilon = 1.0 / self.step
        
        if (random.random() > self.epsilon or state == None):
            # pick best
            if (state in self.q_table):
                # get best action if any present
                # do this by iterating the actions dictionary assoicated with the state key and returning max value
                max_key_value = None
                max_qsa_value = 0
                for key in self.q_table[state]:
                    if self.q_table[state][key] > max_qsa_value:
                        max_key_value = key
                        max_qsa_value = self.q_table[state][key]
                return max_key_value
            else:
                # state not in q_table so return None
                return 'random'
        else:
            # pick random triggered by epsilon
            return 'random'
        
    def show_results(self):
        # show Q Table
        print self.q_table
        
        # make results array into np array (check why have to do this again)
        out_array = np.array(self.results_table)   
        
        # print the number of successes, trials, percent success, total steps, total reward
        t_success = np.size(out_array[np.where(out_array[:,2] > 3)],0)
        t_trials = np.size(out_array[:,2])
        t_success_percent = (t_success * 1.0) / t_trials
        t_average_steps = np.average(out_array[:,1]) 
        t_average_neg_reward = np.average(out_array[:,3]) 
        print "Results: successes = {}, trials = {}, success_percent = {}, \
                 average steps = {}, average_neg_reward_count = {}".format(t_success, t_trials, 
                   t_success_percent, t_average_steps, t_average_neg_reward)
        
        # build x and y axes for graph output and show graph       
        a = out_array[:,4]
        b = out_array[:,3]
        c = out_array[:,1]
        x = out_array[:,0]
        plt.plot(x, a, label='Net Trial Reward')
        plt.plot(x, b, label='Negative Reward Count')
        plt.plot(x, c, label='Trial Steps')
        plt.legend()
        plt.show()
        

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
        
        # track steps
        self.step_count = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        
        # reset state variable initialised to empty tuple for each initial step of a trial
        self.state = ()
        
        # reset the qlearner as it learns over each trial
        self.q_learner.reset(self.step_count)
       
        
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.step_count = t

        # switch to allow for running in different states
        if self.run_type == 'random':
            # if random run type, all actions chosen randomly so ignore state
            self.state = None
        elif self.run_type == 'way_light_only':
            # if way_light_only then combine traffic light and way point
            self.state = (self.next_waypoint,inputs['light'])
        elif self.run_type == 'way_light_vehicles':
            # incorporate the vehicle states at the intersection
            self.state = (self.next_waypoint,inputs['light'],inputs['oncoming'],inputs['left'],inputs['right'])
        else:
            # otherwise do a combined state to reflect possible safe moves
            # state definition is waypoint + allowed turn true false for left, right, forward            
            if (inputs['light'] == "red"):
                # lights are red so just check left car state
                if (inputs['left'] == "forward"):
                    # lights are red and turn right is not possible
                    self.state = (self.next_waypoint,False,False,False)
                else:
                    # can turn right
                    self.state = (self.next_waypoint,False,True,False)
            else:
                # lights are green so more options
                # don't need to consider cars from left or right
                turn_left = True
                if (inputs['oncoming'] == "forward"
                  or inputs['oncoming'] == "right"):
                    turn_left = False
                self.state = (self.next_waypoint,turn_left,True,True)
        
        # TODO: Select action according to your policy
        
        # set action to None as a default
        action = None

        # call the q learner to get the action and increment the step
        self.q_learner.step = self.q_learner.step + 1
        action = self.q_learner.get_action(self.state)
            
        if action == 'random':
            # run type is random or qtable returned random choice, so choose random 
            action = random.choice(self.env.valid_actions)
            
        # Execute action and get reward
        reward = self.env.act(self, action)
        
        # TODO: Learn policy based on state, action, reward

        # learn one step in arrears - assuming not a random run
        if self.run_type != 'random':
            self.q_learner.update(self.state,action,reward)
            
        # print "LearningAgent.update(): way_point = {}, deadline = {}, inputs = {}, action = {}, reward = {}".format(self.next_waypoint,deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""
    
    # create common place to set debug values
    dbg_deadline = True
    dbg_update_delay = 0.01
    dbg_display = False
    dbg_trials = 100 
    
    # create switches to run as random, way_light, way_light_vehicles
    # random = take random actions only
    # way_light_only = Traffic Light, Way Point
    # way_light_Vehicle = Traffic Light, Way Point, Left, Right, Oncoming
    # way_light_modified (or any other value) = Way Point, Combination Light and Vehicle State
    dbg_runtype = 'way_light_only'

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

    # at the end of the simulation show results
    # call qlearner reset to get last trial result
    a.q_learner.reset(a.step_count)
    a.q_learner.show_results()
    

if __name__ == '__main__':
    run()

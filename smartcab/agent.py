import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        
        # override state variable initialised to None as empty dictionary
        self.state = {}

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

        # reset state variable initialised to empty dictionary
        self.state = {}

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        # initial view of state takes all inputs
        self.state['heading'] = self.env.agent_states[self]['heading']
        self.state['location'] = self.env.agent_states[self]['location']
        self.state['destination'] = self.env.agent_states[self]['destination']
        self.state['light'] = inputs['light']
        self.state['left'] = inputs['left']
        self.state['right'] = inputs['right']
        self.state['oncoming'] = inputs['oncoming']
        
        print self.state
        
        # TODO: Select action according to your policy
        action = None
        # update action to be a random choice
        action = random.choice(self.env.valid_actions)
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""
    
    # create common place to set debug values
    dbg_deadline = True
    dbg_update_delay = 0.5
    dbg_display = False
    dbg_trials = 1

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

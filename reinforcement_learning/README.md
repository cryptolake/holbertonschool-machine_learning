# Reinforcement Learning


## Intro:

The goal of RL is to solve problems with interactions with the environment,
where we have an "agent" (our model), pick an "action" (result of the model),
to which the "environment" (the space in which we want to solve the problem) responds
with a "reward" (we use rewards to update the weights of our model) that can be
positive or negative.

We can solve this in a supervised manner, but the difficulty of acquiring data with
the impossibility of surpassing humans (given that data is sampled from humans) renders
supervised learning not suitable, also some problems can not be solved using gradient
descent given that we can not derive the gradient of the loss functin or the loss
function is unknown so we can't use supervised learning.


## Markov Decision Processes:

Markov Decision Processes or MDPs are the basis for RL problems,
in the intro we were basically describing MDPs where we have an agent
that interacts with the environment which select through time (this where
the markov property comes in) at each times-step given the state of environment
the agent selects the best action which changes the state of environment
and the agent is given a reward as a result of its previous action, through
this process it's the agent goal to produce as much reward as possible.

This is a model for sequential decision making.

Components of an MDP:
The environment
The Agent
The set of possible states of the environment
The set of actions that the agent can do
The set of the rewards that the agent can take

At and Rt meaning the action and the reward of time step t have well defined
probability distributions that means that there is some probability Rt = r
r is a particular reward from the set of the rewards (these are transition probabilities)
just like in a vanilla markov process.


### Expected Return:

Expected is the sum of future rewards if we have Gt return at time t
then Gt = Rt+1 + Rt+2 + ... + RT where T is the final time step, 
it is the agent's goal to maximize the expected return of rewards
for continuous tasks: where we have an infinite processs we use discounted
rewards to insure our series Gt converges and to make sure that our current
reward is more important.


## Policies and Value functions:

A policy is the probability that an agent will select an action given
from a specific state, value fuctions on the other hand gives us how
good is a specific action or state for the agent.
If an agent follows policy pi then pi(a|s) is the probability of At=a if
St = s this means that at time t under policy pi the probability of taking
action a in state s is pi(a|s), for each state s pi is a probability
distribution the actions.

Value functions gives us the notion of how good a state or an action is
in terms of expected return this means its tied to the way an agen acts
and since the way an agent acts is influenced by the policy it's following
then value functions are defined with respect to policy.

we have two types of value functions:
- The state value function:

  It tells us how good any given state is for an agent following policy pi
  in other words it gives us the value of a state under pi
	  for vpi(s) = E[Gt | St = s] = E[Y*Rt+k+1|St = s]
	  


## Policy network:

Policy networks are the model we use to determine which action we choose,
in the example of pong we input the gameframe and output probability of an action,
example: 30% UP 70% DOWN, instead of choosing the higher probability so we
avoid doing the same thing over and over we sample from distribution in which
30% of UP and 70% DOWN this called exploration, we train policy networks with a method
called policy gradients, we reward our network given a property of the environment
in our example the score in pong so the goal the model is to score as much as possible
we are going to use rewards in the backward pass of the model to update the weights.


## Credit assignment problem:

A limitation of this approach is that imagine in the game of pong, our model
played really well most of the game but in the end made a mistake the problem
is that it's going to get negative score only for the last action which going
to reduce the likelihood of doing all the good actions before it. this happens
when we have a sparse reward system which to get a reward we need to take multiple
actions.


## Reward shaping

Reward is used to counter the credit assignment problem, we essantially
make a custom reward pattern for our environment, this however will result
in needing to design the reward system for every environment, another
problem is alignment meaning the model will try creative ways to get the
reward without doing the intended actions.

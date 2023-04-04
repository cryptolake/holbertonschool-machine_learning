# 0x01 Deep Q-Learning

## Deep Q Networks DQN

In Deep Q-learning instead of using value iteration with the value function (Q-table)
anymore to determine the next action to do, instead we use a DNN (deep neural netwok)
to approximate the value function in which we input a state vector and get the action
as the output, we then interact with the environment to get our reward which we use in the loss
function that computes the difference between the target q-value (estimated using the bellman equation) and the current q-value

we then update the weights of our network using gradient descent.

Note: In the DQN paper they only update the weights every k state change (for example in atari games we get every 4 frames in the state vector)
This will help the model predict next action by understanding how the current state is changing.

## Replay Memory

At time t, the agent's experience et is defined as this tuple:

et = (st, at, rt+1, st+1) 

et consists of the current state the action taken given the state
the reward from the action and the next state, we store agent's
experience in the replay memory with size N.

We sample a random batch from the replay memory to train with,
this alleviates the correlation between consecutive states.

## Fixed Q-targets and Dueling Double DQN

https://cugtyt.github.io/blog/rl-notes/201807201658.html

As we discussed DQNs calculate the next state q value for loss this means
that as we train our network both the current q value and target q value
will go in the same direction, this will result in instability, so 
we can use fixed q-targets in which we designate a seperate network
for estimating the q-value or [Dueling double DQN](https://arxiv.org/pdf/1511.06581.pdf)
which makes the DQN output two seperate estimators one for action and one for the target
q value, this is also particularly helpful as not every state needs an action take for
a reward ex. in racing game if the car already on the right track it doesn't need to
move to avoid obstacles.

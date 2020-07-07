# A2C_Cartpole
Actor critic algorithm combines the best of both worlds; the value function and also the policy gradient function. In this algorithm we can use the sample efficiency of value function algorithms and policy based methods for continuous space.
# Implementation:
A2C algorithm has 2 main classes, those are implemented in the code as policyest and value_est. The policyest class estimates the policy of a given state, which is the actor in A2C. It also has the advantage function in its loss function which is nothing but the critic in A2C. The advantage function is nothing but the TD error of value function which is calculated from value_est class. The advantage here says to the model how good the action is and how better it can be. The final class implemented is the actor_critic class. This is the soul of the model which combines both the above mentioned classes and also interacts with the environment. It collects the rewards and observation states from the environment and sends to policyest and value_est classes to update its values.
The environment used is the Cliff walking environment.
Cliff walking environment has reward -1 to reach the destination and all the other times the reward received is -100.
# Results:
Here initially the agent struggles to find optimal path but after 100 episodes it figures it out and thus rewards received is -1.
In the below graph we can see the length of episode (or time taken for the agent to find path in 1 episode). After 100 episodes it found the optimal path and thus time taken is also less.

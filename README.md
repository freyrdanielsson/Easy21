# Easy21

This project is mainly about applying, on one hand, a Monte Carlo Control on the game Easy21 and on the other hand applying Sarsa Lambda on the game Easy21

# Introduction

## Easy21
The game Easy21 is built around the famous casino game [Blackjack](https://en.wikipedia.org/wiki/Blackjack) with a slighty different set of rules.
The different rules are:
* The game is played with an infinite deck of cards

* There are no aces or picture (face) cards in this game
  - Each draw from the deck results in a value between 1 and 10 (uniformly distributed)
  
* The cards do have collors
  - Black (probability 2/3)
  - Red (probability 1/3)
  
* At the start of the game both the player and the dealer draw one black
card (fully observed)

* The values of the player’s cards are added (black cards) or subtracted (red cards)

* If the player’s sum exceeds 21, or becomes less than 1, then she “goes bust” and loses the game (reward −1)


## Policies
The **player** uses an ε-greedy policy. Meaning she choses a random action with a probability related to ε, otherwise the best action.

The **dealer** plays with a policy where he always sticks on any sum of 17 or greater, and hits otherwise.

# Implementation

## Monte Carlo Control
### Applying Monte-Carlo control to Easy21.
I initialized the value function to zero and used a time-varying scalar step-size of α<sub>t</sub> = 1/N(s<sub>t</sub>,a<sub>t</sub>). We use the policy described above for the player (ε-greedy) with ε<sub>t</sub> = N0/(N0 + N(s<sub>t</sub>), where N0 = 1000 is a constant. N(s) is the number of times that state s has been visited, and N(s,a) is the number of times that action a has been selected from state s.

### Applying TD learning control to Easy21.
I implemented Sarsa(λ) for Easy21. The pseudocode can be found [here](https://towardsdatascience.com/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287). For this implementation I used the same step-size and exploration schedules as in the Monte-Carlo contril implementation. I ran the algorithm with
λ ∈ {0, 0.1, 0.2, . . . , 1} for 1000 episodes each and reported the mean squared error comparing the true values Q<sup>∗</sup>(s,a) computed in the [Monte Carlo implementation](#monte-carlo-control) with the estimated values Q(s, a) computed by Sarsa. at the end, as well as adding an option of receiving a report for the mean squared error for each episode during the run.

# Results and Interpretation

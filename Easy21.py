import numpy as np
import matplotlib
from collections import defaultdict
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# actions: hit or stand
ACTION_HIT = 0
ACTION_STAND = 1  #  "strike" in the book
ACTIONS = [ACTION_HIT, ACTION_STAND]

def epsilon_greedy_policy(Q, epsilon, numActions, state):
	A = np.ones(numActions, dtype=float) * epsilon / numActions
	
	# Ternary if sententece:
	# Chose action 1 (stand) if (state, 0) is lower than (state, 1)
	# Chose action 0 (hit) if (state, 1) is lower than (state, 0)
	# else they are equal, so chose randomly
	best_action = 1 if Q[(state, 0)] < Q[(state, 1)] else 0 if Q[(state, 1)] < Q[(state, 0)] else np.random.choice([0,1])
	A[best_action] += (1.0 - epsilon)
	return A

# get a new card
def get_card(firstCard=False):
	card = np.random.randint(1, 11)
	return card * get_collor(firstCard)

# get collor of card
def get_collor(firstCard):
	if(firstCard):
		return 1 #black

	x = np.random.uniform(0, 1)
	if (x < (1/3)):
		return -1 #red
	return 1 #black

# Take a step in the game
# @state: player sum and dealer first card
# @action: action to be taken by the player
#
# returns: next state and reward
def step(state, action):
	player_sum, dealer_sum = state
	reward = None

	# is this the begining state?
	if (player_sum == 0 and dealer_sum == 0):
		# get first cards
		player_sum = get_card(firstCard=True)
		dealer_sum = get_card(firstCard=True)
		return [player_sum, dealer_sum], reward

	# If the player stands the dealer plays out
	if (action == ACTION_STAND):
		while True:
			if (dealer_sum < 17 and 0 < dealer_sum):
				dealer_sum += get_card()
			else:
				break

		# dealer stands, check for winner
		if (dealer_sum < 1 or 21 < dealer_sum or dealer_sum < player_sum):
			reward = 1
		elif (player_sum < 1 or 21 < player_sum or player_sum < dealer_sum):
			reward = -1
		else:
			reward = 0

	else:
		player_sum += get_card()
		if (player_sum < 1 or 21 < player_sum):
			reward = -1

	return (player_sum, dealer_sum), reward


def monte_carlo_controll(numGames):
	wins = 0
	for i in range(numGames):
		state = (0, 0)
		reward = None
		trajectory = []
		
		while True:
			# update epsilon
			epsilon = N0/(N0 + Ns[state])

			# get probability array for actions
			probs = epsilon_greedy_policy(Q, epsilon, len(ACTIONS), state)

			# get action w.r.t. probabilities
			action = np.random.choice(np.arange(len(ACTIONS)), p=probs)

			# get next_state and reward
			next_state, reward = step(state, action)

			# keep track of states visited and action taken in this episode
			trajectory.append((state, action))

			# update number of times the state has been visited
			Ns[state] += 1
			
			# update number of times action was selected from this state
			Nas[(state, action)] += 1

			state = tuple(next_state)
			
			if (reward is not None):
				if reward == 1:
					wins += 1
				Ns[state] += 1
				break
			
		for state_action in trajectory:
			Q[state_action] = Q[state_action] + 1/Nas[state_action] * (reward - Q[state_action])

	return wins



# initialize value function - when asked to return a value for a missing key, 
# efaultdict creates the key with a value 0.0 and returns 0.0
# this is the same as initializing an array with values 0.0, but for the dict we dont need to know the size at the begining
Q = defaultdict(float)
Nas = defaultdict(int)  # Will use a tuple of ints as a key!
Ns = defaultdict(int)
N0 = 1000
wins = 0

game = monte_carlo_controll(50000)
print(game)
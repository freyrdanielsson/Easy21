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

	x = np.random.randint(0, 3)
	if (x < 1):
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

	# If the player stands the dealer plays out
	if (action == ACTION_STAND):
		while True:
			# try what happens if the dealer stands only when he is winning or drawing
			#if (dealer_sum >= player_sum or dealer_sum < 1 or 17 <= dealer_sum):
			if (17 <= dealer_sum or dealer_sum < 1):
				break
			else:
				dealer_sum += get_card()

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


def monte_carlo_controll(numGames, Qmc, N0):
	wins = 0

	Nsa = defaultdict(int)  # nr of action a picked from state s
	Ns = defaultdict(int) # nr state s visited
	c = 0

	for i in range(0, numGames):
		# get first cards
		player_sum = get_card(firstCard=True)
		dealer_sum = get_card(firstCard=True)
		state = (player_sum, dealer_sum)
		reward = None
		trajectory = []
		
		while True:
			# update number of times the state has been visited
			Ns[state] += 1

			# update epsilon
			epsilon = N0/(N0 + Ns[state])

			# get probability array for actions
			probs = epsilon_greedy_policy(Qmc, epsilon, len(ACTIONS), state)

			# get action w.r.t. probabilities
			action = np.random.choice(np.arange(len(ACTIONS)), p=probs)

			# keep track of states visited and action taken in this episode
			trajectory.append((state, action))
			
			# update number of times action was selected from this state
			Nsa[(state, action)] += 1

			# get next_state and reward
			next_state, reward = step(state, action)
			
			if (reward is not None):
				if reward == 1:
					wins += 1
				Ns[next_state] += 1
				break
			
			state = next_state
			

		for state_action in trajectory:
			Qmc[state_action] = Qmc[state_action] + 1/Nsa[state_action] * (reward - Qmc[state_action])

	return wins, Qmc


def sarsa_lambda(numgames, Qsarsa, Qmc, _lambda=0.1):
	wins = 0
	Nsa = defaultdict(int)  # nr of action a picked from state s
	E = defaultdict(float)
	Ns = defaultdict(int) # nr state s visited

	for i in range(0, numgames):
		# get first cards
		player_sum = get_card(firstCard=True)
		dealer_sum = get_card(firstCard=True)
		state = (player_sum, dealer_sum)

		# initial epsilon
		epsilon = N0/(N0 + Ns[state])

		# initial probs
		probs = epsilon_greedy_policy(Qsarsa, epsilon, len(ACTIONS), state)

		# initial action
		action = np.random.choice(np.arange(len(ACTIONS)), p=probs)

		reward = None

		while True:
			# update number of times the state has been visited
			Ns[state] += 1

			# update number of times action was selected from this state
			# for the first game this is the action found above this loop
			Nsa[(state, action)] += 1

			# get next_state and reward
			next_state, reward = step(state, action)

			# update epsilon
			epsilon = N0/(N0 + Ns[state])

			# get probability array for next action from next state
			probs = epsilon_greedy_policy(Qsarsa, epsilon, len(ACTIONS), next_state)

			# get next_action w.r.t. probabilities
			next_action = np.random.choice(np.arange(len(ACTIONS)), p=probs)

			# update delta
			delta = (0 if not reward else reward) + Qsarsa[(next_action, next_action)] - Qsarsa[(state, action)]

			# update eligibility
			E[(state, action)] += 1

			# for all state-action we have taken
			for state_action in Nsa:
				alpha = 1/Nsa[state_action]
				Qsarsa[state_action] = Qsarsa[state_action] + alpha * delta * E[state_action]
				E[state_action] = _lambda * E[state_action]

			if (reward is not None):
				if reward == 1:
					wins += 1
				Ns[next_state] += 1
				break
			
			state = next_state
			action = next_action

	mean_err = 0.0
	for state_action in Qsarsa:
		mean_err += pow(Qsarsa[state_action] - Qmc[state_action], 2)

	return wins, mean_err, Qsarsa


def plot_MC(Q):
	# For plotting: Create value function from action-value function
	# by picking the best action at each state
	state_action_values = np.zeros((12, 10))
	for player_sum in range(10, 22):
		for dealer_first in range(1, 11):
			# pick value from best action
			best = np.max([Q[((player_sum, dealer_first), 0)], Q[((player_sum, dealer_first), 1)]])
			state_action_values[player_sum-10, dealer_first-1] = best


	_, axes = plt.subplots(1, 1, figsize=(10, 8))

	fig = sns.heatmap(np.flipud(state_action_values), cmap="YlGnBu", xticklabels=range(1, 11),
							yticklabels=list(reversed(range(10, 22))))
	fig.set_ylabel('player sum', fontsize=16)
	fig.set_xlabel('dealer showing', fontsize=16)
	fig.set_title('Optimal value function', fontsize=16)
		
	plt.show()
	plt.savefig('plotValue.png')


N0 = 1000
# initialize value function - when asked to return a value for a missing key, 
# defaultdict creates the key with a value 0.0 and returns 0.0
# this is the same as initializing an array with values 0.0, but for the dict we dont need to know the size at the begining
# and it will never be to big or to small, keys must also be hashable so i use ints or tuples
Qmc = defaultdict(float) #  Will use a tuple of (state, action) as a key!

win, Qmc = monte_carlo_controll(100000, Qmc, N0)

plot_MC(Qmc)
# plot the optimal value function on a heat map, TODO change this to 3D?
# is the player so likely to win when the dealer card is 1 for there is no usable ace and he might go bust? (draw a negative card)
print(win)

''' # get lambda from 0, 0.1, ..., 1.0
lmbda = np.arange(0, 1.1, 0.1)

# to keep track of mean-squared errors for plotting later
errors = []

# For all lambdas, run sarsa lambda for 1000 episodes, calc the mean-sq error 
for l in lmbda:
	Qsarsa = defaultdict(float)
	win, mean_err, Qsarsa = sarsa_lambda(1000, Qsarsa, Qmc, l)
	errors.append(mean_err)

# print the mean-squared errors for each lambda for observation
for i in range(0, len(errors)):
	print('lambda: ',lmbda[i], ' mean-squared-error: ', errors[i]) '''
	



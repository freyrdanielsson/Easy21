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
	# Chose action 1 (stand) if (state, hit) is lower than (state, stand)
	# Chose action 0 (hit) if (state, stand) is lower than (state, hit)
	# else they are equal, so chose randomly 👍
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
			#if (dealer_sum >= player_sum or dealer_sum < 1):
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


def monte_carlo_controll(numGames, Qmc):
	wins = 0
	N0 = 1000
	Nsa = defaultdict(int)  # nr of action a picked from state s
	Ns = defaultdict(int) # nr state s visited
	c = 0

	for episode in range(0, numGames):
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


def sarsa_lambda(numgames, Qsarsa, Qmc, _lambda=0.1, calc_per_episode=False):
	wins = 0
	N0 = 1000
	Nsa = defaultdict(int)  # nr of action a picked from state s
	Ns = defaultdict(int) # nr state s visited
	mean_per_episode = []

	for episode in range(0, numgames):
		# Initialize eligibility for this episode
		E = defaultdict(float)
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

			# update delta. Note: if the next_state is terminal, Qsarsa[(next_state, next_action)] returns 0.0
			# always, because this state-value will not be updated, and is by default 0.0
			delta = (0 if not reward else reward) + Qsarsa[(next_state, next_action)] - Qsarsa[(state, action)]

			# update eligibility
			E[(state, action)] += 1

			# for all state-action we have taken during all episodes to this point
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
			
			
		if calc_per_episode:
			err = 0
			for state_action in Qmc:
				err += pow(Qsarsa[state_action] - Qmc[state_action], 2)
			mean_per_episode.append(err/len(Qmc))

			

	mean_err = 0.0
	for state_action in Qmc:
		mean_err += pow(Qsarsa[state_action] - Qmc[state_action], 2)

	return wins, mean_err/len(Qmc), mean_per_episode, Qsarsa

# Parameter: Q - is a defaultdict with key (state, action) and the value is a value for that state-action pair
# state is a tuple
# action is an int
def plot_heatMap(Q, title):
	# For plotting: Create value function from action-value function
	# by picking the best action at each state
	state_action_values = np.zeros((12, 10))
	for player_sum in range(10, 22):
		for dealer_first in range(1, 11):
			# pick value from best action
			best = np.max([Q[((player_sum, dealer_first), 0)], Q[((player_sum, dealer_first), 1)]])
			state_action_values[player_sum-10, dealer_first-1] = best


	_, axes = plt.subplots(1, 1, figsize=(10, 8))
	plt.figure()
	fig = sns.heatmap(np.flipud(state_action_values), cmap="YlGnBu", xticklabels=range(1, 11),
							yticklabels=list(reversed(range(10, 22))))
	fig.set_ylabel('player sum', fontsize=16)
	fig.set_xlabel('dealer showing', fontsize=16)
	fig.set_title(title, fontsize=16)
		
	plt.show()
	plt.savefig(title + '.png')

def plotError_perEpisode(_lambda, meanError):
	plt.figure()
	for i in range(0, len(_lambda)):
		plt.step(range(0,1000), meanError[i], label = 'lambda ' + str(_lambda[i]))
		plt.xlabel('Episodes')
		plt.ylabel('Mean squared error')
		plt.legend()
	
	plt.savefig('lambda.png')

# initialize value function - when asked to return a value for a missing key, 
# defaultdict creates the key with a value 0.0 and returns 0.0
# this is the same as initializing an array with values 0.0, but for the dict we dont need to know the size at the begining
# and it will never be to big or to small, keys must also be hashable so i use ints or tuples
Qmc = defaultdict(float)
win, Qmc = monte_carlo_controll(50000, Qmc)

# plot the optimal value function on a heat map, TODO change this to 3D?
plot_heatMap(Qmc, 'MC_Controll_V(s)_50k_episodes')

# get lambda from 0, 0.1, ..., 1.0
lmbda = np.arange(0.0, 1.1, 0.1)
# to keep track of mean-squared errors for plotting later
errors = []
errors_perEpisode = []
# For all lambdas, run sarsa lambda for 1000 episodes, calc the mean-sq error 
for l in lmbda:
	Qsarsa = defaultdict(float)
	win, mean_err, meanPerEpisode, Qsarsa = sarsa_lambda(1000, Qsarsa, Qmc, _lambda=l, calc_per_episode=True)
	errors.append(mean_err)
	errors_perEpisode.append(meanPerEpisode)

plotError_perEpisode([0.0, 1.0], [errors_perEpisode[0], errors_perEpisode[10]])


plot_heatMap(Qsarsa, 'Sarsa_lambda')

	



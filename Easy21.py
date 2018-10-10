import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
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
# returns: (next_state) and reward
def step(state, action):
	player_sum, dealer_sum = state
	reward = None

	# If the player stands the dealer plays out
	if (action == ACTION_STAND):
		while True:
			# try what happens if the dealer stands only when he is winning or drawing
			# this makes the dealer much much better! 🧠
			#if (dealer_sum >= player_sum or dealer_sum < 1):
			if (17 <= dealer_sum or dealer_sum < 1):
				break
			else:
				dealer_sum += get_card()

		# dealer stands, check for winner
		# Nota: we don't need to check if the player is bust at this point we know he isn't
		if (dealer_sum < 1 or 21 < dealer_sum or dealer_sum < player_sum):
			reward = 1
		elif (player_sum < dealer_sum):
			reward = -1
		else:
			reward = 0

	else:
		player_sum += get_card()
		if (player_sum < 1 or 21 < player_sum):
			reward = -1

	return (player_sum, dealer_sum), reward

# Parameters: 
# @numGames: number of episodes to run - int
# @Qmc: action-value function to improve, typically initialized with 0.0 - defaultdict(float)
# 
# returns:
# @wins: nmbr of wins by the agent - int
# @Qmc: improved action-value function over numgames episodes - defaultdict(float)
def monte_carlo_controll(numGames, Qmc):
	wins = 0
	N0 = 1000
	Nsa = defaultdict(int)  # nr times action a was picked from state s
	Ns = defaultdict(int) # nr times state s visited

	for episode in range(0, numGames):
		# get first cards & initialize first state
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

			# get probability matrix for actions
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
			
		# Update the action value function after each episode
		for state_action in trajectory:
			Qmc[state_action] = Qmc[state_action] + 1/Nsa[state_action] * (reward - Qmc[state_action])

	return wins, Qmc

# Parameters:
# @numgames: number of episodes to run - int
# @Qsarsa: action-value function to optimize, typically comes initialized with 0.0 - defaultdict(float)
# @Qmc: supposedely an optimal action-value function
# @_lambda: value used in updating the value function @Qsarsa - float
# @calc_per_episode: whether to calculate the mean square error for each episode or not - boolean
#
# returns:
# @wins: number of games won by agent - int
# @mean_err: mean squared error over all episodes - array
# @mean_per_episode: mean squared error for each episode - array
# @Qsarsa: improved action-value function over @numgames episodes - defaultdict(float)
def sarsa_lambda(numgames, Qsarsa, Qmc, _lambda=0.1, calc_per_episode=False):
	wins = 0
	N0 = 1000
	Nsa = defaultdict(int)
	Ns = defaultdict(int)
	mean_per_episode = []

	for episode in range(0, numgames):
		# Initialize eligibility for this episode
		E = defaultdict(float)

		player_sum = get_card(firstCard=True)
		dealer_sum = get_card(firstCard=True)
		state = (player_sum, dealer_sum)

		# initial epsilon
		epsilon = N0/(N0 + Ns[state])

		# initial probability matrix
		probs = epsilon_greedy_policy(Qsarsa, epsilon, len(ACTIONS), state)

		# initial action
		action = np.random.choice(np.arange(len(ACTIONS)), p=probs)

		reward = None

		while True:
			# update number of times the state has been visited
			Ns[state] += 1

			# update number of times action was selected from this state
			Nsa[(state, action)] += 1

			# get next_state and reward by taking the action
			next_state, reward = step(state, action)

			# update epsilon
			epsilon = N0/(N0 + Ns[state])

			# get probability matrix for next action from next state
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


# Parameter:
# @Q: is a defaultdict with key (state, action) and the value is a value for that state-action pair
# state is a tuple
# action is an int
#
# Plots a heat map for the best action in each state from player_sum = [10, 21], dealer_first = [1, 10]
def plot_heatMap(Q, title):
	# For plotting: Create value function from action-value function
	# by picking the best action at each state
	state_action_values = np.zeros((21, 10))
	for player_sum in range(1, 22):
		for dealer_first in range(1, 11):
			# pick value from best action
			best = np.max([Q[((player_sum, dealer_first), 0)], Q[((player_sum, dealer_first), 1)]])
			state_action_values[player_sum-1, dealer_first-1] = best


	_, axes = plt.subplots(1, 1, figsize=(10, 8))
	plt.figure()
	fig = sns.heatmap(np.flipud(state_action_values), cmap="YlGnBu", xticklabels=range(1, 11),
							yticklabels=list(reversed(range(1, 22))))
	fig.set_ylabel('player sum', fontsize=16)
	fig.set_xlabel('dealer showing', fontsize=16)
	fig.set_title('Optimal Value Function', fontsize=16)
		
	plt.savefig(title + '_heatmap.png')

	# 3d plot

	x_scale=1
	y_scale=1.5
	z_scale=1

	scale=np.diag([x_scale, y_scale, z_scale, 1.0])
	scale=scale*(1.0/scale.max())
	scale[3,3]=1.0

	def short_proj():
		return np.dot(Axes3D.get_proj(ax), scale)


	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.get_proj=short_proj

	Y = np.arange(1, 22)
	X = np.arange(1, 11)
	X, Y = np.meshgrid(X, Y)

	ax.plot_wireframe(X, Y, state_action_values, rstride=1, cstride=1)
	plt.ylabel('Player sum')
	plt.xlabel('Dealer showing')
	plt.title('Optimal Value Function')
	plt.savefig(title + '_3d.png')

	


# Parameters:
# _lambda: array containing values of lambda 
# meanError: array containing the mean error for each episode
#
# Plots learning curve of each value in _lambda
def plotError_perEpisode(_lambda, meanError):
	plt.figure()
	for i in range(0, len(_lambda)):
		plt.step(range(0,1000), meanError[i], label = 'lambda ' + str(_lambda[i]))
		plt.xlabel('Episodes')
		plt.ylabel('Mean squared error')
		plt.legend()
	
	plt.savefig('lambda_episode.png')



# Parameters:
# _lambda: array containing values of lambda 
# meanError: array containing the mean error after all episodes with the lambda value in _lambda
#
# Plots the mean-squared error for each value in _lambda
def plotError_perLambda(_lambda, meanError):
	plt.figure()
	plt.plot(_lambda, meanError, label=None)
	plt.xlabel('lambda')
	plt.ylabel('Mean squared error')
	plt.legend()
	
	plt.savefig('lambda_mean.png')

def main():
	# initialize value function - when asked to return a value for a missing key, 
	# defaultdict creates the key with a value 0.0 and returns 0.0
	# this is the same as initializing an array with values 0.0, but for the dict we dont need to know the size at the begining
	# and it will never be to big or to small, keys must also be hashable so i use ints or tuples
	Qmc = defaultdict(float)
	win, Qmc = monte_carlo_controll(500000, Qmc)

	# plot the optimal value function on a heat map, TODO change this to 3D?
	plot_heatMap(Qmc, 'MC Control')

	# generate lambda from 0, 0.1, ..., 1.0
	lmbda = np.arange(0.0, 1.1, 0.1)
	# to keep track of mean-squared errors per lambda and per episode for plotting later
	errors = []
	errors_perEpisode = []
	# For all lambdas, run sarsa lambda for 1000 episodes, calc the mean-square error
	# For lambda 0.0 and 1.0 also count mean-square error for each episode
	for l in lmbda:
		perEpisode = (l == 0.0) or (l == 1.0)
		Qsarsa = defaultdict(float)
		win, mean_err, meanPerEpisode, Qsarsa = sarsa_lambda(1000, Qsarsa, Qmc, _lambda=l, calc_per_episode=perEpisode)
		errors.append(mean_err)
		errors_perEpisode.append(meanPerEpisode)

	# Plot mean square error against episodes for lambda 0.0 and lambda 1.0
	plotError_perEpisode([0.0, 1.0], [errors_perEpisode[0], errors_perEpisode[10]])
	# Plot mean square error against lambda for lambda = [0.0,...1.0]
	plotError_perLambda(lmbda, errors) 


	# Code bellow plots value function for Qsarsa lambda = 1.0, this was not asked for, just for me
	# Qsarsa = defaultdict(float)
	#win, mean_err, meanPerEpisode, Qsarsa = sarsa_lambda(50000, Qsarsa, Qmc, _lambda=1.0)
	#plot_heatMap(Qsarsa, 'Sarsa_lambda')

# Run the main function of the assignment
if __name__ == '__main__':
	main()

	



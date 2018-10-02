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

# policy for player
POLICY_PLAYER = np.zeros(22)
for i in range(12, 20):
    POLICY_PLAYER[i] = ACTION_HIT
POLICY_PLAYER[20] = ACTION_STAND
POLICY_PLAYER[21] = ACTION_STAND

# function form of target policy of player
def target_policy_player(state):
	player_sum, dealer_sum = state
	return POLICY_PLAYER[player_sum]

# function form of behavior policy of player
# all actions tried with non-zero probability
def behavior_policy_player(state):
    if np.random.binomial(1, 0.5) == 1:
        return ACTION_STAND
    return ACTION_HIT

# policy for dealer
POLICY_DEALER = np.zeros(22)
for i in range(12, 17):
    POLICY_DEALER[i] = ACTION_HIT
for i in range(17, 22):
    POLICY_DEALER[i] = ACTION_STAND

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

	if (player_sum == 0 and dealer_sum == 0):
		# get first cards
		player_sum = get_card(firstCard=True)
		dealer_sum = get_card(firstCard=True)
		return [player_sum, dealer_sum], reward

	# If the player stands the dealer plays out
	if (action == ACTION_STAND):
		while True:
			# get action based on current sum
			dealer_action = POLICY_DEALER[dealer_sum]
			if (dealer_action == ACTION_STAND):
				break
			dealer_sum += get_card()
			if (21 < dealer_sum):
				break

		# dealer stands, check for winner
		if (dealer_sum < 0 or 21 < dealer_sum or dealer_sum < player_sum):
			reward = 1
		elif (dealer_sum == player_sum):
			reward = 0
		else:
			reward = -1
	else:
		player_sum += get_card()
		if (player_sum < 1 or 21 < player_sum):
			reward = -1

	return [player_sum, dealer_sum], reward


	def monte_carlo_controll(episodes, policy_player):
		for game in episodes:
			state = (0, 0)
			while True:
				#get action
				action = policy_player(state)
				step(state, action)
			break


# play a game
# @policy_player: specify policy for player
# @initial_state: [whether player has a usable Ace, sum of player's cards, one card of dealer]
# @initial_action: the initial action
def play(policy_player, initial_state=None, initial_action=None):

	# initiate state
	state = [0, 0]

	# initial action
	
	while True:
		action = policy_player(state)
		state, reward = step(state, action)
		if (reward is not None):
			return state, reward


# initialize value function - when asked to return a value for a missing key, 
# efaultdict creates the key with a value 0.0 and returns 0.0
# this is the same as initializing an array with values 0.0, but for the dict we dont need to know the size at the begining
Q = defaultdict(float)
Nas = defaultdict(int)  # Will use a tuple of ints as a key!
Ns = defaultdict(int)
N0 = 100





''' initial_state = [
	np.random.choice(range(12, 22)),
	np.random.choice(range(1, 11))	] '''

# value function sem er [states, actions]
#print(np.zeros((3, 3, 2, 2)))

''' wins = 0
draws = 0
for i in range(0, 50):
	state, reward = play(behavior_policy_player)
	print(state)

	if (reward == 1):
		wins += 1
	elif (reward == 0):
		draws += 1

print("wins: ", wins, " draws: ", draws , " loses: ", 50-wins-draws) '''
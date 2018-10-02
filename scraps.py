import numpy as np
from collections import defaultdict



''' 
i = defaultdict(int)
f = defaultdict(float)

print('i: ', i, ' f: ', f)

i['a'] = 100
f['a'] = 100.0

print('i: ', i.items(), ' f: ', f.items())

i['b']
f['b']

print('i: ', i.items(), ' f: ', f.items())

i.get('c')
f.get('c')

print('i: ', i.items(), ' f: ', f.items())

# can i now get b and c from both i and f? Expecting give 0 and 0.0
print('i: ', i.get('b'), i.get('c'), ' f: ', f.get('b'), f.get('c'))

# Results: .get('c') returnes None but .get('b') returnes 0/0.0 so .get does not insert a key value pair but using i['..'] does (Y) '''

''' Q = defaultdict(float)
state = ('playersum', 'dealersum')
action = 'action'
state_action = (state, action) # (('playersum', 'dealersum'), action)
value = 1.0
Q[state_action] = value
print(Q[state_action]) '''

a = defaultdict(int)
a[(1,2)]
print(a[(1,2)])
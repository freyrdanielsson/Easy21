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

state = ('a', 'b')

print(state[1])

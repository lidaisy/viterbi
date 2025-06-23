import numpy as np

def viterbi(Ps0, transition, emission, time_steps, values, observed):
    pi = np.zeros((time_steps, values))
    phi = np.zeros((time_steps, values))
    s = np.zeros(time_steps)

    # base case
    for j in range(values):
        pi[0,j] = Ps0[j] * emission[j, observed[0]]

    # Using dynamic programming
    for k in range(1, time_steps):
        for j in range(values):
            Pfunc = lambda z: transition[z,j]
            P = np.array(list(map(Pfunc, range(values))))
            pi_P = pi[k-1]*P
            pi[k, j] = emission[j, observed[k]] * np.max(pi_P)
            phi[k, j] = np.argmax(pi_P)
    
    print(pi)
    print(phi)

    # Find states in reverse
    s[time_steps - 1] = np.argmax(pi[time_steps - 1])

    for k in range(time_steps - 1, 0, -1):
        print(k)
        s[k - 1] = phi[k, int(s[k])]
    return s

'''
transition: P(s_t | s_{t-1})
            s_t     0     1
s_{t-1} 0
s_{t-1} 1

emission: P(o_t | s_t)
        o_t     0     1
s_t 0
s_t 1
'''

Ps0 = np.array([0.6, 0.4])
transition = np.array([[0.8, 0.2], [0.3, 0.7]])
emission = np.array([[0.8, 0.2], [0.1, 0.9]])
time_steps = 4
values = 2
observed = np.array([1, 1, 1, 0])

res = viterbi(Ps0, transition, emission, time_steps, values, observed)
print(res)

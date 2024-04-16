import numpy as np

def policy_evaluation(env, policy, df, theta):
    nS = env.observation_space.n
    nA = env.action_space.n
    v = np.zeros(nS)
    delta = np.inf
    while (delta > theta):
        for s in range(nS):
            v_previous = v[s]
            sum = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, new_state, reward, _ in env.unwrapped.P[s, a]:
                    sum += action_prob * prob * (reward + df * v[new_state])
            v[s] = sum
            delta = max(delta, np.abs(v_previous - v[s]))
    return v




import numpy as np

N = 0
E = 1
S = 2
W = 3

np.random.seed(3407 + 42)

class GridWorld:
    def __init__(self, target, directionP):
        MAX_X = MAX_Y = 6
        self.MAX_Position = MAX_X * MAX_Y  # 状态总数
        self.nA = 4                        # 动作总数
        self.target = target               # 终止状态
        self.P = {}                        # 状态转移表


        for y in range(MAX_Y):
            for x in range(MAX_X):
                position = y * MAX_X + x
                self.P[position] = {a: [] for a in range(self.nA)}
                terminal = position in self.target
                reward = 0.0 if terminal else -1.0

                if terminal:
                    for a in range(self.nA):
                        self.P[position][a] = [(1.0, position, reward, True)]
                else:
                    next_states = [
                        position if y == 0 else position - MAX_X,  # 北
                        position if x == MAX_X-1 else position + 1,  # 东
                        position if y == MAX_Y-1 else position + MAX_X,  # 南
                        position if x == 0 else position - 1  # 西
                    ]
                    for a, next_s in enumerate(next_states):
                        done = next_s in self.target
                        self.P[position][a] = [(directionP[a], next_s, reward, done)]
        self.isd = np.ones(self.MAX_Position) / self.MAX_Position
        self.action_space = range(self.nA)
        self.observation_space = range(self.MAX_Position)
        self.s = None

    def reset(self):
        self.s = np.random.choice(self.MAX_Position, p=self.isd)
        return self.s

    def step(self, action):
        transitions = self.P[self.s][action]

        probs = np.array([t[0] for t in transitions])
        probs /= probs.sum()  
        #多加了一个归一化设置，不然会因为dtype有奇怪的bug
        i = np.random.choice(len(transitions), p=probs)
        probability, next_state, reward, targets = transitions[i]
        self.s = next_state
        return next_state, reward, targets, {}

def mc_learning(gw, num_episodes=8, discount=0.9, epsilon=0.001, method='first-visit'):
    Q = np.zeros((gw.MAX_Position, gw.nA))  
    N = np.zeros((gw.MAX_Position, gw.nA))
    policy = np.zeros(gw.MAX_Position, dtype=int)
    round = 0
    for _ in range(num_episodes):
        episode = []
        state = gw.reset()
        target = False
        round += 1
        while not target:

            

            #逐步提高随机性的设置，其实固定0.1也可以，但是会多大概50000次：
            epsilon=min(0.1, epsilon * 1.005)

            if np.random.rand() < epsilon:
                action = np.random.choice(gw.action_space)
            else:
                action = np.argmax(Q[state])
            next_state, reward, target, _ = gw.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        G = 0
        visited = set()  #用于First-Visit
        
        for t in reversed(range(len(episode))):
            s_t, a_t, r_t = episode[t]
            G = discount * G + r_t
            
            # First-Visit：仅首次出现的(s,a)更新
            if method == 'first-visit':
                if (s_t, a_t) not in visited:
                    N[s_t, a_t] += 1
                    Q[s_t, a_t] += (G - Q[s_t, a_t]) / N[s_t, a_t]
                    visited.add((s_t, a_t))
            
            # Every-Visit：所有(s,a)均更新
            elif method == 'every-visit':
                N[s_t, a_t] += 1
                Q[s_t, a_t] += (G - Q[s_t, a_t]) / N[s_t, a_t]
    
    policy = np.argmax(Q, axis=1)
    return policy, Q, round


def td_zero(gw, num_episodes=8, alpha=0.1, discount=0.9, epsilon=0.1):
    Q = np.zeros((gw.MAX_Position, gw.nA)) 
    policy = np.zeros(gw.MAX_Position, dtype=int)
    round = 0
    for _ in range(num_episodes):
        state = gw.reset()
        target = False
        action = np.argmax(Q[state]) if np.random.rand() > epsilon else np.random.choice(gw.action_space)
        round += 1
        while not target:

            
            #逐步降低随机性的设置，对TD策略这个设置影响不大：
            epsilon=max(0.001, epsilon * 0.995)

            next_state, reward, target, _ = gw.step(action)
            
            next_action = np.argmax(Q[next_state]) if np.random.rand() > epsilon else np.random.choice(gw.action_space)
            
            # TD(0) 更新：Q(s,a) += α * [r + γ*Q(s',a') - Q(s,a)]
            td_target = reward + discount * Q[next_state][next_action] * (not target)
            Q[state][action] += alpha * (td_target - Q[state][action])
            
            state, action = next_state, next_action
    
    policy = np.argmax(Q, axis=1)
    return policy, Q, round
if __name__ == '__main__':
    gw = GridWorld(target={1, 35}, directionP=[0.25, 0.25, 0.25, 0.25])
    method1='first-visit'
    method2='every-visit'
    
    policy_mc, Q_mc, round_mc = mc_learning(gw, method=method1)
    print("MC学习策略矩阵（6x6），使用", method1)
    print(policy_mc.reshape(6, 6))
    print(round_mc)

    policy_mc, Q_mc, round_mc = mc_learning(gw, method=method2)
    print("MC学习策略矩阵（6x6），使用", method2)
    print(policy_mc.reshape(6, 6))
    print(round_mc)

    policy_q, Q_q, round_q = td_zero(gw)
    print("\nQ-Learning策略矩阵（6x6）：")
    print(policy_q.reshape(6, 6))
    print(round_q)
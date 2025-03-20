import numpy as np

N = 0
E = 1
S = 2
W = 3

class GridWorld:
    def __init__(self, target={1, 35}, directionP=[0.25, 0.25, 0.25, 0.25]):
        # 初始化网格世界参数
        MAX_X = MAX_Y = 6
        self.MAX_Position = MAX_X * MAX_Y  # 状态总数
        self.nA = 4                        # 动作总数
        self.target = target               # 终止状态
        self.P = {}                         # 状态转移表


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
        self.isd = np.ones(self.MAX_Position) / self.MAX_Position  # 均匀初始分布
        self.action_space = range(self.nA)          # 动作空间
        self.observation_space = range(self.MAX_Position)  # 状态空间
        self.s = None  # 当前状态

    def reset(self):
        """重置环境，返回初始状态"""
        self.s = np.random.choice(self.MAX_Position, p=self.isd)
        return self.s

    def step(self, a):
        """执行动作，返回 (next_state, reward, done, info)"""
        transitions = self.P[self.s][a]
        i = np.random.choice(len(transitions), p=[t[0] for t in transitions])
        probability, next_state, reward, targets = transitions[i]
        self.s = next_state
        return next_state, reward, targets, {}
'''-------------------------------------'''     
def value_iteration(gw, theta=0.00001, discount = 0.9):

    V = np.zeros(gw.MAX_Position)
    round = 0
    while True:
        round += 1
        delta = 0 
        for position in range(gw.MAX_Position):
            A = np.zeros(gw.nA)
            for direction in range(gw.nA):
                for probability, next_state, reward, targets in gw.P[position][direction]:
                    if targets:
                        A[direction] += probability *(reward) 
                    else: 
                        A[direction] += probability *(reward + discount*V[next_state]) 
            best_action = np.max(A)
            delta = max(delta, np.abs(best_action - V[position]))
            V[position] = best_action
        if delta < theta:
            break
    
    policy = np.zeros([gw.MAX_Position])
    for position in range(gw.MAX_Position):
        A2 = np.zeros(gw.nA)
        for direction in range(gw.nA):
            for probability, next_state, reward, targets in gw.P[position][direction]:
                if targets:
                    A2[direction] += probability *(reward) 
                else: 
                    A2[direction] += probability *(reward + discount*V[next_state]) 
        best_action2 = np.argmax(A2)
        policy[position] = best_action2
    return policy, V, round

'''-----------------------------------'''
def policy_iteration(gw, theta=0.00001, discount=0.9):
    policy = np.zeros(gw.MAX_Position)
    V = np.zeros(gw.MAX_Position)
    round = 0
    policy_stable = False

    while not policy_stable:
        round += 1
        while True:
            delta = 0
            for position in range(gw.MAX_Position):
                if position in gw.target:  
                    new_v = 0.0
                else:
                    direction = policy[position]
                    probability, next_state, reward, targets = gw.P[position][direction][0]
                    if targets:
                        new_v = probability*(reward)
                    else:    
                        new_v = probability*(reward + discount * V[next_state])
                delta = max(delta, abs(V[position] - new_v))
                V[position] = new_v
            if delta < theta:
                break
        policy_stable = True

        for position in range(gw.MAX_Position):
            if position in gw.target:
                continue
            old_action = policy[position]
            Q = []
            for direction in range(gw.nA):
                probability, next_state, reward, targets = gw.P[position][direction][0]
                if targets:
                    q_val = probability*(reward)
                else:    
                    q_val = probability*(reward + discount * V[next_state])
                Q.append(q_val)
            best_action2 = np.argmax(Q)
            if best_action2 != old_action:
                policy_stable = False #若有变化继续迭代

            #如果所有状态的动作均未改变，则策略稳定，停止迭代。
            policy[position] = best_action2

    return policy, V, round


if __name__ == '__main__':
    gw = GridWorld(target={1,35}, directionP = [0.25,0.25,0.25,0.25])

    policy_vi, V_vi, round_vi= value_iteration(gw)
    print("值迭代策略矩阵（6x6）：")
    print(policy_vi.reshape(6,6))
    print(V_vi)
    print(round_vi)

    policy_pi, V_pi, round_pi = policy_iteration(gw)
    print("\n策略迭代策略矩阵（6x6）：")
    print(policy_pi.reshape(6,6))
    print(V_pi)
    print(round_pi)

    assert np.allclose(V_vi, V_pi, atol=0.1), "价值函数不一致"
    print("\n验证通过：两种算法收敛到相同策略")
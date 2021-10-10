import matplotlib.pyplot as plt
import time
import numpy as np
from cv2 import norm

import wandb
from collections import defaultdict

from rooms import RoomsEnv
from adaptive_horizons import *

class PolicyIteration:
    def __init__(self, env, gamma, planning_horizon):
        self.env = env
        self.gamma = gamma
        self.planning_horizon = planning_horizon
        self.num_generator_queries = 0
        self.V = np.zeros(self.env.num_states)
        self.pi = np.zeros(self.env.num_states, dtype=int)
        self.past_Vs = []
        self.past_V_stars = []
        self.past_num_queries = []
        self.current_horizon = 1
        self.V_star, self.pi_star = self.run_value_iteration()

        self.local_V_star, self.local_pi_star = self.run_local_value_iteration()
        self.local_V_star_extended = np.zeros(self.env.num_states)
        self.local_pi_star_extended = np.zeros(self.env.num_states, dtype=int)
        for s_id in range(self.env.num_states):
            ls_id = self.env.state_to_local_state_map[s_id]
            self.local_V_star_extended[s_id] = self.local_V_star[ls_id]
            self.local_pi_star_extended[s_id] = self.local_pi_star[ls_id]

        self.fig, self.axs = plt.subplots(2, 2)
        self.x_sub = 0
        self.y_sub = 0
        self.used_horizons = np.zeros(self.env.num_states)

        # self.render_value(self.V_star)
        # self.y_sub += 1
        # self.render_value(self.local_V_star_extended)
        # plt.show()

    def render_value(self, value_func):
        mat = np.zeros((self.env.rows, self.env.cols)) - 1
        for s_id, state in enumerate(self.env.idx_state_map):
            mat[state] = value_func[s_id]
        self.axs[self.x_sub, self.y_sub].imshow(mat)

    def run_local_value_iteration(self, max_num_iterations=1000000, tol=0.001, update_num_queries=False):
        V = np.zeros(self.env.num_local_states)
        new_V = np.zeros(self.env.num_local_states)
        policy = np.zeros(self.env.num_local_states, dtype=int)
        for t in range(max_num_iterations):
            V = new_V.copy()
            for s_id in range(self.env.num_local_states):
                Q = np.zeros(self.env.num_actions)
                for action in range(self.env.num_actions):
                    state_action_id = self.env.local_state_action_idx_map[(s_id, action)]
                    Q[action] = self.env.local_r[state_action_id] + self.gamma * np.dot(self.env.local_p[state_action_id], V)
                new_V[s_id] = np.max(Q)
                policy[s_id] = np.argmax(Q)
            if np.max(np.abs(new_V - V)) <= tol:
                return new_V, policy
        return None

    def run_value_iteration(self, max_num_iterations=1000000, tol=0.001, update_num_queries=False):
        V = np.zeros(self.env.num_states)
        new_V = np.zeros(self.env.num_states)
        policy = np.zeros(self.env.num_states, dtype=int)
        for t in range(max_num_iterations):
            V = new_V.copy()
            if update_num_queries:
                self.V = new_V.copy()
            self.past_V_stars.append(V)

            for s_id, state in enumerate(self.env.idx_state_map):
                Q = np.zeros(self.env.num_actions)
                if type(self.planning_horizon) == str and self.planning_horizon.startswith('VI') and update_num_queries:
                    effective_horizon = int(self.planning_horizon[-1])
                    Q = self.forward_backward_dp(state, effective_horizon, update_num_queries).copy()
                else:
                    for action in range(self.env.num_actions):
                        state_action_id = self.env.state_action_idx_map[(state, action)]
                        Q[action] = self.env.r[state_action_id] + self.gamma * np.dot(self.env.p[state_action_id], V)
                        if update_num_queries:
                            self.num_generator_queries += 1
                new_V[s_id] = np.max(Q)
                policy[s_id] = np.argmax(Q)
            if update_num_queries:
                wandb.log({'|V_star - V_pi|': np.max(np.abs(new_V - self.V_star)),
                           '|V_star - V_pi|avg': np.average(np.abs(new_V - self.V_star)),
                           '|V_t - V_{t-1}|': np.max(np.abs(new_V - V)),
                           'num_generator_queries': self.num_generator_queries,
                           'iteration': t})
            if np.max(np.abs(new_V - V)) <= tol:
                self.past_V_stars.append(new_V)
                return new_V, policy
        return None

    def policy_evaluation(self, iter, max_num_iterations=1000, tol=0.001, update_num_queries=True):
        V = np.zeros(self.env.num_states)
        new_V = np.zeros(self.env.num_states)
        for t in range(max_num_iterations):
            V = new_V.copy()
            for s_id, state in enumerate(self.env.idx_state_map):
                state_action_id = self.env.state_action_idx_map[(state, self.pi[s_id])]
                new_V[s_id] = self.env.r[state_action_id] + self.gamma * np.dot(self.env.p[state_action_id], V)
            if update_num_queries:
                self.num_generator_queries += self.env.num_states
            if np.max(np.abs(new_V - V)) <= tol:
                wandb.log({'|V_star - V_pi|': np.max(np.abs(new_V - self.V_star)),
                           '|V_star - V_pi|avg': np.average(np.abs(new_V - self.V_star)),
                           '|V_t - V_{t-1}|': np.max(np.abs(new_V - self.V)),
                           'num_generator_queries': self.num_generator_queries,
                           'iteration': iter})
                self.V = new_V.copy()
                return

    def forward_backward_dp(self, state, effective_horizon, update_num_queries):
        # forward pass - discover reachable states
        S = discover_reachable_states(state, effective_horizon, self.env)
        if True:#type(self.planning_horizon) == str and self.planning_horizon.startswith('Discover'):
            thresh = 50
            sum_sizes = 0
            for h in range(effective_horizon + 1):
                sum_sizes += len(S[h])
                if sum_sizes >= thresh:
                    effective_horizon = h
                    break

        # backward pass - compute V
        temp_V = np.zeros((effective_horizon + 1, self.env.num_states))
        for last_state in S[effective_horizon]:
            last_state_id = self.env.state_idx_map[last_state]
            temp_V[effective_horizon, last_state_id] = self.V[last_state_id]
        for h in range(effective_horizon - 1, -1, -1):
            for prev_state in S[h]:
                prev_state_id = self.env.state_idx_map[prev_state]
                Q = np.zeros(self.env.num_actions)
                for action in range(self.env.num_actions):
                    state_action_id = self.env.state_action_idx_map[(prev_state, action)]
                    Q[action] = self.env.r[state_action_id] + self.gamma * np.dot(self.env.p[state_action_id], temp_V[h + 1])
                    if update_num_queries:
                        self.num_generator_queries += 1
                temp_V[h, prev_state_id] = np.max(Q)
        return Q

    def policy_improvement(self, t, update_num_queries=True):
        Q_improved_values = np.zeros((self.env.num_states, self.env.num_actions))
        for s_id, state in enumerate(self.env.idx_state_map):
            effective_horizon = get_current_planning_horizon(self.planning_horizon, 0, state, self, t)
            Q_improved_values[s_id] = self.forward_backward_dp(state, effective_horizon, update_num_queries).copy()

        # if t % 4 == 0:#t > 1 and t % 5 == 0:
        #     V_improved_values = np.max(Q_improved_values, axis=1).copy()
        #     pivot_V = self.V_star
        #     #dists = np.abs(pivot_V - V_improved_values)
        #     # self.axs[self.x_sub, self.y_sub].hist(dists)
        #     if t == 0:
        #         self.render_value(self.V_star)
        #     else:
        #         dists = np.abs(self.past_Vs[-1] - self.past_Vs[0])
        #         self.render_value(dists)#self.used_horizons)
        #     self.y_sub += 1
        #     if self.x_sub == 1 and self.y_sub == 2:
        #         plt.show()
        #     if self.y_sub == 3:
        #         self.x_sub = 1
        #         self.y_sub = 0

        for s_id, state in enumerate(self.env.idx_state_map):
            self.used_horizons[s_id] = 1.0 / 8

        if type(self.planning_horizon) == str:
            if self.planning_horizon.startswith('TryStar'):
                effective_horizon = int(self.planning_horizon[-1])
                V_improved_values = np.max(Q_improved_values, axis=1).copy()
                pivot_V = self.V_star
                dists = pivot_V - V_improved_values
                indices_to_contract = np.where(dists > (self.gamma ** effective_horizon) * np.max(self.V_star - self.V))[0]
                for s_id in indices_to_contract:
                    state = self.env.idx_state_map[s_id]
                    Q_improved_values[s_id] = self.forward_backward_dp(state, effective_horizon, update_num_queries).copy()
            elif self.planning_horizon.startswith('CON2q') and t > 1:
                V_improved_values = np.max(Q_improved_values, axis=1).copy()
                pivot_V = self.V_star
                dists = np.abs(pivot_V - V_improved_values)
                params = planning_horizon.split(":")[1].split(",")
                for h, q in [(2, float(params[0])), (4, float(params[1])), (8, float(params[2]))]:
                    quantile = np.quantile(dists, q)
                    indices_to_contract = np.where(dists > quantile)[0]
                    for s_id in indices_to_contract:
                        state = self.env.idx_state_map[s_id]
                        Q_improved_values[s_id] = self.forward_backward_dp(state, h, update_num_queries).copy()
                        V_improved_values[s_id] = np.max(Q_improved_values[s_id])
                        dists[s_id] = np.abs(pivot_V[s_id] - V_improved_values[s_id])
            elif self.planning_horizon.startswith('CON2') and t > 1:
                V_improved_values = np.max(Q_improved_values, axis=1).copy()
                pivot_V = self.V_star
                dists = np.abs(pivot_V - V_improved_values)
                params = planning_horizon.split(":")[1].split(",")
                quantile = np.quantile(dists, float(params[0]))
                for h in [2, 4, 8]:
                    indices_to_contract = np.where(dists > quantile)[0]
                    for s_id in indices_to_contract:
                        state = self.env.idx_state_map[s_id]
                        Q_improved_values[s_id] = self.forward_backward_dp(state, h, update_num_queries).copy()
                        V_improved_values[s_id] = np.max(Q_improved_values[s_id])
                        dists[s_id] = np.abs(pivot_V[s_id] - V_improved_values[s_id])
            elif self.planning_horizon.startswith('AppVstar'):
                V_improved_values = np.max(Q_improved_values, axis=1).copy()
                params = planning_horizon.split(":")[1].split(",")
                past_V_star_idx = int(params[3])
                pivot_V = self.past_V_stars[- past_V_star_idx]
                dists = np.abs(pivot_V - V_improved_values)
                for h, q in [(2, float(params[0])), (4, float(params[1])), (8, float(params[2]))]:
                    quantile = np.quantile(dists, q)
                    indices_to_contract = np.where(dists > quantile)[0]
                    for s_id in indices_to_contract:
                        state = self.env.idx_state_map[s_id]
                        Q_improved_values[s_id] = self.forward_backward_dp(state, h, update_num_queries).copy()
                        V_improved_values[s_id] = np.max(Q_improved_values[s_id])
                        dists[s_id] = np.abs(pivot_V[s_id] - V_improved_values[s_id])
            elif self.planning_horizon.startswith('Loc'):
                V_improved_values = np.max(Q_improved_values, axis=1).copy()
                params = planning_horizon.split(":")[1].split(",")
                pivot_V = self.local_V_star_extended
                dists = np.abs(pivot_V - V_improved_values)
                for h, q in [(2, float(params[0])), (4, float(params[1])), (8, float(params[2]))]:
                    quantile = np.quantile(dists, q)
                    indices_to_contract = np.where(dists > quantile)[0]
                    for s_id in indices_to_contract:
                        state = self.env.idx_state_map[s_id]
                        Q_improved_values[s_id] = self.forward_backward_dp(state, h, update_num_queries).copy()
                        V_improved_values[s_id] = np.max(Q_improved_values[s_id])
                        dists[s_id] = np.abs(pivot_V[s_id] - V_improved_values[s_id])

        prev_pi = self.pi.copy()
        self.pi = np.argmax(Q_improved_values, axis=1).copy()
        return not np.array_equal(self.pi, prev_pi)

    def run_policy_iteration(self, num_iterations=1000, num_eval_iterations=1000):
        for t in range(num_iterations):
            self.policy_evaluation(t, max_num_iterations=num_eval_iterations)
            self.past_Vs.append(self.V.copy())
            self.past_num_queries.append(self.num_generator_queries)
            policy_changed = self.policy_improvement(t)
            if t > 0 and not policy_changed:
                return


if __name__ == '__main__':
    rows = 30
    cols = 30
    spatial = False
    max_steps = 1000
    horz_wind = (0.0, 0.0)
    vert_wind = (0.0, 0.0)
    gamma = 0.98
    learn_num_ep = 1000
    eval_num_ep = 10
    budget = 20000
    planning_horizon = 'Loc:qqq'

    wandb.init(project='ttt', entity='arosenberg', name='FBDPPI%s:%s' % (rows, planning_horizon))

    env = RoomsEnv(rows=rows,
                   cols=cols,
                   spatial=spatial,
                   max_steps=max_steps,
                   horz_wind=horz_wind,
                   vert_wind=vert_wind,
                   agg_num=5)
    agent = PolicyIteration(env,
                            gamma=gamma,
                            planning_horizon=planning_horizon)
    if type(planning_horizon) == str and planning_horizon.startswith('VI'):
        agent.run_value_iteration(update_num_queries=True)
    else:
        agent.run_policy_iteration()

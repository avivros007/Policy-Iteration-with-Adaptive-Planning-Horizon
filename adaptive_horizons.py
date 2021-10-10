import numpy as np


def get_current_planning_horizon(planning_horizon, h_step, state, agent, t):
    if type(planning_horizon) == int:
        return planning_horizon
    elif type(planning_horizon) == str:
        params = planning_horizon.split(":")[1].split(",")
        horizon_func = None
        if planning_horizon.startswith('SmallCloseWall'):
            horizon_func = get_horizon_larger_when_closer_to_walls(int(params[0]), int(params[1]), int(params[2]))
        elif planning_horizon.startswith('Random'):
            horizon_func = get_random_horizon(int(params[0]), int(params[1]))
        elif planning_horizon.startswith('Discover'):
            return 7
        elif planning_horizon.startswith('Decreasing'):
            horizon_func = get_decreasing_horizon(int(params[0]), int(params[1]), int(params[2]))
        elif planning_horizon.startswith('Increasing'):
            horizon_func = get_increasing_horizon(int(params[0]), int(params[1]), int(params[2]))
        elif planning_horizon.startswith('SmallCloseWallDecreasing'):
            horizon_func = get_decreasing_horizon_larger_when_closer_to_walls(int(params[0]),
                                                                              int(params[1]),
                                                                              int(params[2]),
                                                                              int(params[3]))
        elif planning_horizon.startswith('SmallAfter'):
            horizon_func = get_smaller_after_n_episodes(int(params[0]), int(params[1]), int(params[2]))
        elif planning_horizon.startswith('LargeAfter'):
            horizon_func = get_larger_after_n_episodes(int(params[0]), int(params[1]), int(params[2]))
        elif planning_horizon.startswith('CON2') \
                or planning_horizon.startswith('TryStar') \
                or planning_horizon.startswith('AppVstar') \
                or planning_horizon.startswith('Loc'):
            return 1
        if horizon_func is not None:
            return horizon_func(h_step, state, agent, t)
    return planning_horizon(h_step, state, agent, t)


def discover_reachable_states(starting_state, horizon, env):
    S = [set() for _ in range(horizon + 1)]
    S[0].add(starting_state)
    for h in range(1, horizon + 1):
        for prev_state in S[h - 1]:
            for action in range(env.num_actions):
                state_action_id = env.state_action_idx_map[(prev_state, action)]
                for next_s_id, next_state in enumerate(env.idx_state_map):
                    if env.p[state_action_id, next_s_id] > 0:
                        S[h].add(next_state)
    return S


def get_horizon_larger_when_closer_to_walls(smaller_horizon, larger_horizon, obs_radius):
    def horizon_larger_when_closer_to_walls(h, s, ag, t):
        low_row = max(s[0], 1)
        low_col = max(s[1], 1)
        high_row = min(s[0] + obs_radius, ag.env.rows - 1)
        high_col = min(s[1] + obs_radius, ag.env.cols - 1)
        if np.sum(ag.env.map[low_row:high_row, low_col:high_col]) > 1:
            return larger_horizon
        return smaller_horizon
    return horizon_larger_when_closer_to_walls


def get_fixed_horizon(hor):
    def fixed_horizon(h, s, ag, t):
        return hor
    return fixed_horizon


def get_smaller_after_n_episodes(smaller_horizon, larger_horizon, n):
    def smaller_after_n_episodes(h, s, ag, t):
        if t < n:
            return larger_horizon
        return smaller_horizon
    return smaller_after_n_episodes


def get_larger_after_n_episodes(smaller_horizon, larger_horizon, n):
    def larger_after_n_episodes(h, s, ag, t):
        if t < n:
            return smaller_horizon
        return larger_horizon
    return larger_after_n_episodes


def get_decreasing_horizon(smaller_horizon, larger_horizon, n):
    def decreasing_horizon(h, s, ag, t):
        return max(smaller_horizon, larger_horizon - n * t)
    return decreasing_horizon


def get_increasing_horizon(smaller_horizon, larger_horizon, n):
    def increasing_horizon(h, s, ag, t):
        return min(larger_horizon, smaller_horizon + n * t)
    return increasing_horizon


def get_random_horizon(smaller_horizon, larger_horizon):
    def random_horizon(h, s, ag, t):
        return np.random.choice(range(smaller_horizon, larger_horizon))
    return random_horizon


def get_decreasing_horizon_larger_when_closer_to_walls(smaller_horizon, larger_horizon, obs_radius, n):
    def decreasing_horizon_larger_when_closer_to_walls(h, s, ag, t):
        low_row = max(s[0], 1)
        low_col = max(s[1], 1)
        high_row = min(s[0] + obs_radius, ag.env.rows - 1)
        high_col = min(s[1] + obs_radius, ag.env.cols - 1)
        if np.sum(ag.env.map[low_row:high_row, low_col:high_col]) > 1:
            return max(smaller_horizon, larger_horizon - n * t)
        return smaller_horizon
    return decreasing_horizon_larger_when_closer_to_walls


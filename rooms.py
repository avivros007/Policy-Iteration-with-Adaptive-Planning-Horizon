import numpy as np
from gym import core, spaces
import matplotlib.pyplot as plt
import cv2
from typing import Union, Tuple, List, Iterable
import scipy.ndimage
import random


c2rgb = [[0, 0, 255],
         [255, 255, 255],
         [0, 255, 0],
         [255, 0, 0]]

MINE_CELLS = [(4, 6), (2, 12), (8, 10), (4, 1), (14, 6), (4, 8), (4, 7), (10, 12), (5, 13), (5, 8), (7, 10), (9, 9), (2, 7), (11, 14), (4, 13), (5, 11), (13, 3), (8, 1), (4, 12), (10, 2)]
START_CELL = [(1, 1)]
APPLE_CELLS = [(1, 4), (4, 4)]
GOAL_CELLS = [(1, 4), (4, 4)]


class RoomsEnv(core.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, rows=15, cols=15, empty=False, random_walls=True, obstacles: Iterable[Union[Tuple, List]] = None,
                 spatial=True, n_apples=0, n_mines=0, action_repeats=1, max_steps=None, seed=None, fixed_reset=True,
                 px=(0.0, 1.0), wind_in_state=False, random_wind=False, vert_wind=(0.2, 0.2), horz_wind=(0.2, 0.2),
                 is_chain=False, chain_reward_loc=0, agg_num=2):
        '''
        vert_wind = (up, down)
        horz_wind = (right, left)
        '''
        if seed == -1:
            seed = np.random.randint(2 ** 30 - 1)

        self.rows = rows
        self.cols = cols
        if max_steps is None:
            self.max_steps = 1 * (rows + cols)
        else:
            self.max_steps = max_steps
        self.px = px
        self.n_apples = n_apples
        self.n_mines = n_mines
        self.random_wind = random_wind
        self.wind_in_state = wind_in_state
        self.vert_wind = np.array(vert_wind)
        self.horz_wind = np.array(horz_wind)
        self.obstacles = obstacles
        self.action_space = spaces.Discrete(4)
        self.spatial = spatial
        self.scale = np.maximum(rows, cols)
        if spatial:
            # n_channels = 4 + wind_in_state * 4 + 2
            n_channels = 4
            self.observation_space = spaces.Box(low=0, high=1, shape=(n_channels, 80, 80),
                                                dtype=np.float32)
        else:
            n_channels = 6 # 2 + (n_apples + n_mines) * 2 + wind_in_state * 4
            self.observation_space = spaces.Box(low=-1, high=1, shape=(n_channels,), dtype=np.float32)
        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1))] + [np.array((0, 1))]
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
        self.random_walls = random_walls
        self.empty = empty
        self.map, self.cell_seed = self._randomize_walls(random=random_walls, empty=empty)
        self.room_with_mines = None
        self.fixed_reset = fixed_reset
        self.taken_positions = np.copy(self.map)
        self.state_cell, self.state = self._random_from_map(1, START_CELL, force_random=False)
        # self.apples_cells, self.apples_map, self.mines_cells, self.mines_map = None, None, None, None
        # self.apples_cells, self.apples_map = self._random_from_map(n_apples, APPLE_CELLS)
        # self.mines_cells, self.mines_map = self._random_from_map(n_mines, MINE_CELLS, force_random=False)
        self.x = None
        self.is_chain = is_chain
        self.chain_reward_loc = chain_reward_loc
        if self.is_chain:
            self._create_chain()
        else:
            self._random_goal()
        self.wind_state = self._generate_wind_state()
        if fixed_reset:
            self.reset_state_cell, self.reset_state = self.state_cell, self.state.copy()
            self.reset_apples_cells, self.reset_apples_map = self.apples_cells, self.apples_map.copy()
            self.reset_mines_cells, self.reset_mines_map = self.mines_cells, self.mines_map.copy()
        else:
            self.reset_state_cell, self.reset_state = None, None
        self.n_resets = 0
        self.tot_reward = 0
        self.viewer = None
        self.action_repeats = action_repeats

        self.state_idx_map, self.idx_state_map = self.get_state_to_idx_map()
        self.state_action_idx_map, self.idx_state_action_map = self.get_state_action_pair_to_idx_map()
        self.num_states = len(self.idx_state_map)
        self.num_state_action_pairs = len(self.idx_state_action_map)
        self.num_actions = self.action_space.n
        self.r, self.p = self.get_r_and_p(False)

        self.agg_num = agg_num
        self.local_state_to_states_map, self.state_to_local_state_map = self.get_local_state_maps(self.agg_num)
        self.num_local_states = len(self.local_state_to_states_map)
        self.local_state_action_idx_map, self.local_idx_state_action_map = self.get_local_state_action_pair_to_idx_map()
        self.num_local_state_action_pairs = len(self.local_idx_state_action_map)
        self.local_r, self.local_p = self.get_local_r_and_p()

        print(f'Initializing {rows}x{cols} Rooms Environment with {n_apples} apples and {n_mines} mines. (seed = {self.rng.seed})')

    def reset(self):
        if self.fixed_reset:
            self.state_cell, self.state = self.reset_state_cell, self.reset_state.copy()
            self.apples_cells, self.apples_map = self.reset_apples_cells, self.reset_apples_map.copy()
            self.mines_cells, self.mines_map = self.reset_mines_cells, self.reset_mines_map.copy()
        else:
            self.wind_state = self._generate_wind_state()
            self.map, self.cell_seed = self._randomize_walls(random=self.random_walls, empty=self.empty)
            self.taken_positions = np.copy(self.map)
            self.state_cell, self.state = self._random_from_map(1, START_CELL, force_random=False)
            self.apples_cells, self.apples_map = self._random_from_map(self.n_apples, APPLE_CELLS)
            self.mines_cells, self.mines_map = self._random_from_map(self.n_mines, MINE_CELLS, force_random=True, px=self.px)
            self._random_goal()

        self.nsteps = 0
        self.tot_reward = 0
        self.n_resets += 1

        obs = self._obs_from_state(self.spatial)

        return obs

    def extract_state(self, obs):
        state = (obs/2+0.5)*self.scale
        return int(np.round(state[0])), int(np.round(state[1]))

    def get_state_to_idx_map(self):
        idx_map = {}
        state_map = []
        idx_cnt = 0
        for i in range(1, self.rows - 1):
            for j in range(1, self.cols - 1):
                if self.map[i][j] == 0:
                    idx_map[(i, j)] = idx_cnt
                    state_map.append((i, j))
                    idx_cnt += 1
        return idx_map, state_map

    def get_state_action_pair_to_idx_map(self):
        idx_map = {}
        state_action_map = []
        idx_cnt = 0
        for i in range(1, self.rows - 1):
            for j in range(1, self.cols - 1):
                if self.map[i][j] == 0:
                    for a in range(self.action_space.n):
                        idx_map[((i, j), a)] = idx_cnt
                        state_action_map.append(((i, j), a))
                        idx_cnt += 1
        return idx_map, state_action_map

    def get_local_state_maps(self, agg_num):
        local_state_to_states_map = []
        state_to_local_state_map = [0 for _ in range(self.num_states)]
        idx_cnt = 0
        for i in range(1, self.rows - 1, agg_num):
            for j in range(1, self.cols - 1, agg_num):
                local_state_to_states_map.append([])
                for k in range(agg_num):
                    for l in range(agg_num):
                        if i + k < self.rows and j + l < self.cols and self.map[i + k, j + l] == 0:
                            state_id = self.state_idx_map[(i + k, j + l)]
                            state_to_local_state_map[state_id] = idx_cnt
                            local_state_to_states_map[-1].append(state_id)
                if local_state_to_states_map[-1] == []:
                    local_state_to_states_map.pop()
                else:
                    idx_cnt += 1
        return local_state_to_states_map, state_to_local_state_map

    def get_local_state_action_pair_to_idx_map(self):
        state_action_map = []
        idx_map = {}
        idx_cnt = 0
        for idx in range(self.num_local_states):
            for action in range(self.num_actions):
                state_action_map.append((idx, action))
                idx_map[(idx, action)] = idx_cnt
                idx_cnt += 1
        return idx_map, state_action_map


    def get_r_and_p(self, random_rewards):
        r = np.zeros(self.num_state_action_pairs)
        p = np.zeros((self.num_state_action_pairs, self.num_states))
        for idx, (state, action) in enumerate(self.idx_state_action_map):
            apples_mines_reward = 1 * self.apples_map[state] - 1.0 * self.mines_map[state]
            r[idx] = apples_mines_reward
            if random_rewards and r[idx] == 0 and np.random.rand() >= 0.8:
                r[idx] = (np.random.rand() - 0.5) / 5
            if apples_mines_reward == 0:
                next_state = tuple(state + self.directions[action])
                if self.map[next_state[0], next_state[1]] != 0:
                    next_state = state
                next_state_prob = 1.0
                next_state_up = tuple(next_state + self.directions[0])
                if self.map[next_state_up[0], next_state_up[1]] == 0:
                    p[idx, self.state_idx_map[tuple(next_state_up)]] = self.vert_wind[0]
                    next_state_prob -= self.vert_wind[0]
                next_state_down = tuple(next_state + self.directions[1])
                if self.map[next_state_down[0], next_state_down[1]] == 0:
                    p[idx, self.state_idx_map[tuple(next_state_down)]] = self.vert_wind[1]
                    next_state_prob -= self.vert_wind[1]
                next_state_left = tuple(next_state + self.directions[2])
                if self.map[next_state_left[0], next_state_left[1]] == 0:
                    p[idx, self.state_idx_map[tuple(next_state_left)]] = self.horz_wind[1]
                    next_state_prob -= self.horz_wind[1]
                next_state_right = tuple(next_state + self.directions[3])
                if self.map[next_state_right[0], next_state_right[1]] == 0:
                    p[idx, self.state_idx_map[tuple(next_state_right)]] = self.horz_wind[0]
                    next_state_prob -= self.horz_wind[0]
                p[idx, self.state_idx_map[tuple(next_state)]] = next_state_prob
            else:
                p[idx] = 1.0 / self.num_states
        return r, p

    def get_local_r_and_p(self):
        r = np.zeros(self.num_state_action_pairs)
        p = np.zeros((self.num_state_action_pairs, self.num_local_states))
        for idx, (ls_id, action) in enumerate(self.local_idx_state_action_map):
            states_id = self.local_state_to_states_map[ls_id]
            for state_id in states_id:
                state = self.idx_state_map[state_id]
                state_action_idx = self.state_action_idx_map[(state, action)]
                r[idx] += self.r[state_action_idx]
                for next_state_id in np.where(self.p[state_action_idx] > 0)[0]:
                    local_next_state_id = self.state_to_local_state_map[next_state_id]
                    p[idx, local_next_state_id] += self.p[state_action_idx, next_state_id]
            r[idx] /= len(states_id)
            p[idx] /= np.sum(p[idx])
        return r, p

    def sample_from_generator(self, state, action):
        # backup current state
        backup_state_cell = (int(self.state_cell[0]), int(self.state_cell[1]))
        backup_state = self.state.copy()

        # set current state to be state
        self.state_cell = state
        self.state = np.zeros_like(self.map)
        self.state[self.state_cell[0], self.state_cell[1]] = 1

        # sample r(s,a) and P( | s,a)
        self._move(action)
        wind_up = np.random.choice([-1, 0, 1], p=[1 - self.vert_wind.sum(), self.vert_wind[0], self.vert_wind[1]])
        wind_right = np.random.choice([-1, 2, 3], p=[1 - self.horz_wind.sum(), self.horz_wind[1], self.horz_wind[0]])
        if wind_up >= 0:
            self._move(wind_up)
        if wind_right >= 0:
            self._move(wind_right)
        apple_collected_location = self.apples_map * self.state
        mine_collected_location = self.mines_map * self.state
        r = 1.0 * np.sum(apple_collected_location) - 1.0 * np.sum(mine_collected_location)
        next_state = self.extract_state(self._obs_from_state(self.spatial))

        # restore backup
        self.state_cell = backup_state_cell
        self.state = backup_state

        return r, next_state

    def step(self, action: int):
        if action == -1:
            return self._step_rule()
        # actions: 0 = up, 1 = down, 2 = left, 3:end = right

        # n_repeats = np.random.choice(range(1, self.action_repeats+1))
        obs = r = done = None
        for _ in range(self.action_repeats):
            self._move(action)
            wind_up = np.random.choice([-1, 0, 1], p=[1 - self.vert_wind.sum(), self.vert_wind[0], self.vert_wind[1]])
            wind_right = np.random.choice([-1, 2, 3], p=[1 - self.horz_wind.sum(), self.horz_wind[1], self.horz_wind[0]])
            if wind_up >= 0:
                self._move(wind_up)
            if wind_right >= 0:
                self._move(wind_right)

            # TODO: apples and mines currently not supporting non spatial obs
            apple_collected_location = self.apples_map * self.state
            mine_collected_location = self.mines_map * self.state
            # if self.n_resets > 1000:
            r = 1 * np.sum(apple_collected_location) - 1.0 * np.sum(mine_collected_location)
            # r -= 0.1 * np.linalg.norm(np.array(self.state_cell) - np.array(self.apples_cells)) / np.sqrt(self.rows * self.cols)
            # else:
            #     r = np.sum(apple_collected_location)
            # self.apples_map -= apple_collected_location
            # self.mines_map -= mine_collected_location
            # if np.sum(mine_collected_location) > 0:
            #     self.state_cell, self.state = self._random_from_map(1, START_CELL)

            # done = np.sum(self.apples_map) == 0 or np.sum(self.mines_map) == 0 or self.nsteps >= self.max_steps
            done = self.nsteps >= self.max_steps

            obs = self._obs_from_state(self.spatial)

            self.tot_reward += r
            self.nsteps += 1
            info = dict()

            if done:
                break

        info['a'] = action
        if done:
            info['episode'] = {'r': np.copy(self.tot_reward), 'l': self.nsteps}

        return obs, r, done, info

    def _move(self, action: int):
        action = int(action)
        if self.is_chain:
            if self.state_cell[0] == self.apples_cells[0] and self.state_cell[1] == self.apples_cells[1]:
                action = 3
            elif self.state_cell[0] == self.apples_cells[0] and self.state_cell[1] == self.apples_cells[1] + 1 and action == 2:
                action = 0
            elif self.state_cell[0] == 2 and self.state_cell[1] == 1 and action == 0:
                action = 2
        next_cell = self.state_cell + self.directions[action]
        if self.map[next_cell[0], next_cell[1]] == 0:
            self.state_cell = next_cell
            self.state = np.zeros_like(self.map)
            self.state[self.state_cell[0], self.state_cell[1]] = 1

    def _random_goal_one_goal_one_mine(self):
        self.apples_map = np.zeros_like(self.map)
        self.mines_map = np.zeros_like(self.map)
        idx = np.random.choice(2, p=self.px)
        self.x = idx
        GOAL_CELLS_tmp = [(1, self.cols - 2), (self.rows - 2, self.cols - 2)]
        goal_cell = GOAL_CELLS_tmp[idx]
        mine_cell = GOAL_CELLS_tmp[1-idx]
        self.apples_map[goal_cell[0], goal_cell[1]] = 1
        self.mines_map[mine_cell[0], mine_cell[1]] = 1
        self.apples_cells = goal_cell
        self.mines_cells = mine_cell

    def _random_goal(self):
        self.apples_map = np.zeros_like(self.map)
        self.mines_map = np.zeros_like(self.map)
        self.apples_cells = [(self.cols - 2, 1), (self.cols - 2, self.cols - 2)]
        # self.apples_cells = [(self.cols - 2, self.cols - 2)]
        self.mines_cells = [(1, self.cols - 2)]
        self.apples_map[self.cols - 2, self.cols - 2] = 1
        self.apples_map[self.cols - 2, 1] = 1
        self.mines_map[1, self.cols - 2] = 1
        num_extra_apples = 0
        while num_extra_apples < 2:
            apple_loc = tuple(np.random.choice(range(3, self.cols - 4), 2))
            if self.map[apple_loc] == 0:
                self.apples_map[apple_loc] = 1
                self.apples_cells.append(apple_loc)
                num_extra_apples += 1

    def _create_chain(self):
        # self.state_cell = (2, 1)
        # self.state = np.zeros_like(self.map)
        # self.state[self.state_cell[0], self.state_cell[1]] = 1
        self.map[1, 1: -1] = 0
        self.map[2, 2: -1] = 1
        self.map[3, 1: -1] = 0
        self.map[4, 1: -1] = 1
        self.apples_map = np.zeros_like(self.map)
        self.mines_map = np.zeros_like(self.map)
        idx = np.random.choice(2, p=self.px)
        goal_cell = (1, self.chain_reward_loc)
        mine_cell = (self.rows - 2, self.cols - 2)
        self.apples_map[goal_cell[0], goal_cell[1]] = 1
        self.mines_map[mine_cell[0], mine_cell[1]] = 1
        self.apples_cells = goal_cell
        self.mines_cells = mine_cell

    def _random_from_map(self, n=1, array=None, force_random=False, px=None):
        map = np.zeros_like(self.map)
        cells = []
        room_num = 0
        if px is not None:
            room_num = self.rng.choice(4, p=px)
            if room_num == 1:
                self.room_with_mines = [np.zeros_like(self.map), np.ones_like(self.map)]
            elif room_num == 2:
                self.room_with_mines = [np.ones_like(self.map), np.zeros_like(self.map)]
        for i in range(n):
            if self.fixed_reset and not force_random and array is not None:
                cell = random.choice(array)
            else:
                cell = self.rng.choice(self.rows), self.rng.choice(self.cols)
                while self.taken_positions[cell[0], cell[1]] == 1 or \
                        (px is not None and self._which_room(cell) != room_num):
                    cell = (self.rng.choice(self.rows), self.rng.choice(self.cols))

            cells.append(cell)
            map[cell[0], cell[1]] = 1
            self.taken_positions[cell[0], cell[1]] = 1

        if n == 1:
            cells = cells[0]

        return cells, map

    def _generate_wind_state(self):
        if self.random_wind:
            vw1 = max(self.rng.rand() - 0.7, 0)
            vw2 = max(self.rng.rand() - 0.7, 0)
            hw1 = max(self.rng.rand() - 0.7, 0)
            hw2 = max(self.rng.rand() - 0.7, 0)
            self.vert_wind = np.array([vw1, vw2])
            self.horz_wind = np.array([hw1, hw2])
        wind_state = [np.ones((self.rows, self.cols)) for _ in range(4)]
        wind_state[0] *= self.vert_wind[0]
        wind_state[1] *= self.vert_wind[1]
        wind_state[2] *= self.horz_wind[0]
        wind_state[3] *= self.horz_wind[1]
        return wind_state

    def _obs_from_state(self, spatial):
        if spatial:
            # im_list = [self.state, self.map, self.apples_map, self.mines_map]
            # if self.room_with_mines is not None:
            #     im_list.extend(self.room_with_mines)
            # if self.wind_in_state:
            #     im_list.extend(self.wind_state)
            im_list = [self.state, self.map, self.apples_map, self.mines_map] # FOR DEBUG

            obs = np.stack(im_list, axis=0)
            obs = scipy.ndimage.zoom(obs, (1, 80./obs.shape[1], 80./obs.shape[2]), order=0)
            return obs.astype('float32')
        else:
            obs = list(self.state_cell)
            obs = np.concatenate([obs, self.apples_cells, self.mines_cells])
            obs = 2 * (np.array(obs) / self.scale - 0.5)
            if self.wind_in_state:
                obs = np.concatenate([obs, [*self.vert_wind, *self.horz_wind]])
            return obs

    def _color_obs(self, obs):
        rgb_obs = np.zeros((3, *obs.shape[1:]))
        for i in range(4):
            rgb_obs += obs[i:i+1] * np.array(c2rgb[i])[:, np.newaxis, np.newaxis]
        res = dict(img=np.moveaxis(rgb_obs, 0, 2), vert_wind=self.vert_wind, horz_wind=self.horz_wind)
        return res

    def _which_room(self, cell):
        if cell[0] <= self.cell_seed[0] and cell[1] <= self.cell_seed[1]:
            return 0
        elif cell[0] <= self.cell_seed[0] and cell[1] > self.cell_seed[1]:
            return 1
        elif cell[0] > self.cell_seed[0] and cell[1] <= self.cell_seed[1]:
            return 2
        else:
            return 3

    def _step_rule(self):
        if self.state_cell[1] + 1 <= self.apples_cells[1]:
            right_free = 1 - self.mines_map[self.state_cell[0], self.state_cell[1] + 1]
        else:
            right_free = 0
        if self.state_cell[0] + 1 <= self.apples_cells[0]:
            down_free = 1 - self.mines_map[self.state_cell[0] + 1, self.state_cell[1]]
        else:
            down_free = 0
        if right_free + down_free == 0:
            action_probs = np.array([0.5, 0, 0.5, 0])
        else:
            action_weights = np.array([0, down_free, 0, right_free])
            action_probs = action_weights / action_weights.sum()
        action = np.random.choice(range(4), p=action_probs)
        return self.step(action)

    def _randomize_walls(self, random=False, empty=False):
        map = np.zeros((self.rows, self.cols))

        map[0, :] = 1
        map[:, 0] = 1
        map[-1:, :] = 1
        map[:, -1:] = 1

        if self.obstacles:
            for obstacle in self.obstacles:
                map[obstacle[0] - 1:obstacle[0] + 2, obstacle[1] - 1:obstacle[1] + 2] = 1

        if random:
            seed = (self.rng.randint(3, self.rows - 3), self.rng.randint(3, self.cols - 3))
            doors = (self.rng.randint(1, seed[0]),
                     self.rng.randint(seed[0] + 1, self.rows - 1),
                     self.rng.randint(1, seed[1]),
                     self.rng.randint(seed[1] + 1, self.cols - 1))
        else:
            seed = (self.rows // 2, self.cols // 2)
            doors = (self.rows // 4, 3 * self.rows // 4, self.cols // 4, 3 * self.cols // 4)

        if empty:
            return map, seed

        map[seed[0]:seed[0] + 1, :] = 1
        map[:, seed[1]:(seed[1] + 1)] = 1
        map[doors[0]:(doors[0]+1), seed[1]:(seed[1] + 1)] = 0
        map[doors[1]:(doors[1]+1), seed[1]:(seed[1] + 1)] = 0
        map[seed[0]:(seed[0] + 1), doors[2]:(doors[2]+1)] = 0
        map[seed[0]:(seed[0] + 1), doors[3]:(doors[3]+1)] = 0

        # [0 -> 1, 2 -> 3, 0 -> 2, 1 -> 3]
        self.doors = [(doors[0], seed[1]), (doors[1], seed[1]), (seed[0], doors[2]), (seed[0], doors[3])]

        return map, seed

    def render(self, mode='human'):
        im = self._obs_from_state(True)
        res = self._color_obs(im)

        img = cv2.resize(res['img'].astype(np.uint8), dsize=(256, 256), interpolation=cv2.INTER_AREA)
        if mode == 'rgb_array':
            return res['img']
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
        return self.rng.seed

    def disconnect(self):
        pass


if __name__ == '__main__':
    env = RoomsEnv(
        rows=7,
        cols=10,
        spatial=False,
        max_steps=1000,
        horz_wind=(0.0, 0.0),
        vert_wind=(0.0, 0.0),
        is_chain=True,
        chain_reward_loc=5,
    )
    # obs = env.reset()
    # env.step(env.action_space.sample())
    # print(env.state_cell)
    # print(env.apples_cells)
    # print(env.mines_cells)
    img = env.render('rgb_array')
    plt.imshow(img)
    # plt.title(f'Horz wind = {res["horz_wind"]}, Vert wind = {res["vert_wind"]}')
    plt.show()

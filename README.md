# Policy Iteration with Adaptive Planning Horizon

Experiments for picking planning horizons adaptively in grid world environments that include goal states and walls.

The options for planning horizons (through the "planning_horizon" parameter) are:
1. **x (int):** a fixed planning horizon of x.
2. **"SmallCloseWall:x,y,z" (str):** use horizon y when at distance at most z from wall, and otherwise use horizon x.
3. **"Random:x,y" (str):** use a random horizon (in each state) in the range [x,y].
4. **"Decreasing:x,y,z" (str):** start with horizon y, and decrease it by z in each iteration until reaching horizon x.
5. **"Increasing:x,y,z" (str):** start with horizon x, and increase it by z in each iteration until reaching horizon y.
6. **"SmallCloseWallDecreasing:x,y,z,w" (str):** use horizon y when at distance at most z from wall and decrease it by w with every iteration until reaching horizon x. Otherwise use horizon x.
7. **"SmallAfter:x,y,z" (str):** start with horizon y, and after z iterations switch to horizon x.
8. **"LargeAfter:x,y,z" (str):** start with horizon x, and after z iterations switch to horizon y.
9. **"LargeAfter:x,y,z" (str):** start with horizon x, and after z iterations switch to horizon y.
10. **"TryStar:x" (str):** perform 1-step improvement and then x-step improvement in the states with contraction worse than gamma^x.
11. **"CON2:x" (str):** perform 1-step improvement, then 2-step improvement in the x fraction of states with worst distance from V*, and then repeat with 4-step improvement and 8-step improvement.
12. **"CON2q:x,y,z" (str):** perform 1-step improvement, then 2-step improvement in the x fraction of states with worst distance from V*, and then repeat with 4-step improvement with y fraction and 8-step improvement with z fraction.
13. **"AppVstar:x,y,z,w" (str):** same as CON2q but distance is measured against the value from w iterations ago and not V*.
14. **"Loc:x,y,z" (str):** same as CON2q but distance is not measured against V*, but instead against the optimal value in a smaller problem that aggregates every w\*w states together (w is set in the parameter "agg_num").
15. **f(x,y,z,w) (function):** the planning horizon in each step is chosen by the functuion f with the parameters x - the step within the episode (only for finite-horizon and not discounted), y - the current state, z - the PI class instance (for implementation purposes), w - the current iteration.
16. **"VI:x" (str):** run value iteration (and not policy iteration) with x-step lookahead.

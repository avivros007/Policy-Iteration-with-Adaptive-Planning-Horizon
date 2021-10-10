import wandb

from rooms import RoomsEnv
from policy_iteration import PolicyIteration

def init_main():
    hyperparameter_defaults = dict(
        rows=30,
        cols=30,
        horz_wind_right=0.0,
        horz_wind_left=0.0,
        vert_wind_up=0.0,
        vert_wind_down=0.0,
        gamma=0.95,
        planning_horizon=1,
        seed=1,
        agg_num=1,
    )
    wandb.init(config=hyperparameter_defaults)
    return wandb.config

if __name__ == '__main__':
    config = init_main()
    env = RoomsEnv(
        rows=config.rows,
        cols=config.cols,
        spatial=False,
        max_steps=1000,
        horz_wind=(config.horz_wind_right, config.horz_wind_left),
        vert_wind=(config.vert_wind_up, config.vert_wind_down),
        is_chain=False,
        agg_num=config.agg_num,
    )
    agent = PolicyIteration(
        env=env,
        gamma=config.gamma,
        planning_horizon=config.planning_horizon,
    )
    if type(config.planning_horizon) == str and config.planning_horizon.startswith('VI'):
        agent.run_value_iteration(update_num_queries=True)
    else:
        agent.run_policy_iteration()

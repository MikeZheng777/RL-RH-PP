import sys
import os
import numpy as np
from gym import spaces
import gym

sys.path.append(os.path.abspath("build/python"))
import warehouse_sim


class WarehouseEnv(gym.Env):
    def __init__(self, **kwargs):
        super(WarehouseEnv, self).__init__()
        self.env = warehouse_sim.RLEnvironment(**kwargs)
        self.action_space = spaces.Box(low=0, high=50, shape=(50,), dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=3000, shape=(50, 20), dtype=np.uint8)
        seed = kwargs.get("seed", 0)
        self.num_agent = kwargs.get("agent_num", 50)
        self.pad_len_max = kwargs.get("observation_window", 40)
        self.simulation_window = kwargs.get("simulation_window", 5)
        self.kappa = kwargs.get("kappa", 1000)
        np.random.seed(seed)
        self.eval_seed = seed
        self.G_size = kwargs.get("map_rows", 10) * kwargs.get("map_cols", 10)

    def step(self, action):
        try:
            obs, reward, done, info = self.env.step(action)
        except Exception as e:
            import traceback
            print("[C++ Exception in step]", str(e))
            traceback.print_exc()
            raise

        truncated = False
        dis_to_goal_after_excution = reward[-2]
        ratio_wait_agent = reward[-1]

        return_reward = -dis_to_goal_after_excution - self.kappa * ratio_wait_agent

        num_task_acc = reward[0]
        info["num_task_acc"] = num_task_acc
        info["solve_time"] = self.env.get_solve_time()
        info["num_wait_agent_ratio"] = ratio_wait_agent
        info["dis_to_goal_after_excution"] = dis_to_goal_after_excution
        info["num_failed_steps"] = 0

        if "num_failed_calls" in info:
            failed_calls = int(info["num_failed_calls"])
            return_reward -= 100 * failed_calls
            if failed_calls > 0:
                info["num_failed_steps"] += 1

        return_obs = np.zeros((self.num_agent, self.pad_len_max))
        for i in range(self.num_agent):
            if len(obs[0][i][:, 0]) > self.pad_len_max:
                return_obs[i] = obs[0][i][:self.pad_len_max, 0]
            else:
                return_obs[i] = np.pad(obs[0][i][:, 0], (0, self.pad_len_max - len(obs[0][i][:, 0])),
                                       'constant', constant_values=self.G_size)

        excuted_next_direction = info["executed_paths"]
        return_excuted_next_direction = np.zeros((self.num_agent, self.simulation_window))
        for i in range(self.num_agent):
            return_excuted_next_direction[i] = excuted_next_direction[i][:self.simulation_window, 0]

        info["next_direction"] = np.int32(return_obs[:, 1] - return_obs[:, 0])
        info["next_excuted_direction"] = np.int32(return_excuted_next_direction[:, 1] - return_excuted_next_direction[:, 0])

        return return_obs, return_reward, done, truncated, info

    def reset(self, seed=None, options=None):
        if seed is None:
            seed = np.random.randint(1e8)

        if options is not None:
            if options['eval']:
                seed = self.eval_seed

        try:
            self.env.reset(seed)
        except Exception as e:
            import traceback
            print("[C++ Exception in reset]", str(e))
            traceback.print_exc()
            raise

        if options is not None:
            if options.get("num_orders", None) is not None:
                action = np.tile(np.arange(0, self.num_agent, dtype=int), (options["num_orders"], 1))
            else:
                action = np.arange(0, self.num_agent, dtype=int)
        else:
            action = np.arange(0, self.num_agent, dtype=int)

        obs, reward, done, info = self.env.step(action)

        return_obs = np.zeros((self.num_agent, self.pad_len_max))
        for i in range(self.num_agent):
            if len(obs[0][i][:, 0]) > self.pad_len_max:
                return_obs[i] = obs[0][i][:self.pad_len_max, 0]
            else:
                return_obs[i] = np.pad(obs[0][i][:, 0], (0, self.pad_len_max - len(obs[0][i][:, 0])),
                                       'constant', constant_values=self.G_size)

        excuted_next_direction = info["executed_paths"]
        return_excuted_next_direction = np.zeros((self.num_agent, self.simulation_window))
        for i in range(self.num_agent):
            return_excuted_next_direction[i] = excuted_next_direction[i][:self.simulation_window, 0]

        info["next_direction"] = np.int32(return_obs[:, 1] - return_obs[:, 0])
        info["next_excuted_direction"] = np.int32(return_excuted_next_direction[:, 1] - return_excuted_next_direction[:, 0])

        return return_obs, info

    def close(self):
        pass

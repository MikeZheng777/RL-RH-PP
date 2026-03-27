import os
import torch
import numpy as np
import random
import yaml
import copy
import argparse
from RL.utils import Config
from RL.agent import PPO_Agent
from RL.gym_env import WarehouseEnv
from tianshou.env import SubprocVectorEnv
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tianshou.env.venvs")

# Map dimensions lookup
MAP_DIMS = {
    "SYMBOTIC": (33, 39),
    "KIVA": (17, 46),
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="RL training for Rolling-Horizon Priority-Based Planning")

    # Scenario
    parser.add_argument("--scenario", type=str, default="SYMBOTIC", choices=["SYMBOTIC", "KIVA"])
    parser.add_argument("--map", type=str, default=None, help="Path to map file (auto-set from scenario if omitted)")
    parser.add_argument("--agent_num", type=int, default=80)
    parser.add_argument("--kappa", type=float, default=1000)

    # Simulation
    parser.add_argument("--simulation_time", type=int, default=800)
    parser.add_argument("--simulation_window", type=int, default=5)
    parser.add_argument("--planning_window", type=int, default=20)

    # Training
    parser.add_argument("--num_epochs", type=int, default=4000)
    parser.add_argument("--num_cpus", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index (-1 for CPU)")
    parser.add_argument("--no_tb", action="store_true", help="Disable TensorBoard logging")

    # Resume / transfer
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file to resume training")
    parser.add_argument("--transfer_learning", action="store_true")

    # Config file override (optional, CLI args take precedence)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    args = parser.parse_args()

    # Load base config from YAML if provided, otherwise use defaults
    if args.config is not None:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {}

    # Auto-set map from scenario
    if args.map is None:
        args.map = f"maps/{args.scenario.lower()}.map"

    # Auto-set map dimensions
    map_rows, map_cols = MAP_DIMS[args.scenario]

    # Auto-set save_dir
    if args.save_dir is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.save_dir = f"./exp/{ts}_RL_{args.scenario.lower()}_{args.agent_num}_agents"

    # CLI args override config file values
    cli_overrides = {
        "scenario": args.scenario,
        "map": args.map,
        "agent_num": args.agent_num,
        "kappa": args.kappa,
        "simulation_time": args.simulation_time,
        "simulation_window": args.simulation_window,
        "planning_window": args.planning_window,
        "low_level_planning_window": args.planning_window,
        "num_epochs": args.num_epochs,
        "num_cpus": args.num_cpus,
        "num_traj": args.num_cpus,
        "lr_model": args.lr,
        "lr_critic": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "save_dir": args.save_dir,
        "use_tb": not args.no_tb,
        "map_rows": map_rows,
        "map_cols": map_cols,
    }

    # Merge: config file is base, CLI overrides on top
    for k, v in cli_overrides.items():
        config_dict[k] = v

    # Fill in defaults for fields not in config file or CLI
    defaults = {
        "solver": "PPBest",
        "num_orders": 1,
        "screen": 0,
        "output": "cpp_logs",
        "log": False,
        "cutoff_time": 1,
        "rotation": False,
        "robust": 0,
        "hold_endpoints": False,
        "dummy_paths": False,
        "single_agent_solver": "SIPP",
        "starting_epoch": 0,
        "global_step_start": 0,
        "plot_grad_flow": False,
        "prioritize_start": False,
        "CAT": False,
        "lazyP": False,
        "travel_time_window": 0,
        "observation_window": 40,
        "save": True,
        "save_visialization": False,
        "potential_function": "None",
        "potential_threshold": 0,
        "suboptimal_bound": 1,
        "use_cuda": True,
        "lr_decay": 0.999,
        "constant_lr": True,
        "hidden_dims": [32, 32],
        "n_GNN": 2,
        "gamma": 0.99,
        "lam": 0.95,
        "eps_clip": 0.2,
        "eval_epoch": 20,
        "rollout_freq": 1,
        "saving": False,
        "checkpoint_epochs": 10,
        "num_reuse_rollout": 3,
        "distributed": False,
        "eval_only": False,
        "no_progress_bar": False,
        "resume": None,
        "max_grad_norm": 0.5,
        "rollout_segement_length": 10,
        "num_heads": 4,
        "ff_factor": 2,
        "mask_inner": True,
        "mask_logits": True,
        "tanh_clipping": 10,
    }
    for k, v in defaults.items():
        config_dict.setdefault(k, v)

    opts = Config(config_dict)
    opts.continue_training = args.checkpoint is not None
    opts.transfer_learning = args.transfer_learning
    opts.checkpoint_file = args.checkpoint

    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    # Create parallel environments
    num_cpus = opts.num_cpus
    config_list = [copy.copy(opts) for _ in range(num_cpus)]
    for i in range(num_cpus):
        config_list[i].seed = i

    env = SubprocVectorEnv([lambda i=i: WarehouseEnv(**config_list[i].__dict__) for i in range(num_cpus)])

    os.makedirs(opts.save_dir, exist_ok=True)
    if opts.use_cuda:
        if args.gpu >= 0 and torch.cuda.is_available():
            opts.device = torch.device(f'cuda:{args.gpu}')
        else:
            opts.device = torch.device('cpu')
            opts.use_cuda = False
    else:
        opts.device = torch.device('cpu')

    agent = PPO_Agent(opts)

    if opts.continue_training:
        if opts.transfer_learning:
            agent.load(opts.checkpoint_file, transfer_learning=True)
            print("Transfer learning from:", opts.checkpoint_file)
        else:
            agent.load(opts.checkpoint_file)
            print("Resumed training from:", opts.checkpoint_file)

    agent.start_training(env)

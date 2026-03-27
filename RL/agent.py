import os
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings
from torch.utils.data import DataLoader, Dataset
from tensorboard_logger import Logger as TbLogger
import numpy as np
import random
from RL.network import NCODecoder, MAPFEncoder_GNN, MHAEncoderLayer
from RL.utils import *
import traceback


def init_parameters(net: nn.Module):
    for param in net.parameters():
        stdv = 1. / math.sqrt(param.size(-1))
        param.data.uniform_(-stdv, stdv)


class Actor(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.encoder = MAPFEncoder_GNN(opts)
        self.decoder = NCODecoder(opts)

    def forward(self, x_in, map_in, only_critic=False, old_action=None):
        agent_embed = self.encoder(x_in, map_in)
        actions, logps = self.decoder(agent_embed, old_action)
        return actions, logps


class Critic(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.encoder = MAPFEncoder_GNN(opts)
        hidden_size = opts.hidden_dims[-1]
        self.value_embedding = nn.Embedding(1, hidden_size)
        self.value_attention = MHAEncoderLayer(hidden_size, opts.num_heads, opts.ff_factor)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        init_parameters(self.linear)
        init_parameters(self.value_attention)

    def forward(self, x_in, map_in):
        agent_embed = self.encoder(x_in, map_in)
        B, N, dim = agent_embed.size()
        attn_token = self.value_embedding.weight.view(1, 1, dim).expand(B, 1, dim)
        agent_embed_attn = torch.cat([attn_token, agent_embed], dim=1)
        out = self.value_attention(agent_embed_attn)[:, 0]
        value = self.linear(out).squeeze(-1)
        return value


class Memory:
    def __init__(self):
        self.actions = []
        self.obs = []
        self.logprobs = []
        self.rewards = []
        self.obj = []

    def clear_memory(self):
        del self.actions[:]
        del self.obs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.obj[:]


class PPO_Agent:
    def __init__(self, opts):
        self.opts = opts

        # Load the map as fixed input
        self.map = load_map_from_file(opts.map)
        self.map_in_raw = torch.tensor(self.map, dtype=torch.int64).to(opts.device).view(-1)
        self.map_in = torch.where(self.map_in_raw)[0]
        self.free_space = torch.where(self.map_in_raw == 0)[0]

        self.actor = Actor(opts)

        if not opts.eval_only:
            self.critic = Critic(opts)

            actor_params = list(self.actor.parameters())
            critic_params = list(self.critic.parameters())

            self.optimizer = torch.optim.Adam(
                [{'params': actor_params, 'lr': opts.lr_model}] +
                [{'params': critic_params, 'lr': opts.lr_critic}]
            )

            if getattr(opts, 'constant_lr', False):
                self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, 1.0, last_epoch=-1,
                )
            else:
                self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, opts.lr_decay, last_epoch=-1,
                )

        print(f'Distributed: {opts.distributed}')
        if opts.use_cuda and not opts.distributed:
            self.actor.to(opts.device)
            if not opts.eval_only:
                self.critic.to(opts.device)

    def load(self, checkpoint_path, transfer_learning=False):
        print('Loading checkpoint from:', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)

        get_inner_model(self.actor).load_state_dict(checkpoint['actor'])
        get_inner_model(self.critic).load_state_dict(checkpoint['critic'])

        if not transfer_learning:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            last_leraning_rate = self.optimizer.param_groups[0]['lr']

            if self.opts.constant_lr:
                print('Set constant lr')
                self.optimizer = torch.optim.Adam(
                    [{'params': self.actor.parameters(), 'lr': last_leraning_rate}] +
                    [{'params': self.critic.parameters(), 'lr': last_leraning_rate}]
                )
                self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, 1, last_epoch=-1,
                )

            torch.set_rng_state(checkpoint['rng_state'])
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])

    def save(self, epoch, global_step=0):
        print('Saving model and state...')
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'critic': get_inner_model(self.critic).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(self.opts.save_dir, 'epoch-{}-{}.pt'.format(epoch, global_step))
        )

    def eval(self):
        torch.set_grad_enabled(False)
        self.actor.eval()
        if not self.opts.eval_only:
            self.critic.eval()

    def train(self):
        torch.set_grad_enabled(True)
        self.actor.train()
        if not self.opts.eval_only:
            self.critic.train()

    def start_training(self, env):
        train(env, self)


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        self.obss = trajectories['obss']
        self.actions = trajectories['actions']
        self.log_probs = trajectories['log_probs']
        self.advantages = trajectories['advantages']
        self.returns = trajectories['returns']

    def __len__(self):
        return len(self.obss)

    def __getitem__(self, idx):
        return {
            'obs': self.obss[idx],
            'action': self.actions[idx],
            'log_prob': self.log_probs[idx],
            'advantage': self.advantages[idx],
            'return': self.returns[idx]
        }


def evaluate(env, agent, epoch, opts, global_step, tb_logger=None):
    obs, _ = env.reset(options={"eval": True, "num_orders": 1})
    obs = torch.tensor(obs).long().to(opts.device)
    num_envs = obs.shape[0]
    total_reward = np.zeros(num_envs)
    total_num_task_acc = np.zeros(num_envs)
    total_failed_calls = np.zeros(num_envs)
    total_dis_to_goal = np.zeros(num_envs)
    total_wait_ratio = np.zeros(num_envs)
    done = 0
    while not done:
        with torch.no_grad():
            action, logp = agent.actor(obs, agent.map_in)
            value = agent.critic(obs, agent.map_in)

        action = action.unsqueeze(1)
        action_np = action.cpu().numpy()
        try:
            next_obs, reward, done, truncated, info = env.step(action_np)
        except Exception as e:
            print(f"[CPP ERROR][evaluate] {e}")
            traceback.print_exc()
            if tb_logger is not None:
                tb_logger.log_value('Errors/cpp_exception_evaluate', 1, global_step)
            raise

        total_reward += np.array(reward)
        total_num_task_acc += np.array([i['num_task_acc'] for i in info])
        total_failed_calls += np.array([i.get('num_failed_calls', 0) for i in info])
        total_dis_to_goal += np.array([i.get('dis_to_goal_after_excution', 0.0) for i in info])
        total_wait_ratio += np.array([i.get('num_wait_agent_ratio', 0.0) for i in info])

        obs = torch.tensor(next_obs).long().to(opts.device)
        done = done[0]

    total_reward = np.mean(total_reward)
    total_num_task_acc = np.mean(total_num_task_acc)
    total_failed_calls_mean = np.mean(total_failed_calls)
    total_dis_to_goal_mean = np.mean(total_dis_to_goal)
    total_wait_ratio_mean = np.mean(total_wait_ratio)

    if opts.use_tb:
        tb_logger.log_value('Metrics/evaluate_sum_reward', total_reward.item(), global_step)
        tb_logger.log_value('Metrics/evaluate_throughput', total_num_task_acc, global_step)
        tb_logger.log_value('Metrics/evaluate_failed_calls', float(total_failed_calls_mean), global_step)
        tb_logger.log_value('Metrics/evaluate_dis_to_goal', float(total_dis_to_goal_mean), global_step)
        tb_logger.log_value('Metrics/evaluate_wait_ratio', float(total_wait_ratio_mean), global_step)

    print(f"Epoch {epoch + 1}, Evaluate Sum Reward: {total_reward.item()}")
    print(f"Epoch {epoch + 1}, Evaluate Num Task Acc: {total_num_task_acc}")
    print(f"Epoch {epoch + 1}, Evaluate Failed Calls: {total_failed_calls_mean}")
    print(f"Epoch {epoch + 1}, Evaluate Dis to Goal: {total_dis_to_goal_mean}")
    print(f"Epoch {epoch + 1}, Evaluate Wait Ratio: {total_wait_ratio_mean}")


def rollout(env, agent, epoch, opts):
    obs, _ = env.reset(options={"eval": False, "num_orders": 1})
    obs = torch.tensor(obs).long().to(opts.device)

    trajectories = {
        'obss': [],
        'actions': [],
        'log_probs': [],
        'advantages': [],
        'returns': []
    }

    num_envs = obs.shape[0]
    env_trajs = {'obss': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': []}

    num_traj_collected = 0
    num_step_collect = 0

    pbar = tqdm(total=opts.num_traj,
                disable=opts.no_progress_bar,
                desc='rollout data collection',
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    while num_traj_collected < opts.num_traj:
        with torch.no_grad():
            action, logp = agent.actor(obs, agent.map_in)
            value = agent.critic(obs, agent.map_in)

        action = action.unsqueeze(1)
        action_np = action.cpu().numpy()
        try:
            next_obs, reward, done, truncated, info = env.step(action_np)
        except Exception as e:
            print(f"[CPP ERROR][rollout] {e}")
            traceback.print_exc()
            raise

        next_obs = torch.tensor(next_obs).long().to(opts.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(opts.device)
        done = torch.tensor(done).long().to(opts.device)

        env_trajs['obss'].append(obs.cpu())
        env_trajs['actions'].append(action.cpu())
        env_trajs['log_probs'].append(logp.cpu())
        env_trajs['rewards'].append(reward.cpu())
        env_trajs['values'].append(value.cpu())
        env_trajs['dones'].append(done.cpu())

        obs = next_obs
        num_step_collect += 1

        if num_step_collect % opts.rollout_segement_length == 0:
            if not torch.all(done):
                current_value = value.detach().to(opts.device)
            else:
                current_value = torch.zeros(num_envs).to(opts.device)

            advantages, returns = compute_return_and_advantage(env_trajs, current_value, opts.device, gamma=opts.gamma)

            for env_id in range(num_envs):
                trajectories['obss'].append(torch.stack(env_trajs['obss'])[:, env_id])
                trajectories['actions'].append(torch.stack(env_trajs['actions'])[:, env_id])
                trajectories['log_probs'].append(torch.stack(env_trajs['log_probs'])[:, env_id])
                trajectories['advantages'].append(advantages[:, env_id])
                trajectories['returns'].append(returns[:, env_id])

            env_trajs = {'obss': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': []}

        if torch.all(done):
            try:
                obs, _ = env.reset(options={"eval": False, "num_orders": 1})
                obs = torch.tensor(obs).long().to(opts.device)
                num_traj_collected += num_envs
                pbar.update(num_envs)
            except Exception as e:
                print(f"[FATAL ERROR] Reset failed: {e}")
                traceback.print_exc()
                import sys
                sys.exit(1)

    return trajectories


def compute_return_and_advantage(traj, current_value, device, gamma=0.99):
    rewards = torch.stack(traj['rewards']).to(device)  # shape: [T, num_envs]
    values = torch.stack(traj['values']).to(device)

    T = rewards.size(0)

    returns = torch.zeros_like(rewards).to(device)
    returns[-1] = current_value
    for t in reversed(range(T - 1)):
        returns[t] = rewards[t] + gamma * returns[t + 1]

    advantages = returns - values
    return advantages, returns


def train(env, agent, rank=0):
    opts = agent.opts

    log_dir = opts.save_dir + '/logs'

    if opts.use_tb:
        os.makedirs(log_dir, exist_ok=True)
        tb_logger = TbLogger(log_dir, flush_secs=5)
    else:
        tb_logger = None

    warnings.filterwarnings("ignore")
    if opts.resume is None:
        torch.manual_seed(opts.seed)
        random.seed(opts.seed)
        np.random.seed(opts.seed)

    for state in agent.optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(opts.device)

    global_step = opts.global_step_start
    if opts.plot_grad_flow:
        os.makedirs(f"{opts.save_dir}/grad_flow_actor", exist_ok=True)
        os.makedirs(f"{opts.save_dir}/grad_flow_critic", exist_ok=True)

    for epoch in range(opts.starting_epoch, opts.starting_epoch + opts.num_epochs):

        if not getattr(opts, 'constant_lr', False):
            agent.lr_scheduler.step(epoch)

        print('\n\n')
        print("|", format(f" Training epoch {epoch} ", "*^60"), "|")
        print("Training with actor lr={:.3e} critic lr={:.3e} for run {}".format(
            agent.optimizer.param_groups[0]['lr'],
            agent.optimizer.param_groups[1]['lr'], opts.save_dir), flush=True)

        training_dataset = rollout(env, agent, epoch, opts)
        training_dataset = TrajectoryDataset(training_dataset)
        training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, shuffle=True)

        pbar = tqdm(total=(opts.num_reuse_rollout * (len(training_dataset) // opts.batch_size + 1)),
                    disable=opts.no_progress_bar or rank != 0, desc='training',
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

        for i in range(opts.num_reuse_rollout):
            for batch in training_dataloader:
                plot_grad_flow_flag = False
                if opts.plot_grad_flow:
                    if epoch % opts.eval_epoch == 1:
                        plot_grad_flow_flag = True

                train_batch(
                    agent, batch, epoch, opts, pbar,
                    global_step, tb_logger,
                    plot_grad_flow_flag=plot_grad_flow_flag
                )
                global_step += 1

        if epoch % opts.eval_epoch == 1:
            evaluate(env, agent, epoch, opts, global_step, tb_logger)
            agent.save(epoch, global_step)

        if opts.use_tb:
            tb_logger.log_value('LearningRate/actor', agent.optimizer.param_groups[0]['lr'], global_step)
            tb_logger.log_value('LearningRate/critic', agent.optimizer.param_groups[1]['lr'], global_step)

        pbar.close()

    env.close()


def train_batch(agent, batch, epoch, opts, pbar, global_step, tb_logger=None, plot_grad_flow_flag=False):
    eps_clip = opts.eps_clip
    agent.train()

    obss = batch['obs'].to(opts.device)
    old_actions = batch['action'].to(opts.device)
    old_log_probs = batch['log_prob'].to(opts.device)
    advantages = batch['advantage'].to(opts.device)
    returns = batch['return'].to(opts.device)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch_size, traj_len, num_agents, obs_dim = obss.shape

    flattened_obss = obss.view(batch_size * traj_len, num_agents, obs_dim)
    flattened_old_actions = old_actions.view(batch_size * traj_len, num_agents)
    actions_pred, log_probs = agent.actor(flattened_obss, agent.map_in, old_action=flattened_old_actions)
    values = agent.critic(flattened_obss, agent.map_in)

    log_probs = log_probs.view(batch_size, traj_len)
    values = values.view(batch_size, traj_len)

    ratios = torch.exp(log_probs - old_log_probs)

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    entropy_loss = -0.01 * log_probs.mean()
    policy_loss += entropy_loss

    value_loss = nn.MSELoss()(values, returns)

    loss = policy_loss + value_loss
    agent.optimizer.zero_grad()
    loss.backward()

    max_grad_norm = 0.5
    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_grad_norm)
    torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_grad_norm)

    actor_total_norm = 0
    for p in agent.actor.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        actor_total_norm += param_norm.item() ** 2
    actor_total_norm = actor_total_norm ** (1. / 2)

    if plot_grad_flow_flag:
        plot_grad_flow(agent.actor, epoch, dir=f"{opts.save_dir}/grad_flow_actor/")
        plot_grad_flow(agent.critic, epoch, dir=f"{opts.save_dir}/grad_flow_critic/")

    critic_total_norm = 0
    for p in agent.critic.parameters():
        param_norm = p.grad.data.norm(2)
        critic_total_norm += param_norm.item() ** 2
    critic_total_norm = critic_total_norm ** (1. / 2)

    agent.optimizer.step()

    if opts.use_tb:
        tb_logger.log_value('Loss/policy_loss', policy_loss.item(), global_step)
        tb_logger.log_value('Loss/value_loss', value_loss.item(), global_step)
        tb_logger.log_value('Metrics/actor_gradient_norm', actor_total_norm, global_step)
        tb_logger.log_value('Metrics/critic_gradient_norm', critic_total_norm, global_step)
        tb_logger.log_value('Metrics/advantage', advantages.mean().item(), global_step)
        tb_logger.log_value('Metrics/log_prob', log_probs.mean().item(), global_step)
        tb_logger.log_value('Metrics/ratios', ratios.mean().item(), global_step)

    print(f"Epoch {epoch+1}, Policy Loss: {policy_loss.item():.3f}, Value Loss: {value_loss.item():.3f}, "
          f"Actor Gradient Norm: {actor_total_norm:.3f}, Critic Gradient Norm: {critic_total_norm:.3f}, "
          f"Log Prob: {log_probs.mean().item():.4f}, Ratios: {ratios.mean().item():.3f}")

    pbar.update(1)

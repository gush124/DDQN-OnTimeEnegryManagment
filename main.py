import argparse
import copy
import importlib
import json
import os
import pandas as pd
import numpy as np
import torch

import DQN
import DDQN
import utils
import HomeSmartGridEnv5
import drawUtils
import numpy as np
import json
from copy import deepcopy

import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def interact_with_environment(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters):
    # For saving files
    global violations_ac, violations_ac
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    log_dir = f"./logs/way5/{args.env}/seed_{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Initialize and load policy
    # policy = DDQN.DDQN(
    #     num_actions,
    #     state_dim,
    #     device,
    #     parameters["discount"],
    #     parameters["optimizer"],
    #     parameters["optimizer_parameters"],
    #     parameters["polyak_target_update"],
    #     parameters["target_update_freq"],
    #     parameters["tau"],
    #     parameters["initial_eps"],
    #     parameters["end_eps"],
    #     parameters["eps_decay_period"],
    #     parameters["eval_eps"],
    #     log_dir
    # )
    policy = DQN.DQN(
        is_atari,
        num_actions,
        state_dim,
        device,
        parameters["discount"],
        parameters["optimizer"],
        parameters["optimizer_parameters"],
        parameters["polyak_target_update"],
        parameters["target_update_freq"],
        parameters["tau"],
        parameters["initial_eps"],
        parameters["end_eps"],
        parameters["eps_decay_period"],
        parameters["eval_eps"],
        log_dir
    )
    # if args.load :
    #     policy.load(f"./models/behavioral_{setting}")
    if args.generate_buffer:
        # For saving files
        # generate_buffer_setting = f"{args.env}_{args.origin_seed}_0"

        policy.load(f"./models/behavioral_{setting}")

    evaluations = []
    all_episode_rewards = []  # 存储每一轮的总奖励
    state, done = env.reset(seed=args.seed), False

    episode_start = True
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    low_noise_ep = np.random.uniform(0, 1) < args.low_noise_p

    beta_start = 0.4  # 起始值
    beta_end = 1.0  # 最终值

    violations_ac = 0  # 空调越限计数
    violations_water = 0  # 热水器越限计数
    violations_ev = 0  # 电动汽车越限计数

    tmp=[]
    water_tmp=[]
    ev=[]
    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        progress = t / args.max_timesteps  # 当前训练进度
        replay_buffer.beta = beta_start + (beta_end - beta_start) * (progress ** 2)  # 指数增长

        # If generating the buffer, episode is low noise with p=low_noise_p.
        # If policy is low noise, we take random actions with p=eval_eps.
        # If the policy is high noise, we take random actions with p=rand_action_p.
        if args.generate_buffer:
            if not low_noise_ep and np.random.uniform(0, 1) < args.rand_action_p - parameters["eval_eps"]:
                action = env.action_space.sample()
            else:
                action = policy.select_action(np.array(state), eval=True)

        if args.train_behavioral:
            if t < parameters["start_timesteps"]:
                action = env.action_space.sample()
            else:
                action = policy.select_action(np.array(state))

        # Perform action and log results
        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        # 检查是否违反约束
        if env.T_room < env.T_comfort_min :
            violations_ac += 1
        if env.s_ev == 1 or env.s_ev == -1:
            if env.daily_charge_counts >= 5 :  # 假设 SOC_ev 是状态的第4个值
                violations_ev += 1
        if env.T_water < env.desired_temp_min :
            violations_water += 1

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, float(done), done, episode_timesteps==1)
        state = copy.copy(next_state)
        # if args.generate_buffer:
        #     #添加水温数据
        #     policy.writer.add_scalar("cost/water", env.T_water, t)
        #
        #     #添加温度数据
        #     policy.writer.add_scalar("cost/ac", env.cost_ac, t)
        #     policy.writer.add_scalar("cost/ev", env.cost_ev, t)
        #     policy.writer.add_scalar("cost/wash", env.cost_water, t)






        policy.writer.add_scalar("state/water", env.T_water, t)

        #添加温度数据
        policy.writer.add_scalar("state/ac", env.T_room, t)
        policy.writer.add_scalar("state/ev", next_state[3], t)
        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(seed=args.seed), False

            all_episode_rewards.append(episode_reward)
            policy.writer.add_scalar("Reward/episode", episode_reward, episode_num)
            # TensorBoard记录
            policy.writer.add_scalar("Violations/AC", violations_ac, episode_num)
            policy.writer.add_scalar("Violations/Water", violations_water, episode_num)
            policy.writer.add_scalar("Violations/EV", violations_ev, episode_num)
            #policy.writer.add_scalar("Violations/episode", violations, episode_num)
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            violations_water = 0  # 初始化违反约束的次数
            violations_ev = 0
            violations_ac=0

            low_noise_ep = np.random.uniform(0, 1) < args.low_noise_p



        # Train agent after collecting sufficient data
        # 训练频率要调整(
        if args.train_behavioral and t >= parameters["start_timesteps"] and (t + 1) % parameters["train_freq"] == 0:
            loss=policy.train(replay_buffer)
            policy.writer.add_scalar("Loss/train", loss, t)




        #Evaluate episode
        #训练dqn的时候这里是按eval_freq=10运行的，evql_freq=25
        # if t>int(args.max_timesteps/2) and args.train_behavioral and (t + 1) % parameters["eval_freq"] == 0:
        #     evaluations.append(eval_policy(policy, args.env, args.seed))

        np.save(f"./results/behavioral_{setting}", episode_reward)

        #画图
    #plot_rewards(all_episode_rewards)

    policy.writer.add_text("Hyperparameters", json.dumps(parameters, indent=4))
    policy.close_writer()

    data_dict = {'充放电状态': env.dischargeAndChargeCounts}
    df = pd.DataFrame(data_dict)
    # 3) 保存为 Excel 文件
    df.to_excel('my_arrays.xlsx', index=False)

    print("cost_ac:",env.cost_ac)
    print("cost_ev:",env.cost_ev)
    print("cost_water:",env.cost_water)

    # Save final policy
    if args.train_behavioral:
        policy.save(f"./models/behavioral_{setting}")

    # Save final buffer and performance
    else:
        evaluations.append(eval_policy(policy, args.env, args.seed))
        np.save(f"./results/buffer_performance_{setting}", evaluations)
        replay_buffer.save(f"./buffers/{buffer_name}")

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env, seed, eval_episodes=10):
    eval_env, _, _, _ = utils.make_env(env, atari_preprocessing)
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state), eval=True)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


# 绘制奖励曲线
def plot_rewards(rewards, window=50):
    plt.figure(figsize=(10, 6))

    # 原始奖励曲线
    plt.plot(rewards, label="Reward per Episode", alpha=0.6)

    # 平滑曲线：计算移动平均
    if len(rewards) >= window:
        smoothed_rewards = [sum(rewards[i:i + window]) / window for i in range(len(rewards) - window + 1)]
        plt.plot(range(window - 1, len(rewards)), smoothed_rewards, label=f"Smoothed (window={window})", color='orange',
                 linewidth=2)

    plt.title("Reward vs Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid()
    plt.show()


# 调用绘图函数



if __name__ == "__main__":

    # Atari Specific
    atari_preprocessing = {
        "frame_skip": 4,
        "frame_size": 84,
        "state_history": 4,
        "done_on_life_loss": False,
        "reward_clipping": True,
        "max_episode_timesteps": 27e3
    }

    atari_parameters = {
        # Exploration
        "start_timesteps": 2e4,
        "initial_eps": 1,
        "end_eps": 1e-2,
        "eps_decay_period": 25e4,
        # Evaluation
        "eval_freq": 5e4,
        "eval_eps": 1e-3,
        # Learning
        "discount": 0.96,
        "buffer_size": 1e6,
        "batch_size": 32,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 0.0000625,
            "eps": 0.00015
        },
        "train_freq": 4,
        "polyak_target_update": False,
        "target_update_freq": 8e3,
        "tau": 1
    }
    # train_freq设置多少比较好
    regular_parameters = {
        # Exploration
        "start_timesteps": 196,
        "initial_eps": 0.9,
        "end_eps": 0.01,
        "eps_decay_period": 1000,
        "eval_freq": 50,
        "eval_eps": 0,
        "discount": 0.97,
        "buffer_size": 1e3,
        "batch_size": 192,
        "experience_replay_alpha": 0.75,
        "experience_replay_beta": 0.45,
        "optimizer": "Adam",
        "optimizer_parameters": {"lr": 1e-4},
        "polyak_target_update": True,
        "target_update_freq": 1,
        "tau": 0.005,
        "train_freq": 10
    }

    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HomeSmartGridEnv")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--origin_seed", default=0, type=int)  # get poor, middle and best data
    parser.add_argument("--buffer_name", default="Default")  # Prepends name to filename
    parser.add_argument("--max_timesteps", default=2016*100, type=int)  # Max time steps to run environment or train for
    parser.add_argument("--BCQ_threshold", default=0.3, type=float)  # Threshold hyper-parameter for BCQ
    parser.add_argument("--low_noise_p", default=0.2,
                        type=float)  # Probability of a low noise episode when generating buffer
    parser.add_argument("--rand_action_p", default=0.2,
                        type=float)  # Probability of taking a random action when generating buffer, during non-low noise episode
    parser.add_argument("--train_behavioral", action="store_true", default=1)  # If true, train behavioral policy
    parser.add_argument("--generate_buffer", action="store_true", default=0)  # If true, generate buffer
    # parser.add_argument("--eval_behavioral", action="store_true", default=0)
    # parser.add_argument("--eval_buffer", action="store_true", default=0)
    #parser.add_argument("--load", default=True,type=bool)  # If true, generate buffer
    args = parser.parse_args()

    print("---------------------------------------")
    if args.train_behavioral:
        print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
    elif args.generate_buffer:
        print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
    else:
        print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if args.train_behavioral and args.generate_buffer:
        print("Train_behavioral and generate_buffer cannot both be true.")
        exit()

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    if not os.path.exists("./buffers"):
        os.makedirs("./buffers")

    # Make env and determine properties
    env, is_atari, state_dim, num_actions = utils.make_env(args.env, atari_preprocessing)
    parameters = regular_parameters

    # Set seeds


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize buffer
    replay_buffer = utils.ReplayBuffer(state_dim, is_atari, atari_preprocessing, parameters["batch_size"],
        parameters["buffer_size"],parameters["experience_replay_alpha"],parameters["experience_replay_beta"], device)

    if args.train_behavioral or args.generate_buffer:
        interact_with_environment(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters)




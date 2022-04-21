import pickle
import argparse
import gym
import skvideo.io
import numpy as np
from pathlib import Path
from softlearning.environments.adapters.gym_adapter import GymAdapter
import time
import matplotlib.pyplot as plt

# Get inputs from user
def get_args():
    parser = argparse.ArgumentParser(description="Plots rollouts")

    parser.add_argument("-d", "--domain", type=str, help="domain name", default="DClawTurnFreeValve3")
    parser.add_argument("-e", "--task", type=str, help="task name", default="Fixed-v0")
    parser.add_argument("-p", "--policy",
                        type=str,
                        help="path to policy",
                        default="")
    parser.add_argument("-r", "--render",
                        type=str,
                        help="onscreen/offscreen rendering",
                        default="onscreen")
    parser.add_argument('-i', '--include',
                        type=str,
                        default='dsuite',
                        help='task suite to import')
    parser.add_argument('-n', '--num_episodes',
                        type=int,
                        default=1,
                        help='number of episodes')
    parser.add_argument('-t', '--horizon_length',
                        type=int,
                        default=50,
                        help='rollout length')
    parser.add_argument('-f', '--filename',
                        type=str,
                        default='',
                        help='offline rendering video path')
    return parser.parse_args()

def main():
    # get args
    args = get_args()

    # load env
    if args.include is not "":
        exec("import " + args.include)


    env_params = {
    }

    env = GymAdapter(
        domain=args.domain,
        task=args.task,
        **env_params,
    )

    rollout_imgs = []
    count_reward_imgs = []

    phased = hasattr(env, "num_phases")
    if phased:
        num_phases = env.num_phases
        phase_idx = 0
    for ep in range(args.num_episodes):
        env.reset()
        ep_rewards = []
        for _ in range(args.horizon_length):
            obs, reward, done, info = env.step(env.action_space.sample())

            rollout_imgs.append(env.render(width=480, height=480, mode="rgb_array"))
            ep_rewards.append(reward)

        obs_dict = env.get_obs_dict()
        rew_dict = env.get_reward_dict(None, obs_dict)
        # print("\nObservations:")
        # for key in obs_keys_to_log:
        #     if key in obs_dict:
        #         print(f"\t{key} = {obs_dict[key]}")
        # print("\nRewards:")
        # for key in rew_keys_to_log:
        #     if key in rew_dict:
        #         print(f"\t{key} = {rew_dict[key]}")

        ep_rewards = np.array(ep_rewards)
        print(f"\nEPISODE #{ep}")
        if len(ep_rewards) > 0:
            print(f"\tMean reward: {ep_rewards.mean()}")
            print(f"\tMax reward: {np.max(ep_rewards)}")
            print(f"\tMin reward: {np.min(ep_rewards)}")
            print(f"\tLast reward: {ep_rewards[-1]}")

    skvideo.io.vwrite(args.filename, np.asarray(rollout_imgs))
    print(f"Done saving videos to {args.filename}")

if __name__ == "__main__":
    main()

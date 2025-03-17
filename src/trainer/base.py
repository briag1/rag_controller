import numpy as np 
import gymnasium as gym
import time
from tqdm import tqdm
from collections import defaultdict
from typing import Tuple, Dict

from src.agent.base import Agent

class Trainer:
    def __init__(self, render:bool= False) -> None:
        self.render = render
        
    
    def play_episode(
        self,
        env: gym.Env,
        agent: Agent,
        train: bool = True,
        explore=True,
        render=False,
        max_steps=200,
        ) -> Tuple[int, float, Dict]:
        
        ep_data = defaultdict(list)
        obs = env.reset()

        if render:
            env.render()

        done = False
        num_steps = 0
        episode_return = 0

        observations = []
        actions = []
        rewards = []

        while not done and num_steps < max_steps:
            action = agent.act(np.array(obs), explore=explore)
            nobs, rew, done, _ = env.step(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(rew)

            if render:
                env.render()

            num_steps += 1
            episode_return += rew

            obs = nobs

        if train:
            new_data = agent.update(rewards, observations, actions)
            for k, v in new_data.items():
                ep_data[k].append(v)

        return num_steps, episode_return, ep_data
    
    def train(self, env: gym.Env,agent: Agent, config:dict, output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Execute training of REINFORCE on given environment using the provided configuration.

        :param env (gym.Env): environment to train on
        :param config: configuration dictionary mapping configuration keys to values
        :param output (bool): flag whether evaluation results should be printed
        :return (Tuple[np.ndarray, np.ndarray, np.ndarray]): average eval returns during training, evaluation
                timesteps and compute times at evaluation
        """
        timesteps_elapsed = 0

        total_steps = config["max_timesteps"]
        eval_returns_all = []
        eval_timesteps_all = []
        eval_times_all = []
        run_data = defaultdict(list)

        start_time = time.time()
        with tqdm(total=total_steps) as pbar:
            while timesteps_elapsed < total_steps:
                elapsed_seconds = time.time() - start_time
                if elapsed_seconds > config["max_time"]:
                    pbar.write(f"Training ended after {elapsed_seconds}s.")
                    break
                agent.schedule_hyperparameters(timesteps_elapsed, total_steps)
                num_steps, ep_return, ep_data = self.play_episode(
                    env,
                    agent,
                    train=True,
                    explore=True,
                    render=False,
                    max_steps=config["episode_length"],
                )
                timesteps_elapsed += num_steps
                pbar.update(num_steps)
                for k, v in ep_data.items():
                    run_data[k].extend(v)
                run_data["train_ep_returns"].append(ep_return)

                if timesteps_elapsed % config["eval_freq"] < num_steps:
                    eval_return = 0
                    if config["env"] == "CartPole-v1" or config["env"] == "Acrobot-v1":
                        max_steps = config["episode_length"]
                    else:
                        raise ValueError(f"Unknown environment {config['env']}")

                    for _ in range(config["eval_episodes"]):
                        _, total_reward, _ = self.play_episode(
                            env,
                            agent,
                            train=False,
                            explore=False,
                            render=self.render,
                            max_steps=max_steps,
                        )
                        eval_return += total_reward / (config["eval_episodes"])
                    if output:
                        pbar.write(
                            f"Evaluation at timestep {timesteps_elapsed} returned a mean return of {eval_return}"
                        )
                    eval_returns_all.append(eval_return)
                    eval_timesteps_all.append(timesteps_elapsed)
                    eval_times_all.append(time.time() - start_time)

        if config["save_filename"]:
            print("\nSaving to: ", agent.save(config["save_filename"]))

        # you may add logging of additional metrics here
        run_data["train_episodes"] = np.arange(1, len(run_data["train_ep_returns"]) + 1).tolist()

        return np.array(eval_returns_all), np.array(eval_timesteps_all), np.array(eval_times_all), run_data
# -*- coding: utf-8 -*-

# Usage:
#   python single_agent/run.py --agent dqn --mode evaluate --episodes 200 --model single_agent/Models/dqn_lunar_lander.pth
#   python single_agent/run.py --agent dqn --mode demo --model single_agent/Models/dqn_lunar_lander.pth
#   python single_agent/run.py --agent heuristic --mode evaluate --episodes 200
#   python single_agent/run.py --agent heuristic --mode demo

#%% Import packages

import argparse
import sys
from pathlib import Path

# Make project root importable when invoked as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import gymnasium as gym
from tqdm import tqdm
from gymnasium.utils.step_api_compatibility import step_api_compatibility

from shared_utils.paths import get_paths
from shared_utils.networks import DQNAgent
from shared_utils.constants import FPS, VIEWPORT_W, VIEWPORT_H
from shared_utils.logging_utils import generate_run_id, stamped, write_evaluation_log
from single_agent.utils.plotting import plot_evaluation
from single_agent.utils import heuristic as heuristic_mod


#%% Functions

def demo_dqn(env, agent, save_video, playback_speed, videos_dir, run_id):
    '''Demo trained single-agent DQN.'''
    total_reward = 0
    steps = 0
    s, info = env.reset()

    out = None
    if save_video:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_fps = int(FPS * playback_speed)
        video_path = str(stamped(videos_dir, 'demo', '.mp4', run_id))
        out = cv2.VideoWriter(video_path, fourcc, video_fps, (VIEWPORT_W, VIEWPORT_H))

    while True:
        a = agent.select_action(s, training=False)
        s, r, terminated, truncated, info = step_api_compatibility(env.step(a), True)
        total_reward += r

        if save_video:
            frame = env.render()
            if frame is not None:
                import cv2
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

        if steps % 20 == 0 or terminated or truncated:
            print(f'step {steps} total_reward {total_reward:+0.2f}')
        steps += 1

        if terminated or truncated:
            break

    if save_video and out is not None:
        hold_frames = int(FPS * playback_speed * 2)
        last_frame = env.render()
        if last_frame is not None:
            import cv2
            last_bgr = cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR)
            for _ in range(hold_frames):
                out.write(last_bgr)
        out.release()

    return total_reward


def evaluate_dqn(env, agent, num_episodes, stats_path):
    '''Evaluate single DQN agent over multiple episodes.'''
    rewards = []
    episode_data = []

    ep_iter = tqdm(range(num_episodes), desc='Evaluating', unit='ep', ncols=60)

    for ep in ep_iter:
        s, info = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            a = agent.select_action(s, training=False)
            s, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            total_reward += r
            steps += 1

        rewards.append(total_reward)
        episode_data.append({
            'episode': ep + 1,
            'reward': total_reward,
            'steps': steps,
            'terminated': terminated,
            'truncated': truncated,
        })

    if stats_path:
        with open(stats_path, 'w') as f:
            f.write('episode,reward,steps,terminated,truncated\n')
            for d in episode_data:
                f.write(f"{d['episode']},{d['reward']:.2f},{d['steps']},"
                        f"{d['terminated']},{d['truncated']}\n")

    return rewards


def demo_heuristic(save_video, playback_speed, videos_dir, run_id):
    '''Demo the heuristic policy. Reuses single_agent.utils.heuristic helpers.'''
    render_mode = 'rgb_array' if save_video else 'human'
    env = gym.make('LunarLander-v3', render_mode=render_mode)

    total_reward = 0
    steps = 0
    s, info = env.reset()

    out = None
    if save_video:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_fps = int(FPS * playback_speed)
        video_path = str(stamped(videos_dir, 'demo_heuristic', '.mp4', run_id))
        out = cv2.VideoWriter(video_path, fourcc, video_fps, (VIEWPORT_W, VIEWPORT_H))

    while True:
        a = heuristic_mod.heuristic(s)
        s, r, terminated, truncated, info = step_api_compatibility(env.step(a), True)
        total_reward += r

        if save_video:
            frame = env.render()
            if frame is not None:
                import cv2
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

        if steps % 20 == 0 or terminated or truncated:
            print(f'step {steps} total_reward {total_reward:+0.2f}')
        steps += 1

        if terminated or truncated:
            break

    if save_video and out is not None:
        hold_frames = int(FPS * playback_speed * 2)
        last_frame = env.render()
        if last_frame is not None:
            import cv2
            last_bgr = cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR)
            for _ in range(hold_frames):
                out.write(last_bgr)
        out.release()

    env.close()
    return total_reward


def evaluate_heuristic(env, num_episodes, stats_path):
    '''Evaluate the heuristic policy. Mirrors evaluate_dqn structure.'''
    rewards = []
    episode_data = []

    ep_iter = tqdm(range(num_episodes), desc='Evaluating', unit='ep', ncols=60)

    for ep in ep_iter:
        s, info = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            a = heuristic_mod.heuristic(s)
            s, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            total_reward += r
            steps += 1

        rewards.append(total_reward)
        episode_data.append({
            'episode': ep + 1,
            'reward': total_reward,
            'steps': steps,
            'terminated': terminated,
            'truncated': truncated,
        })

    if stats_path:
        with open(stats_path, 'w') as f:
            f.write('episode,reward,steps,terminated,truncated\n')
            for d in episode_data:
                f.write(f"{d['episode']},{d['reward']:.2f},{d['steps']},"
                        f"{d['terminated']},{d['truncated']}\n")

    return rewards


def run_single(agent_type, mode, episodes=200, model=None,
               save_video=True, playback_speed=0.5, run_id=None):
    '''Run the single-agent setup in the given mode.'''
    paths = get_paths('single')
    if run_id is None:
        run_id = generate_run_id()

    print(f'Run ID: {run_id}')
    print(f'Agent: {agent_type} | Mode: {mode}')

    label = 'DQN' if agent_type == 'dqn' else 'Heuristic'

    if agent_type == 'dqn':
        agent = DQNAgent()
        model_path = model or (paths['models_dir'] / 'dqn_lunar_lander.pth')
        agent.load(model_path)
        agent.epsilon = 0

        if mode == 'demo':
            render_mode = 'rgb_array' if save_video else 'human'
            env = gym.make('LunarLander-v3', render_mode=render_mode)
            demo_reward = demo_dqn(env, agent, save_video, playback_speed,
                                   paths['videos_dir'], run_id)
            env.close()

            write_evaluation_log(
                stamped(paths['logs_dir'], 'demo', '.log', run_id),
                run_id, 'demo', agent, None, None,
                rewards1=[demo_reward],
                model_path_1=str(model_path),
                variant='single',
            )

        elif mode == 'evaluate':
            env = gym.make('LunarLander-v3')
            print('Evaluating single DQN agent...')
            stats_path = stamped(paths['plots_dir'], 'episode_stats', '.csv', run_id)
            eval_rewards = evaluate_dqn(env, agent, episodes, stats_path)
            env.close()

            plot_evaluation(
                eval_rewards,
                save_path=stamped(paths['plots_dir'], 'evaluation', '.png', run_id),
                stats_path=stamped(paths['plots_dir'], 'stats', '.csv', run_id),
                agent_label=label,
            )

            write_evaluation_log(
                stamped(paths['logs_dir'], 'evaluate', '.log', run_id),
                run_id, 'evaluate', agent, None, None,
                rewards1=eval_rewards,
                model_path_1=str(model_path),
                num_episodes=episodes,
                variant='single',
            )
        else:
            raise ValueError(f'Unknown mode {mode!r} for single-agent run.')
        return

    if agent_type == 'heuristic':
        if mode == 'demo':
            demo_reward = demo_heuristic(save_video, playback_speed,
                                         paths['videos_dir'], run_id)

            write_evaluation_log(
                stamped(paths['logs_dir'], 'demo', '.log', run_id),
                run_id, 'demo', None, None, None,
                rewards1=[demo_reward],
                variant='single-heuristic',
            )

        elif mode == 'evaluate':
            env = gym.make('LunarLander-v3')
            print('Evaluating heuristic agent...')
            stats_path = stamped(paths['plots_dir'], 'episode_stats', '.csv', run_id)
            eval_rewards = evaluate_heuristic(env, episodes, stats_path)
            env.close()

            plot_evaluation(
                eval_rewards,
                save_path=stamped(paths['plots_dir'], 'evaluation', '.png', run_id),
                stats_path=stamped(paths['plots_dir'], 'stats', '.csv', run_id),
                agent_label=label,
            )

            write_evaluation_log(
                stamped(paths['logs_dir'], 'evaluate', '.log', run_id),
                run_id, 'evaluate', None, None, None,
                rewards1=eval_rewards,
                num_episodes=episodes,
                variant='single-heuristic',
            )
        else:
            raise ValueError(f'Unknown mode {mode!r} for single-agent run.')
        return

    raise ValueError(f'Unknown agent type {agent_type!r}. Choose dqn or heuristic.')


def main():
    parser = argparse.ArgumentParser(description='Run / evaluate / demo a single-agent policy')
    parser.add_argument('--agent', required=True, choices=['dqn', 'heuristic'])
    parser.add_argument('--mode', required=True, choices=['demo', 'evaluate'])
    parser.add_argument('--episodes', type=int, default=200,
                        help='Evaluation episodes (default: 200)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to agent weights (DQN only)')
    parser.add_argument('--save-video', action='store_true', default=True,
                        help='Save demo video (default: True)')
    parser.add_argument('--no-video', action='store_true',
                        help='Disable video saving')
    parser.add_argument('--playback-speed', type=float, default=0.5,
                        help='Video playback speed (default: 0.5)')

    args = parser.parse_args()
    run_single(
        agent_type=args.agent,
        mode=args.mode,
        episodes=args.episodes,
        model=args.model,
        save_video=args.save_video and not args.no_video,
        playback_speed=args.playback_speed,
    )


if __name__ == '__main__':

    if len(sys.argv) > 1:
        main()
    else:
        #%% IDE CONFIGURATION
        AGENT          = 'dqn'         # 'dqn' or 'heuristic'
        MODE           = 'evaluate'    # 'demo' or 'evaluate'
        EVAL_EPISODES  = 200
        MODEL          = None          
        SAVE_VIDEO     = True
        PLAYBACK_SPEED = 0.5

        run_single(
            agent_type=AGENT,
            mode=MODE,
            episodes=EVAL_EPISODES,
            model=MODEL,
            save_video=SAVE_VIDEO,
            playback_speed=PLAYBACK_SPEED,
        )

#%% End of Script
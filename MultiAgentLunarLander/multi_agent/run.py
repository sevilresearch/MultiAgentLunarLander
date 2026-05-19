# -*- coding: utf-8 -*-

# Usage:
#   python multi_agent/run.py --config partial --mode baseline --warm-start single_agent/Models/dqn_lunar_lander.pth
#   python multi_agent/run.py --config full    --mode evaluate --episodes 200 --model-1 multi_agent/Models/coop_full/agent1.pth --model-2 multi_agent/Models/coop_full/agent2.pth
#   python multi_agent/run.py --config mixed   --mode demo --model-1 multi_agent/Models/coop_mixed/agent1.pth --model-2 multi_agent/Models/coop_mixed/agent2.pth


#%% Import packages

import argparse
import sys
from pathlib import Path

# Make project root importable when invoked as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm

from shared_utils.paths import get_paths
from shared_utils.constants import (
    COOP_SUCCESS_THRESHOLD, FPS, VIEWPORT_W, VIEWPORT_H,
)
from shared_utils.logging_utils import (
    generate_run_id, stamped, write_evaluation_log,
)
from multi_agent.utils.agents import PartialObsAgent, FullObsAgent
from multi_agent.utils.coop_env import CooperativeLunarLander
from multi_agent.utils.plotting import plot_coop_evaluation


#%% Variant configuration table 

#!!! MUST mirror multi_agent/train.py
CONFIG_TABLE = {
    'partial': {
        'obs_mode': 'partial',
        'agent1_cls': PartialObsAgent,
        'agent2_cls': PartialObsAgent,
    },
    'full': {
        'obs_mode': 'full',
        'agent1_cls': FullObsAgent,
        'agent2_cls': FullObsAgent,
    },
    'mixed': {
        'obs_mode': 'mixed',
        'agent1_cls': PartialObsAgent,
        'agent2_cls': FullObsAgent,
    },
}


#%% Functions

def evaluate_coop(coop_env, agent1, agent2, num_episodes, stats_path):
    '''Evaluate cooperative agents over multiple episodes.'''
    rewards1 = []
    rewards2 = []
    joint_successes = []
    episode_data = []

    ep_iter = tqdm(range(num_episodes), desc='Evaluating', unit='ep', ncols=60)

    for ep in ep_iter:
        s1, s2 = coop_env.reset()
        total_r1, total_r2 = 0.0, 0.0
        steps = 0

        while True:
            a1 = agent1.select_action(s1, training=False)
            a2 = agent2.select_action(s2, training=False)

            (ns1, r1, done1), (ns2, r2, done2), both_done = coop_env.step(a1, a2)

            total_r1 += r1
            total_r2 += r2
            s1, s2 = ns1, ns2
            steps += 1

            if both_done or steps >= 1000:
                break

        rewards1.append(total_r1)
        rewards2.append(total_r2)
        joint_success = (total_r1 > COOP_SUCCESS_THRESHOLD) and (total_r2 > COOP_SUCCESS_THRESHOLD)
        joint_successes.append(joint_success)

        episode_data.append({
            'episode': ep + 1,
            'reward1': total_r1,
            'reward2': total_r2,
            'steps': steps,
            'joint_success': joint_success,
            'contacts': coop_env.contact_count,
        })

    if stats_path:
        with open(stats_path, 'w') as f:
            f.write('episode,reward1,reward2,steps,joint_success,contacts\n')
            for d in episode_data:
                f.write(f"{d['episode']},{d['reward1']:.2f},{d['reward2']:.2f},"
                        f"{d['steps']},{d['joint_success']},{d['contacts']}\n")

    return rewards1, rewards2, joint_successes


def demo_coop(coop_env, agent1, agent2, save_video, playback_speed,
              videos_dir, run_id, model_name_1='Agent 1', model_name_2='Agent 2'):
    '''Demo both cooperative agents in the same frame.'''
    s1, s2 = coop_env.reset()
    total_r1, total_r2 = 0.0, 0.0
    steps = 0

    out = None
    if save_video:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_fps = int(FPS * playback_speed)
        video_path = str(stamped(videos_dir, 'demo', '.mp4', run_id))
        out = cv2.VideoWriter(video_path, fourcc, video_fps, (VIEWPORT_W, VIEWPORT_H))

    while True:
        a1 = agent1.select_action(s1, training=False)
        a2 = agent2.select_action(s2, training=False)

        (ns1, r1, done1), (ns2, r2, done2), both_done = coop_env.step(a1, a2)

        total_r1 += r1
        total_r2 += r2

        if save_video:
            frame = coop_env.render()
            if frame is not None:
                import cv2
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.35
                thickness = 1
                cv2.putText(frame_bgr, f'A1: {model_name_1}',
                            (8, 15), font, font_scale,
                            (230, 140, 100), thickness, cv2.LINE_AA)
                text_size = cv2.getTextSize(f'A2: {model_name_2}',
                                            font, font_scale, thickness)[0]
                cv2.putText(frame_bgr, f'A2: {model_name_2}',
                            (VIEWPORT_W - text_size[0] - 8, 15),
                            font, font_scale,
                            (60, 150, 230), thickness, cv2.LINE_AA)
                cv2.putText(frame_bgr, run_id,
                            (8, VIEWPORT_H - 8), font, font_scale,
                            (180, 180, 180), thickness, cv2.LINE_AA)
                out.write(frame_bgr)

        if steps % 20 == 0 or both_done:
            print(f'step {steps} | R1: {total_r1:+.2f} | R2: {total_r2:+.2f}')

        s1, s2 = ns1, ns2
        steps += 1

        if both_done or steps >= 1000:
            break

    if save_video and out is not None:
        hold_frames = int(FPS * playback_speed * 2)
        last_frame = coop_env.render()
        if last_frame is not None:
            import cv2
            last_bgr = cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            thickness = 1
            cv2.putText(last_bgr, f'A1: {model_name_1}',
                        (8, 15), font, font_scale,
                        (230, 140, 100), thickness, cv2.LINE_AA)
            text_size = cv2.getTextSize(f'A2: {model_name_2}',
                                        font, font_scale, thickness)[0]
            cv2.putText(last_bgr, f'A2: {model_name_2}',
                        (VIEWPORT_W - text_size[0] - 8, 15),
                        font, font_scale,
                        (60, 150, 230), thickness, cv2.LINE_AA)
            cv2.putText(last_bgr, run_id,
                        (8, VIEWPORT_H - 8), font, font_scale,
                        (180, 180, 180), thickness, cv2.LINE_AA)
            for _ in range(hold_frames):
                out.write(last_bgr)
        out.release()

    if coop_env.render_mode == 'human':
        coop_env.close()

    return total_r1, total_r2


def run_baseline(config, cfg, paths, run_id, warm_start_path):
    '''Warm-start from single-agent weights and run smoke-test episodes.'''
    agent1 = cfg['agent1_cls']()
    agent2 = cfg['agent2_cls']()

    if warm_start_path:
        print(f'Warm-starting both agents from: {warm_start_path}')
        agent1.warm_start_from_single(warm_start_path)
        agent2.warm_start_from_single(warm_start_path)
        agent1.epsilon = 0
        agent2.epsilon = 0
    else:
        print('Creating baseline with random (untrained) weights...')

    agent1.save(paths['models_dir'] / f'agent1_{agent1.state_size}d_baseline.pth')
    agent2.save(paths['models_dir'] / f'agent2_{agent2.state_size}d_baseline.pth')

    print('\nRunning 5 smoke-test episodes...')
    coop_env = CooperativeLunarLander(obs_mode=cfg['obs_mode'])
    smoke_r1, smoke_r2 = [], []
    for ep in range(5):
        s1, s2 = coop_env.reset()
        total_r1, total_r2 = 0.0, 0.0
        steps = 0
        while steps < 1000:
            a1 = agent1.select_action(s1, training=False)
            a2 = agent2.select_action(s2, training=False)
            (s1, r1, d1), (s2, r2, d2), both_done = coop_env.step(a1, a2)
            total_r1 += r1
            total_r2 += r2
            steps += 1
            if both_done:
                break
        smoke_r1.append(total_r1)
        smoke_r2.append(total_r2)
        print(f'  Episode {ep+1}: R1={total_r1:+.1f}, R2={total_r2:+.1f}, '
              f'steps={steps}, contacts={coop_env.contact_count}')
    coop_env.close()

    write_evaluation_log(
        stamped(paths['logs_dir'], 'baseline', '.log', run_id),
        run_id, 'baseline', agent1, agent2, coop_env,
        rewards1=smoke_r1, rewards2=smoke_r2,
        warm_start_path=warm_start_path,
        variant=f'coop_{config}',
    )

    print('\nBaseline summary:')
    print(f'  Agent 1 avg: {np.mean(smoke_r1):+.1f}  (min {np.min(smoke_r1):+.1f}, max {np.max(smoke_r1):+.1f})')
    print(f'  Agent 2 avg: {np.mean(smoke_r2):+.1f}  (min {np.min(smoke_r2):+.1f}, max {np.max(smoke_r2):+.1f})')
    print(f'Baseline weights saved to {paths["models_dir"]}/. Run demo or evaluate next.')


def run_multi(config, mode, episodes=200, model_1=None, model_2=None,
              warm_start=None, save_video=True, playback_speed=0.5,
              run_id=None):
    '''Run a cooperative two-agent setup in the given mode.'''
    if config not in CONFIG_TABLE:
        raise ValueError(f'Unknown config {config!r}. Choose from {list(CONFIG_TABLE.keys())}.')

    cfg = CONFIG_TABLE[config]
    variant_key = f'coop_{config}'
    paths = get_paths(variant_key)
    if run_id is None:
        run_id = generate_run_id()

    print(f'Run ID: {run_id}')
    print(f'Config: {config} | Mode: {mode}')

    if mode == 'baseline':
        run_baseline(config, cfg, paths, run_id, warm_start)
        return

    obs_mode = cfg['obs_mode']
    agent1 = cfg['agent1_cls']()
    agent2 = cfg['agent2_cls']()

    m1 = model_1 or (paths['models_dir'] / 'agent1.pth')
    m2 = model_2 or (paths['models_dir'] / 'agent2.pth')
    agent1.load(m1)
    agent2.load(m2)
    agent1.epsilon = 0
    agent2.epsilon = 0

    label1 = f'Agent 1 ({agent1.state_size}d)'
    label2 = f'Agent 2 ({agent2.state_size}d)'

    if mode == 'demo':
        render_mode = 'rgb_array' if save_video else 'human'
        coop_env = CooperativeLunarLander(obs_mode=obs_mode, render_mode=render_mode)

        demo_r1, demo_r2 = demo_coop(
            coop_env, agent1, agent2, save_video, playback_speed,
            paths['videos_dir'], run_id,
            model_name_1=Path(m1).stem,
            model_name_2=Path(m2).stem,
        )
        coop_env.close()

        write_evaluation_log(
            stamped(paths['logs_dir'], 'demo', '.log', run_id),
            run_id, 'demo', agent1, agent2, coop_env,
            rewards1=[demo_r1], rewards2=[demo_r2],
            model_path_1=m1, model_path_2=m2,
            variant=variant_key,
        )

    elif mode == 'evaluate':
        coop_env = CooperativeLunarLander(obs_mode=obs_mode)
        print(f'Evaluating cooperative agents ({obs_mode} obs)...')

        stats_path = stamped(paths['plots_dir'], 'episode_stats', '.csv', run_id)
        r1, r2, js = evaluate_coop(coop_env, agent1, agent2, episodes, stats_path)
        coop_env.close()

        plot_coop_evaluation(
            r1, r2,
            save_path=stamped(paths['plots_dir'], 'evaluation', '.png', run_id),
            stats_path=stamped(paths['plots_dir'], 'stats', '.csv', run_id),
            agent1_label=label1, agent2_label=label2,
        )

        write_evaluation_log(
            stamped(paths['logs_dir'], 'evaluate', '.log', run_id),
            run_id, 'evaluate', agent1, agent2, coop_env,
            rewards1=r1, rewards2=r2, joint_successes=js,
            model_path_1=m1, model_path_2=m2,
            num_episodes=episodes, variant=variant_key,
        )
    else:
        raise ValueError(f'Unknown mode {mode!r}. Choose demo, evaluate, or baseline.')


def main():
    parser = argparse.ArgumentParser(description='Run / evaluate / demo two cooperative DQN agents')
    parser.add_argument('--config', required=True,
                        choices=list(CONFIG_TABLE.keys()),
                        help='Knowledge configuration: partial, full, or mixed')
    parser.add_argument('--mode', required=True,
                        choices=['demo', 'evaluate', 'baseline'],
                        help='Run mode')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Evaluation episodes (default: 200)')
    parser.add_argument('--model-1', type=str, default=None,
                        help='Path to agent 1 weights')
    parser.add_argument('--model-2', type=str, default=None,
                        help='Path to agent 2 weights')
    parser.add_argument('--warm-start', type=str, default=None,
                        help='Single-agent weights for baseline warm start')
    parser.add_argument('--save-video', action='store_true', default=True,
                        help='Save demo video (default: True)')
    parser.add_argument('--no-video', action='store_true',
                        help='Disable video saving')
    parser.add_argument('--playback-speed', type=float, default=0.5,
                        help='Video playback speed (default: 0.5)')

    args = parser.parse_args()
    run_multi(
        config=args.config,
        mode=args.mode,
        episodes=args.episodes,
        model_1=args.model_1,
        model_2=args.model_2,
        warm_start=args.warm_start,
        save_video=args.save_video and not args.no_video,
        playback_speed=args.playback_speed,
    )


if __name__ == '__main__':

    if len(sys.argv) > 1:
        main()
    else:
        #%% IDE CONFIGURATION
        CONFIG         = 'full'        # 'partial', 'full', or 'mixed'
        MODE           = 'evaluate'    # 'demo', 'evaluate', or 'baseline'
        EVAL_EPISODES  = 200
        MODEL_1        = None          
        MODEL_2        = None          
        WARM_START     = None          
        SAVE_VIDEO     = True
        PLAYBACK_SPEED = 0.5

        run_multi(
            config=CONFIG,
            mode=MODE,
            episodes=EVAL_EPISODES,
            model_1=MODEL_1,
            model_2=MODEL_2,
            warm_start=WARM_START,
            save_video=SAVE_VIDEO,
            playback_speed=PLAYBACK_SPEED,
        )

#%% End of Script
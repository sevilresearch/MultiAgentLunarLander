# -*- coding: utf-8 -*-
# multi_agent/utils/agents.py
# Full and Partial DQNAgent Classes

#%% Import packages

import torch
import torch.nn as nn

from shared_utils.networks import DQNAgent


#%% Functions

def _warm_start(agent, single_agent_path, target_state_size):
    '''
    Shared warm-start logic: expand fc1 from 8-dim to target_state_size.
    '''
    checkpoint = torch.load(single_agent_path, map_location=agent.device, weights_only=True)
    single_sd = checkpoint['policy_net']

    new_sd = {}
    for key, param in single_sd.items():
        if key == 'fc1.weight':
            expanded = torch.zeros(param.shape[0], target_state_size, device=agent.device)
            expanded[:, :8] = param
            nn.init.xavier_uniform_(expanded[:, 8:])
            expanded[:, 8:] *= 0.1
            new_sd[key] = expanded
        elif key == 'fc1.bias':
            new_sd[key] = param
        else:
            new_sd[key] = param

    agent.policy_net.load_state_dict(new_sd)
    agent.target_net.load_state_dict(new_sd)

    agent.epsilon = checkpoint.get('epsilon', 1.0)
    agent.episodes_done = checkpoint.get('episodes_done', 0)

    print(f'Warm-started agent from {single_agent_path} '
          f'(fc1 expanded 8->{target_state_size}, eps={agent.epsilon:.4f})')


#%% Class definitions

class CoopDQNAgent(DQNAgent):
    '''DQN Agent with 12-dim observation (own 8 + partner x, y, vx, vy).'''

    def __init__(self, **kwargs):
        kwargs.setdefault('state_size', 12)
        kwargs.setdefault('action_size', 4)
        super().__init__(**kwargs)

    def warm_start_from_single(self, single_agent_path):
        _warm_start(self, single_agent_path, self.state_size)


# Aliases for the mixed-observation variant
PartialObsAgent = CoopDQNAgent  # 12-dim default


class FullObsAgent(DQNAgent):
    '''DQN Agent with 16-dim observation (own 8 + partner's full 8).'''

    def __init__(self, **kwargs):
        kwargs.setdefault('state_size', 16)
        kwargs.setdefault('action_size', 4)
        super().__init__(**kwargs)

    def warm_start_from_single(self, single_agent_path):
        _warm_start(self, single_agent_path, self.state_size)

#%% End of Script
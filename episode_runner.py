import numpy as np
import torch
import parameters as params


def run_episode(model, env):
    episode_reward = 0
    episode_events = 0
    state = env.reset()
    episode_experience = []

    for step in range(params.episode_steps):
        state = np.array(state).flatten()
        old_state = np.copy(state)
        action = model.forward(torch.FloatTensor(state))
        action = action.data.numpy()
        
        # TODO: detect discrete or continuous
        if env.action_dim > 2:
            action = np.argmax(action)

        state, reward, done, event = env.step(action)
        episode_experience.append({'state': old_state, 'step': step, 'action': action,
                                   'reward': reward, 'next_state': np.copy(state), 'event': event})
        episode_reward += reward
        episode_events += event

        if done:
            break

    return episode_reward, episode_events, episode_experience

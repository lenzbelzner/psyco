import numpy as np
from scipy.stats import beta
from episode_runner import run_episode
import parameters as params


def verify(model, env, early_stopping=True):
    # init
    successes = 1
    failures = 1

    rewards = []
    events = []

    for _ in range(params.max_verification_episodes):
        # run episode
        episode_reward, episode_events, episode_experience = run_episode(
            model, env)
        rewards.append(episode_reward)
        events.append(episode_events)

        # determine success or failure
        if episode_events <= params.constraint:
            successes += 1
        else:
            failures += 1

        # compute c_sat
        c_sat = 1. - beta(successes, failures).cdf(params.p_req -
                                                   params.verification_margin_p_req)
        print('c_sat', c_sat, end='\r')

        # sat or not sat?
        if early_stopping and (c_sat >= params.c_req or 1 - c_sat >= params.c_req):
            break

    # in any case (sat, not sat, undecided) return result
    return np.mean(rewards), np.mean(events), c_sat, successes, failures

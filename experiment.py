import numpy as np
import pandas as pd
from collections import deque
from world import DiscreteWorld, World, TargetWorld, MoveWorld, MoveWorldContinuous
from renderer import *
from datetime import datetime
import os
from evolution import EvolutionStrategies
from episode_runner import run_episode
from verification import verify
import parameters as params
from scipy.stats import beta


def run():
    directory = "results/{}".format(
        datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    if not os.path.exists(directory + '/trajectories/'):
        os.makedirs(directory + '/trajectories/')

    env = World(2)
    # env = MoveWorld()
    # env = MoveWorldContinuous()
    state = env.reset()
    model = EvolutionStrategies(inputs=env.state_dim, outputs=env.action_dim)

    experience = []
    log = []
    rewards = deque(maxlen=100)
    events = deque(maxlen=100)

    rewards_sat = deque(maxlen=100)
    events_sat = deque(maxlen=100)

    rewards_not_sat = deque(maxlen=100)
    events_not_sat = deque(maxlen=100)

    sats = []
    c_sats = deque(maxlen=100)
    p_sats = deque(maxlen=100)
    n_sat = 0
    c_sat_verification = 0

    for episode in range(params.episodes):
        reward, n_event, _ = run_episode(model, env)

        # update c_sat
        if params.constraint:
            sat = int(n_event <= params.constraint)
            if sat:
                rewards_sat.append(reward)
                events_sat.append(n_event)
            else:
                rewards_not_sat.append(reward)
                events_not_sat.append(n_event)
        else:
            sat = 1

        sats.append(sat)
        n_sat = sum(sats)  # += sat

        successes = n_sat + 1  # incl. prior
        failures = len(sats) - n_sat + 1  # incl. prior
        c_sat = 1. - beta(successes, failures).cdf(params.p_req)
        p_sat = beta(successes, failures).ppf(1 - params.c_req)

        # direct
        if params.calibration == 'direct':
            model.c_sat = c_sat
        elif params.calibration == 'hard':
            model.c_sat = 0 if c_sat < params.c_req else 1
        elif params.calibration == 'soft':
            model.c_sat = max(0, c_sat - params.c_req) / (1 - params.c_req)
        elif params.calibration == 'naive':
            model.c_sat = max(0, np.mean(sats) -
                              params.p_req) / (1 - params.p_req)

        # TODO: move to verify.py
        if params.verify and constraint is not None:
            if episode % 1000 == 0:
                # TODO: get true model: as method in evolution.py
                v_model = EvolutionStrategies(
                    inputs=env.state_dim, outputs=env.action_dim)
                for i, param in enumerate(v_model.parameters()):
                    param.data = model.master_weights[i]
                _, _, c_sat_verification, _, _ = verify(v_model, env)
                print(c_sat_verification)

        if params.constraint:
            model.log_reward(reward, -1 *
                             max(n_event - params.constraint, 0))
        else:
            model.log_reward(reward, 0)

        # log results
        rewards.append(reward)
        events.append(n_event)
        c_sats.append(c_sat)
        p_sats.append(p_sat)

        if episode % model.population_size == 0:
            log_entry = {'episode': episode,
                         'reward': '{0:.2f}'.format(np.mean(rewards)),
                         'r sat': '{0:.2f}'.format(np.mean(rewards_sat)),
                         'r not sat': '{0:.2f}'.format(np.mean(rewards_not_sat)),
                         'events': '{0:.4f}'.format(np.mean(events)),
                         'e sat': '{0:.4f}'.format(np.mean(events_sat)),
                         'e not sat': '{0:.4f}'.format(np.mean(events_not_sat)),
                         'n_sat': '{0:.4f}'.format(np.mean(sats)),
                         'c_sat': '{0:.4f}'.format(np.mean(c_sats)),
                         'p_sat': '{0:.4f}'.format(np.mean(p_sats)),
                         'c_sat_verification': '{0:.4f}'.format(np.mean(c_sat_verification)),
                         'constraint': params.constraint,
                         'calibration': params.calibration,
                         'lr': params.learning_rate}
            log.append(log_entry)
            df = pd.DataFrame(log)
            df.to_csv(directory + '/log.csv')
            print(log_entry)

            if params.render:
                ImgRenderer(directory + '/trajectories/' + str('%.4f' % reward) +
                            '_' + str('%.4f' % n_event) +
                            '_' + str(episode), env).render_img()


if __name__ == "__main__":
    for _ in range(100):
        for constraint in [None, 1, 4]:
            for calibration in ['soft']:
                params.constraint = constraint
                params.calibration = calibration
                run()

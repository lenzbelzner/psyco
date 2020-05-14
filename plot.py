import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import parameters
from scipy.stats import sem
sns.set_context('paper', font_scale=1.5)


def plot(label, filename):
    plt.ylabel(label)
    sns.despine()
    plt.tight_layout()
    plt.savefig(filename)


path = './results'
paths = next(os.walk(path))[1]

df = pd.DataFrame()
for path in paths:
    df = df.append(pd.read_csv('results/' + path + '/log.csv'))

df = df.fillna(0.)
df = df[df['episode'] > 100]

# df = df[df['constraint'] != 0]

df['constraint'] = ['$%s$' % x for x in df['constraint']]

df = df[df['calibration'] != 'direct']

df['calibration'] = ['MLE' if x == 'naive' else 'SNES' for x in df['calibration']]

ci = 'sd'

# hue = 'constraint'
hue = 'calibration'

plt.figure()
pal = sns.color_palette()
if hue == 'constraint':
    sns.lineplot(x="episode", y="reward", hue=hue, ci=ci,
                data=df, palette=[pal[2], pal[0], pal[1]])
else:
    sns.lineplot(x="episode", y="reward", hue=hue, ci=ci,
                data=df)
plt.ylabel('return')
sns.despine()
plt.tight_layout()
plt.savefig('plots/reward.png')

print(df.head())
df = df[df['constraint'] != '$0.0$']

plt.figure(figsize=(8, 4))
plt.subplot(121)
ax = sns.lineplot(x="episode", y="r sat", hue=hue, ci=ci, data=df)
plt.ylabel('return sat')
plt.subplot(122, sharey=ax)
sns.lineplot(x="episode", y="r not sat", hue=hue, ci=ci, data=df)
plt.ylabel('return not sat')
sns.despine()
plt.tight_layout()
plt.savefig('plots/return_sat.png')

df_zoom = df[df['episode'] >= 50000]

plt.figure(figsize=(8, 4))
plt.subplot(121)
ax = sns.lineplot(x="episode", y="r sat", hue=hue, ci=ci, data=df_zoom)
plt.ylabel('return sat')
plt.subplot(122, sharey=ax)
sns.lineplot(x="episode", y="r not sat", hue=hue, ci=ci, data=df_zoom)
plt.ylabel('return not sat')
sns.despine()
plt.tight_layout()
plt.savefig('plots/return_sat_zoom.png')

plt.figure(figsize=(8, 4))
plt.subplot(121)
ax = sns.lineplot(x="episode", y="e sat", hue=hue, ci=ci, data=df_zoom)
plt.ylabel('cost sat')
plt.subplot(122, sharey=ax)
sns.lineplot(x="episode", y="e not sat", hue=hue, ci=ci, data=df_zoom)
plt.ylabel('cost not sat')
sns.despine()
plt.tight_layout()
plt.savefig('plots/cost_sat.png')


plt.figure()
sns.lineplot(x="episode", y="n_sat", hue=hue,
             estimator=np.mean, ci=ci, data=df)
plt.axhline(parameters.p_req, ls='--')
plt.ylabel('fraction sat')
sns.despine()
plt.tight_layout()
plt.savefig('plots/n_sat.png')

plt.figure()
sns.lineplot(x="episode", y="c_sat", hue=hue,
             estimator=np.mean, ci=ci, data=df)
plt.axhline(parameters.c_req, ls='--')
plt.ylabel('$c_\mathrm{sat}$ while learning')
sns.despine()
plt.tight_layout()
plt.savefig('plots/c_sat.png')

plt.figure()
sns.lineplot(x="episode", y="p_sat", hue=hue,
             estimator=np.mean, ci=ci, data=df)
plt.axhline(parameters.p_req, ls='--')
plt.ylabel('$p_\mathrm{sat}$')
sns.despine()
plt.tight_layout()
plt.savefig('plots/p_sat.png')

plt.figure()
sns.lineplot(x="episode", y="c_sat_verification",
             hue=hue, estimator=np.mean, ci=ci, data=df)
plt.axhline(parameters.c_req, ls='--')
plt.ylabel('$c_\mathrm{sat}$ when verifying')
sns.despine()
plt.tight_layout()
plt.savefig('plots/c_sat_verification.png')

plt.show()

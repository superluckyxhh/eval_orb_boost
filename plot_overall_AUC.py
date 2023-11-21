import matplotlib.pyplot as plt
import numpy as np

names = ['ORB', 'ORB+Boost-B']
colors = ['red', 'purple']
linestyles = ['-', '-.']

orb_auc = np.load('/home/xiaohunan/WorkSpace/MyCode/test_featurebooster/eval_dumps/npy/orb_mnn_aucs.npy')
orb_boost_auc = np.load('/home/xiaohunan/WorkSpace/MyCode/test_featurebooster/eval_dumps/npy/orb_boost_mnn_aucs.npy')

# Plot
n_i = 52
n_v = 56
plt_lim = [1, 9]
plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)
plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=25)

plt.figure(figsize=(5, 5))
# ((n_i + n_v) * 5
plt.plot(plt_rng, [orb_auc[thr] for thr in plt_rng], color=colors[0], ls=linestyles[0], linewidth=2, label=names[0])
plt.plot(plt_rng, [orb_boost_auc[thr] for thr in plt_rng], color=colors[1], ls=linestyles[1], linewidth=2, label=names[1])

plt.title('Overall')
plt.xlim(plt_lim)
plt.xticks(plt_rng)
plt.ylabel('H est. AUC')
plt.ylim([0, 1])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend()

plt.savefig('overall_AUC.png', bbox_inches='tight', dpi=300)


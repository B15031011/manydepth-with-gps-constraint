import numpy as np
import matplotlib.pyplot as plt
import pickle

# make data:
all_data = []
sc_ratios=np.genfromtxt(r'C:\Users\user\Desktop\manydepth\ratios\sc_ratios').astype(np.float32)
print(np.median(sc_ratios),np.std(sc_ratios))
all_data.append(sc_ratios)
#
monodepth_ratios=np.genfromtxt(r'C:\Users\user\Desktop\manydepth\ratios\monodepth2_ratios.txt').astype(np.float32)
print(np.median(monodepth_ratios),np.std(monodepth_ratios))
all_data.append(monodepth_ratios)

manydepth_ratios=np.load(r'C:\Users\user\Desktop\manydepth\ratios\manydepth_ratios.npy')
print(np.median(manydepth_ratios),np.std(manydepth_ratios))
all_data.append(manydepth_ratios)

adjust_ratios=np.load(r'C:\Users\user\Desktop\manydepth\ratios\adjust_weight_ratios.npy')
all_data.append(adjust_ratios)
x=[i for i in range(1,698)]

fig, axes = plt.subplots()

axes.plot(x,manydepth_ratios,'r',label='Manydepth',linewidth=0.7)
axes.plot(x,monodepth_ratios,'y',label='Monodepth2',linewidth=0.7)
axes.plot(x,sc_ratios,'b',label='SC_SfMLearner',linewidth=0.7,linestyle='--')
axes.plot(x,adjust_ratios,'g',label='Ours',linewidth=0.8,linestyle='-.')
plt.legend(loc='lower left',bbox_to_anchor=(0,0.1))

axes.set_ylabel('Scale')
axes.set_xlabel('Image index')

axes.set_yticks([5,10,15,20,25,30,35,40,45,50])
axes.set_xticks([50,150,250,350,450,550,650,700])

plt.savefig(r'C:\Users\user\Desktop\毕业论文材料\图片\ratios2.png')
plt.show()
#
# axes.violinplot(all_data,widths=1,
#                    showmeans=False,showmedians=True,showextrema=True
#                    )
# # adding horizontal grid lines
#
#
# axes.yaxis.grid(False)
# # axes.set_xticks([y + 1 for y in range(len(all_data))], )
# axes.set_yticks([5,10,15,20,25,30,35,40,45,50])
# axes.set_ylabel('Scale factors')
#
# plt.setp(axes, xticks=[y + 1 for y in range(len(all_data))],
#          xticklabels=['SC_SfMLearner', 'Monodepth2', 'Manydepth', 'ours']
#          )
# plt.savefig(r'C:\Users\user\Desktop\毕业论文材料\图片\ratios.png')
# plt.show()





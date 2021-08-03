'''
from numpy import *
import numpy as np
import math
import matplotlib.pyplot as plt

def average_plot(list,margin=50):
    avg_list = []
    for i in range(list.shape[0] - margin):
        temp = 0
        for j in range(margin):
            temp = temp + list[i + j]
        temp = np.float32(temp) / margin
        avg_list.append(temp)
    avg_list = np.asarray(avg_list, dtype=np.float32)
    return avg_list

#r1 = np.load('/home/chen-2/model_savedir/pong_rpf00/reward.npy')
r2 = np.load('/home/chen-2/model_savedir/pong_rpf_safe_t_0.1_c_0.0100/reward.npy')
r3 = np.load('/home/chen-2/model_savedir/pong_rpf_safe_t_0.1_c_0.0500/reward.npy')
r4 = np.load('/home/chen-2/model_savedir/pong_rpf_safe_t_0.1_c_0.100/reward.npy')
#r5 = np.load('/home/chen-2/model_savedir/pong_rpf_safe_t_0.1_c_0.500/reward.npy')

#r1 = average_plot(r1)
r2 = average_plot(r2)
r3 = average_plot(r3)
r4 = average_plot(r4)
#r5 = average_plot(r5)


plt.figure()

plt.ylabel('reward')
#plt.plot(r1, 'r') # plotting t, a separately
plt.plot(r2, 'r') # plotting t, b separately
plt.plot(r3, 'g') # plotting t, b separately
plt.plot(r4, 'b') # plotting t, b separately
#plt.plot(r5, 'c') # plotting t, b separately
#plt.xlim([0, 1750])

#plt.legend(["No Safety included", "Safety with LCB_constant=0.01", "Safety with LCB_constant=0.05","Safety with LCB_constant=0.1","Safety with LCB_constant=0.5"], loc ="lower right")
#plt.legend(["Safety with LCB_constant=0.01", "Safety with LCB_constant=0.05","Safety with LCB_constant=0.1","Safety with LCB_constant=0.5"], loc ="lower right")
plt.legend(["Safety with LCB_constant=0.01","Safety with LCB_constant=0.05","Safety with LCB_constant=0.1"], loc ="lower right")

plt.savefig('./reward_comparision_temp.png')

plt.figure()

#u1 = np.load('/home/chen-2/model_savedir/pong_rpf00/uncertainity.npy')
u2 = np.load('/home/chen-2/model_savedir/pong_rpf_safe_t_0.100_c_0.01/uncertainity.npy')
u3 = np.load('/home/chen-2/model_savedir/pong_rpf_safe_t_0.1_c_0.0500/uncertainity.npy')
u4 = np.load('/home/chen-2/model_savedir/pong_rpf_safe_t_0.1_c_0.100/uncertainity.npy')
u5 = np.load('/home/chen-2/model_savedir/pong_rpf_safe_t_0.1_c_0.500/uncertainity.npy')



#u1 = average_plot(u1)
u2 = average_plot(u2)
#u3 = average_plot(u3)
u4 = average_plot(u4)
u5 = average_plot(u5)




plt.ylabel('uncertainity')
#plt.plot(u1, 'r') # plotting t, a separately
plt.plot(u2, 'b') # plotting t, b separately
plt.plot(u3, 'g') # plotting t, b separately
plt.plot(u4, 'm') # plotting t, b separately
plt.plot(u5, 'c') # plotting t, b separately

#plt.xlim([0, 1750])


#plt.legend(["Safety with LCB_constant=0.01", "Safety with LCB_constant=0.05","Safety with LCB_constant=0.1","Safety with LCB_constant=0.5"], loc ="upper right")
plt.legend(["Safety with LCB_constant=0.01","Safety with LCB_constant=0.1","Safety with LCB_constant=0.5"], loc ="upper right")

plt.savefig('./uncertainity_comparision.png')

plt.figure()

#u1 = np.load('/home/chen-2/model_savedir/pong_rpf00/uncertainity.npy')
u2 = np.load('/home/chen-2/model_savedir/pong_rpf_safe_t_0.100_c_0.01/no_of_uncertainity.npy')
u3 = np.load('/home/chen-2/model_savedir/pong_rpf_safe_t_0.1_c_0.0500/no_of_uncertainity.npy')
u4 = np.load('/home/chen-2/model_savedir/pong_rpf_safe_t_0.1_c_0.100/no_of_uncertainity.npy')
u5 = np.load('/home/chen-2/model_savedir/pong_rpf_safe_t_0.1_c_0.500/no_of_uncertainity.npy')



#u1 = average_plot(u1)
u2 = average_plot(u2)
u3 = average_plot(u3)
u4 = average_plot(u4)
u5 = average_plot(u5)




plt.ylabel('uncertainity')
#plt.plot(u1, 'r') # plotting t, a separately
plt.plot(u2, 'r') # plotting t, b separately
plt.plot(u3, 'b') # plotting t, b separately
#plt.plot(u4, 'g') # plotting t, b separately
plt.plot(u5, 'c') # plotting t, b separately

#plt.xlim([0, 1750])


plt.legend(["Safety with LCB_constant=0.01","Safety with LCB_constant=0.1","Safety with LCB_constant=0.5"], loc ="lower right")

plt.savefig('./no_of_uncertainity_comparision.png')




from numpy import *
import numpy as np
import math
import matplotlib.pyplot as plt

def average_plot(list,margin=50):
    avg_list = []
    for i in range(list.shape[0] - margin):
        temp = 0
        for j in range(margin):
            temp = temp + list[i + j]
        temp = np.float32(temp) / margin
        avg_list.append(temp)
    avg_list = np.asarray(avg_list, dtype=np.float32)
    return avg_list

r1 = np.load('/home/chen-2/model_savedir/pong_rpf00/reward.npy')
r2 = np.load('/home/chen-2/model_savedir/pong_rpf_safe_t_0.1_c_0.001_safe_action00/reward.npy')
r3 = np.load('/home/chen-2/model_savedir/pong_rpf_safe_t_0.1_c_0.05_safe_action00/reward.npy')
r4 = np.load('/home/chen-2/model_savedir/pong_rpf_safe_t_0.1_c_0.5_safe_action00/reward.npy')


r1 = average_plot(r1)
r2 = average_plot(r2)
r3 = average_plot(r3)
r4 = average_plot(r4)




plt.figure()

plt.ylabel('reward')
#plt.plot(r1, 'r') # plotting t, b separately
plt.plot(r2, 'b') # plotting t, b separately
plt.plot(r3, 'g') # plotting t, b separately
plt.plot(r4, 'c') # plotting t, b separately
plt.axhline(y=14, color='y', linestyle='-')

plt.xlim([0, 750])



#plt.legend(["Unsafe","Safety with LCB_constant=0.001","Safety with LCB_constant=0.05","Safety with LCB_constant=0.5"], loc ="lower right")
plt.legend(["Safety with LCB_constant=0.001","Safety with LCB_constant=0.05","Safety with LCB_constant=0.5"], loc ="lower right")

plt.savefig('./reward_comparision_temp2.png')

plt.figure()

u1 = np.load('/home/chen-2/model_savedir/pong_rpf00/uncertainity.npy')
u2 = np.load('/home/chen-2/model_savedir/pong_rpf_safe_t_0.1_c_0.001_safe_action00/uncertainity.npy')
u3 = np.load('/home/chen-2/model_savedir/pong_rpf_safe_t_0.1_c_0.05_safe_action00/uncertainity.npy')
u4 = np.load('/home/chen-2/model_savedir/pong_rpf_safe_t_0.1_c_0.5_safe_action00/uncertainity.npy')



u1 = average_plot(u1)
u2 = average_plot(u2)
u3 = average_plot(u3)
u4 = average_plot(u4)





plt.ylabel('uncertainity')
plt.plot(u1, 'r') # plotting t, b separately
plt.plot(u2, 'b') # plotting t, b separately
plt.plot(u3, 'g') # plotting t, b separately
plt.plot(u4, 'c') # plotting t, b separately



plt.xlim([0, 1750])


#plt.legend(["Safety with LCB_constant=0.01", "Safety with LCB_constant=0.05","Safety with LCB_constant=0.1","Safety with LCB_constant=0.5"], loc ="upper right")
plt.legend(["Unsafe","Safety with LCB_constant=0.001","Safety with LCB_constant=0.05","Safety with LCB_constant=0.5"], loc ="upper right")

plt.savefig('./uncertainity_comparision2.png')

'''

from numpy import *
import numpy as np
import math
import matplotlib.pyplot as plt

def average_plot(list,margin=50):
    avg_list = []
    for i in range(list.shape[0] - margin):
        temp = 0
        for j in range(margin):
            temp = temp + list[i + j]
        temp = np.float32(temp) / margin
        avg_list.append(temp)
    avg_list = np.asarray(avg_list, dtype=np.float32)
    return avg_list

r2 = np.load('/home/chen-2/model_savedir/breakout_rpf_safe06/reward.npy')
r3 = np.load('/home/chen-2/model_savedir/breakout_rpf01/reward.npy')


r2 = average_plot(r2)
r3 = average_plot(r3)



l = []
i = 500
while(i < 2500):
    l.append(i)
    i = i + 500

plt.figure()
plt.ylabel('reward')
plt.xlim(0, 5000)

#plt.plot(r1, 'r') # plotting t, b separately
plt.plot(r2, 'b') # plotting t, b separately
plt.plot(r3, 'g') # plotting t, b separately
plt.axhline(y=264, color='y', linestyle='-')


#plt.xlim([0, 750])



#plt.legend(["Unsafe","Safety with LCB_constant=0.001","Safety with LCB_constant=0.05","Safety with LCB_constant=0.5"], loc ="lower right")
plt.legend(["Aget with help from Baseline ","Normal Agent","Baseline Value"], loc ="lower right")

plt.savefig('./breakout_reward.png')


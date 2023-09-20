#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 09:08:23 2022

@author: tjards
"""
import matplotlib.pyplot as plt

#%%

fig, ax = plt.subplots()
ax.plot(t_all[4::],metrics_order_all[4::,1])

ax.set(xlabel='time (s)', ylabel='mean distance from target [m]',
       title='Distance from Target')
ax.grid()

#fig.savefig("test.png")
plt.show()



#%% Produce plots
# --------------

start = 10

#%% Convergence to target 
#-------------------------
fig, ax = plt.subplots()
ax.plot(t_all[start::],metrics_order_all[start::,1],'-b')
ax.plot(t_all[start::],metrics_order_all[start::,5],':b')
ax.plot(t_all[start::],metrics_order_all[start::,6],':b')
ax.fill_between(t_all[start::], metrics_order_all[start::,5], metrics_order_all[start::,6], color = 'blue', alpha = 0.1)
#note: can include region to note shade using "where = Y2 < Y1
ax.set(xlabel='Time [s]', ylabel='Mean Distance to Target [m]',
       title='Convergence to Target')
#ax.plot([70, 70], [100, 250], '--b', lw=1)
#ax.hlines(y=5, xmin=Ti, xmax=Tf, linewidth=1, color='r', linestyle='--')
ax.set_xlim([0, Tf])
ax.grid()

#fig.savefig("test.png")
plt.show()


#%% Energy
# ------------
fig, ax = plt.subplots()

# set forst axis
ax.plot(t_all[start::],metrics_order_all[start::,7],'-g')
#ax.plot(t_all[4::],metrics_order_all[4::,7]+metrics_order_all[4::,8],':g')
#ax.plot(t_all[4::],metrics_order_all[4::,7]-metrics_order_all[4::,8],':g')
ax.fill_between(t_all[start::], metrics_order_all[start::,7], color = 'green', alpha = 0.1)

#note: can include region to note shade using "where = Y2 < Y1
ax.set(xlabel='Time [s]', title='Energy Consumption')
ax.set_ylabel('Total Acceleration [m^2]', color = 'g')
ax.tick_params(axis='y',colors ='green')
ax.set_xlim([0, Tf])
ax.set_ylim([0, 10])
#ax.plot([70, 70], [100, 250], '--b', lw=1)
#ax.hlines(y=5, xmin=Ti, xmax=Tf, linewidth=1, color='r', linestyle='--')
total_e = np.sqrt(np.sum(cmds_all**2))
# ax.text(3, 2, 'Total Energy: ' + str(round(total_e,1)), style='italic',
#         bbox={'facecolor': 'green', 'alpha': 0.1, 'pad': 1})


# set second axis
ax2 = ax.twinx()
#ax2.set_xlim([0, Tf])
#ax2.set_ylim([0, 1])
ax2.plot(t_all[start::],1-metrics_order_all[start::,0], color='tab:blue', linestyle = '--')
#ax2.fill_between(t_all[4::], 1-metrics_order_all[4::,0], color = 'tab:blue', alpha = 0.1)
ax2.set(title='Energy Consumption')
ax2.set_ylabel('Disorder of the Swarm', color='tab:blue')
#ax2.invert_yaxis()
ax2.tick_params(axis='y',colors ='tab:blue')
ax2.text(Tf-Tf*0.3, 0.1, 'Total Energy: ' + str(round(total_e,1)), style='italic',
        bbox={'facecolor': 'green', 'alpha': 0.1, 'pad': 1})

ax.grid()
#fig.savefig("test.png")
plt.show()

#%% Spacing
# ---------

fig, ax = plt.subplots()

# set forst axis
ax.plot(t_all[start::],metrics_order_all[start::,9],'-g')
ax.plot(t_all[start::],metrics_order_all[start::,11],'--g')
ax.fill_between(t_all[start::], metrics_order_all[start::,9], metrics_order_all[start::,11], color = 'green', alpha = 0.1)

#note: can include region to note shade using "where = Y2 < Y1
ax.set(xlabel='Time [s]', title='Spacing between Agents [m]')
ax.set_ylabel('Mean Distance [m]', color = 'g')
ax.tick_params(axis='y',colors ='green')
#ax.set_xlim([0, Tf])
#ax.set_ylim([0, 40])
total_e = np.sqrt(np.sum(cmds_all**2))

# set second axis
ax2 = ax.twinx()
#ax2.set_xlim([0, Tf])
#ax2.set_ylim([0, 100])
ax2.plot(t_all[start::],metrics_order_all[start::,10], color='tab:blue', linestyle = '-')
ax2.set_ylabel('Number of Connections', color='tab:blue')
ax2.tick_params(axis='y',colors ='tab:blue')
#ax2.invert_yaxis()

ax.legend(['Within Range', 'Oustide Range'], loc = 'upper left')
ax.grid()
#fig.savefig("test.png")
plt.show()


# #%% departures

# # radii from target
# radii = np.zeros([states_all.shape[2],states_all.shape[0]])
# for i in range(0,states_all.shape[0]):
#     for j in range(0,states_all.shape[2]):
#         radii[j,i] = np.linalg.norm(states_all[i,:,j] - targets_all[i,:,j])
        
# fig, ax = plt.subplots()
# for j in range(0,states_all.shape[2]):
#     for k in range(0,t_all.shape[0]):
#         if pins_all[k,j,j] == 1:
#             ax.plot(t_all[k-1:k],radii[j,k-1:k].ravel(),'-b')
#         else:
#             ax.plot(t_all[k-1:k],radii[j,k-1:k].ravel(),'-r')
            


# ax.set(xlabel='Time [s]', ylabel='Distance from Target for Each Agent [m]',
#        title='Distance from Target')
# plt.axhline(y = 5, color = 'k', linestyle = '--')
# plt.show()


#%% spacing 
from scipy.spatial.distance import cdist
seps_all = np.zeros((states_all.shape[0],states_all.shape[2],states_all.shape[2]))
for i in range(0,states_all.shape[0]):
    seps_all[i,:,:]=cdist(states_all[i,0:3,:].transpose(), states_all[i,0:3,:].transpose())    

fig, ax = plt.subplots()
ax2 = ax.twinx()

line_list = ["-","--",":"]
l1 = 0

for i in [3]: #range(0,states_all.shape[2]):
    for j in [5,6,2]: #range(0,states_all.shape[2]):
        ax.plot(t_all[int(35/0.02):int(75/0.02)],seps_all[int(35/0.02):int(75/0.02),i,j],line_list[l1], color = 'tab:blue')
        cmds_sum = cmds_all[int(35/0.02):int(75/0.02),0,3]+cmds_all[int(35/0.02):int(75/0.02),1,3]+cmds_all[int(35/0.02):int(75/0.02),2,3]        
        #ax2.plot(t_all[int(35/0.02):int(75/0.02)],cmds_all[int(35/0.02):int(75/0.02),0,3], '--', linewidth = 0.7, color = 'tab:red')
        #ax2.plot(t_all[int(35/0.02):int(75/0.02)],cmds_all[int(35/0.02):int(75/0.02),1,3], '--',linewidth = 0.7, color ='tab:red')
        #ax2.plot(t_all[int(35/0.02):int(75/0.02)],cmds_all[int(35/0.02):int(75/0.02),2,3], '--', linewidth = 0.7, color = 'tab:red')
        ax2.plot(t_all[int(35/0.02):int(75/0.02)],cmds_sum, '-', linewidth = 1, color = 'tab:green')
        l1+=1


#ax.axvline(x = 20, color = 'black', linestyle = '--')
ax.axvline(x = 45, color = 'black', linestyle = '--')
#ax.axvline(x = 75, color = 'black', linestyle = '--')
#ax.axvline(x = 100, color = 'black', linestyle = '--')
#ax.axvline(x = 115, color = 'black', linestyle = '--')

#ldg = list(range(0,states_all.shape[2]))
ax.legend(['Leading','Lagging - Departing','Lagging - Compensating'], loc = 'upper center')

ax.tick_params(axis='y',colors ='tab:blue')
ax2.tick_params(axis='y',colors ='tab:green')

ax.set(xlabel='Time [s]',title='Compensation after Agent Departure')
#ax2.set(title='Control Inputs')
ax2.set_ylabel('Sum of Control Inputs [m^2]', color='tab:green')
ax.set_ylabel('Separation [m]', color='tab:blue')
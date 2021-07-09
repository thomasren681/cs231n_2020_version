import numpy as np
import matplotlib.pyplot as plt
import torch

# x = np.linspace(-1,1)
# y1 = x**3
# y2 = x**2+2*x+1
#
# plt.figure(num=3, figsize=(10,10))
# line1,=plt.plot(x, y2, label = 'quadratic')
# line2,=plt.plot(x,y1, color = 'red', linestyle = '--', label = 'cubic')
#
# # plt.xticks([-1,0.5,1])
# # plt.yticks([-1,0,1,2,3],
# #            ['a', 'b', 'c', 'd', 'e'])
#
# ax = plt.gca()
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# ax.spines['bottom'].set_position(('data',0))
# ax.spines['left'].set_position(('data',0))
#
#
# plt.legend(handles = [line1,line2, ], labels = ['first', 'second'], loc = 'best')
#
# x0 = 1
# y0 = x0**2+2*x0+1
#
# plt.scatter(x0,y0,s=250,color = 'black')
# plt.plot([x0,x0], [y0,0], 'k--', lw = 3)
# plt.plot([x0,0], [y0,y0], 'k--', lw = 3)
#
# #method 1: annotation based on the point's position
# plt.annotate(r'$this\ point\ is\ (1,4)$', xy=(x0,y0), xycoords = 'data', xytext = (+40, -40),
#              textcoords = 'offset points', fontsize = 20, arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3, rad = .3'))
#
# #method 2: annotation based on setted position on the coordinate
# #use \ to print greek letter and _ to add subscript
# plt.text(0.25,-0.5, r'$here\ is\ the\ text\ plot\ , \mu\ \sigma_i\ \epsilon_j$',
#          fontdict = {'size':16, 'color':'red'})
#
#
# # here we can change the labels of the x and y coordinates
# for label in ax.get_xticklabels() + ax.get_yticklabels():
#     label.set_fontsize(20)
#     label.set_bbox(dict(facecolor = 'white', edgecolor = 'black', alpha = 0.5))
#
#
# num_points = 1024
# X = np.random.normal(loc=0, scale=1, size=num_points)
# Y = np.random.normal(loc=0, scale=1, size=num_points)
#
#
#
#
#
# plt.show()

import os

N = [6.9, -3.4, -2.1, -5.3, -51.8, -8.8, -14.3, -15.7, -12.5, -47.6, -14.2, -18.7, -20.0, -16.1, -47.6]
for i in range(len(N)):
    print(i+1)
    print(0.775*10**(N[i]/20))
    print(1.414*0.775*10**(N[i]/20))

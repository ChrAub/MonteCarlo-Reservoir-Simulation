# -*- coding: utf-8 -*-
import scipy
import matplotlib.pyplot as plt
import math
import numpy as np

# Input parameters for a 2D reservoir. 
# There are 22 different layers with dimensions in x and y direction (geometry) that belong to a certain category (cat). 
# Each layer has distinct petropysical properties that are given as expected value (petrophysics) and standard deviation (petrophysicsstdev). Such values are obtained from measurements/scientific literature. 
geometry = [[1000,22.8],[1000,27.8],[893,12],[155,15],[900,13],[1000,17],[1000,7.4],[825,17.5],[175,18],[1000,12.8],[1000,18.6],[1000,19],[1000,9.5],[1000,4.4],[1000,9.15],[1000,15.5],[611,21.7],[389,23],[1000,14.5],[1000,8.1],[1000,25.5],[370,6]]
petrophysics = [[0.16,0.09],[0.233,0.07],[0.22,0.076],[0.001,0.0007],[0.16,0.16],[0.15,0.14],[0.1,0.68]]
petrophysicsstdev = [[0.0016,0.028],[0.025,0.014],[0.018,0.043],[0,0],[0.039,0.05],[0.015,0.014],[0,0]]
cat = [7,6,3,4,1,7,3,6,1,2,3,5,2,4,2,4,1,2,3,4,2,4]

numberruns = 10 # number of runs for the Monte-Carlo-Simulation; anything below 1,000 is not useful, 10,000 is a good compromise between time and meaningfulness

oillayers = 0

cat2 = [] # categories 4 and 7 don't contain oil, so they must be sorted out
for i in cat:
    if i != 4 and i != 7:
        cat2.append(i)
        oillayers += 1

saver = [] # save total oil for each MC run
share = [[]]*oillayers # save share of each oillayer for each MC run                      

for i in range(0,numberruns):
    helper = []
    shareh = []
    for j in range(0,len(cat)):
        if cat[j] != 4 and cat[j] != 7: # ignore layers 4 and 7 as they don't contain oil
            length = geometry[j][0] #lengh is assumed to be known with certainty
            thickness = geometry[j][1] #thickness is assumed to be known with certainty
            por = scipy.stats.norm(petrophysics[cat[j]-1][0], petrophysicsstdev[cat[j]-1][0]).rvs() # generate random vaue for porosity
            sw = scipy.stats.norm(petrophysics[cat[j]-1][1], petrophysicsstdev[cat[j]-1][1]).rvs() # generate random value for water saturation
            pseudo = abs(length*thickness*por*(1-sw))
            helper.append(pseudo)
    for j in range(0,len(helper)):
        share[j].append(helper[j]/sum(helper))
        shareh.append(j/sum(helper))
    depth = 1 # set 3rd dimension to unity
    bo = scipy.stats.norm(1.04,0.03).rvs() # generate random value for FVF only once for each MC cycle
    total = abs(depth/bo*sum(helper))
    saver.append(total) # oil guess for each MC run
   
total_share = [] 
total_std = []
 
for i in share: #calculate a distribution of values for share
        total_share.append(sum(i)/len(i)) # mean
        total_std.append(np.std(i)) # standard deviation

plot_std = [] # parameters for plotting of share 
plot_mean = [] 

for i in range(0,7): #for each category sum um the mean and stdev of share
    helper1 = []
    helper2 = []
    for j in range(0,len(cat2)):
        if i == cat2[j]:
            helper1.append(total_share[j])
            helper2.append(total_std[j])
    plot_mean.append(sum(helper1))
    plot_std.append(np.mean(helper2))

plot_std = plot_std[1:4] + plot_std[5:7] # ignore layers that don't contain oil
plot_mean = plot_mean[1:4] + plot_mean[5:7]                        
                                                                                                                                                    
(a,b,c) = scipy.stats.mstats.mquantiles(saver,[0.1,0.5,0.9]) # get quantiles from data

shape, loc, scale = scipy.stats.lognorm.fit(saver, floc=0) # get parameters for best fitting lognorm distribution
loc2, scale2, = scipy.stats.norm.fit(saver) # get parameters for best fitting normal distribution
mu = np.log(scale)
mu2 = loc2
sigma = shape
sigma2 = scale2

# plot results into 4 charts
# 221 is a histogram of random data of oil in place 
# 222 uses a normal and a log-normal distribution to fit the random data as pdf
# 223 is a cdf of the the random results and depicts P90, P50, P10 and best guess
# 224 shows the rock types where the oil is placed initially  
plt.subplot(221)
n, bins, patches = plt.hist(saver,bins=50,range=(min(saver),max(saver)))
plt.title('STOOIP simulated')
plt.xlabel('oil in place [cubic meter]')
plt.ylabel('PDF')
plt.xlim(xmin=min(saver))
plt.xlim(xmax=max(saver))
plt.subplot(223)
x = np.linspace(min(saver), max(saver), num=100)
p1 = plt.plot(x, scipy.stats.lognorm.pdf(x, shape, loc=0, scale=scale), 'b', linewidth=1) 
p2 = plt.plot(x, scipy.stats.norm.pdf(x, mu2, sigma2), 'r', linewidth=1)
plt.legend((p2[0],p1[0]),('normal','log-normal'))
plt.title('Best Fitting Distributions')
plt.xlabel('oil in place [cubic meter]')
plt.ylabel('PDF')
plt.xlim(xmin=min(saver))
plt.xlim(xmax=max(saver))
plt.subplot(222)
n, bins, patches = plt.hist(saver,bins=200,range=(min(saver),max(saver)),cumulative=True)
plt.title('Cumulative STOOIP')
plt.xlabel('oil in place [cubic meter]')
plt.ylabel('number of simulations')
plt.xlim(xmin=min(saver))
plt.xlim(xmax=max(saver))
l1 = plt.axvline(x = a, color = 'r', linewidth = 3, ymin = 0, ymax=8000)
l2 = plt.axvline(x = b, color = 'r', linewidth = 3, ymin = 0, ymax=8000)
l3 = plt.axvline(x = c, color = 'r', linewidth = 3, ymin = 0, ymax=8000)
l4 = plt.axvline(x =36214, color = 'r', linewidth = 3, ymin = 0, ymax= 8000)
plt.text(s = 'P90', x = a + 200, y = 8000, color = 'r')
plt.text(s = 'P50', x = b + 200, y = 8000, color = 'r')
plt.text(s = 'P10', x = c + 200, y = 8000, color = 'r')
plt.text(s = 'det', x = 36414, y = 8000, color = 'r')
plt.subplot(224)
w = 0.1
p1 = plt.bar(1,plot_mean[0],w,color = 'r',yerr = plot_std[0],ecolor = 'k')
p2 = plt.bar(1,plot_mean[1],w,bottom=plot_mean[0],color = 'y',yerr = plot_std[1],ecolor = 'k')
p3 = plt.bar(1,plot_mean[2],w,bottom=plot_mean[0]+plot_mean[1],color = 'b',yerr = plot_std[2],ecolor = 'k')
p4 = plt.bar(1,plot_mean[3],w,bottom=plot_mean[0]+plot_mean[1]+plot_mean[2],color = 'c',yerr = plot_std[3],ecolor = 'k')
p5 = plt.bar(1,plot_mean[4],w,bottom=plot_mean[0]+plot_mean[1]+plot_mean[2]+plot_mean[3],color = 'm')
plt.title('Where does the oil come from?') 
plt.xlabel('oil in place [cubic meter]')
plt.ylabel('share')
plt.legend((p5[0],p4[0],p3[0],p2[0],p1[0]),('carbonatic grainstone','conglomerat','coarse sandstone','medium sandstone','fine sandstone'))
plt.xlim(xmin=0.9)
plt.xlim(xmax=1.3)
plt.axis('off')

plt.show()


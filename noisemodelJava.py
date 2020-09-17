# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:49:50 2020

@author: alexs
"""
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np

#importing data and stitching quarters 13-19

search = lk.search_lightcurvefile('KIC3656476', cadence = 'short', mission = 'Kepler')
files = search[13:19].download_all()


lc = files.PDCSAP_FLUX.stitch()
lc = lc.remove_nans().remove_outliers().flatten(window_length=3351).remove_outliers().normalize(unit = 'ppm')


pg = lc.to_periodogram(method='lombscargle', normalization='psd')

#PARSEVALS CORRECTION
flux = lc.flux
pow_ts = np.sum((flux-np.nanmean(flux))**2)/len(flux)
pow_psd = np.sum(pg.power.value)*(pg.frequency.value[2]-pg.frequency.value[1])
correction = pow_ts/pow_psd

print('psd power', pow_psd)
print('time series power', pow_ts)

correction = np.round(correction, decimals = 13)
pg = (correction)*pg
print('correction = ', correction)


pg.plot(scale = 'log')


#making nu and power dimensionless in order to be able to work with them without errors
nu    = pg.frequency.value
power = pg.power.value #parseval correction

#defining new ranges
nu_gran=nu[(nu > 50.) & (nu <1000.)] #picking a range for granulation effects, but not indexing them
nu_act=nu[(nu > 0.) & (nu < 50.)]
nu_b=nu[(nu > 5000.) & (nu < 10**4.)]

power_gran=power[(nu > 50.) & (nu <1000.)]
power_act=power[(nu > 0.) & (nu < 50.)]
power_b=power[(nu > 5000.) & (nu < 10**4.)]


#plotting only in select range
plt.plot(nu_gran, power_gran, 'm.', markersize = 3) 
plt.plot(nu_act,power_act, 'b.', markersize = 3)
plt.plot(nu_b,power_b, 'y.', markersize = 3)


#intitial guesses
tgran = 100000.
tact  = 1.0e8
pgran = 100
pact  = 100
b_guess     = np.mean(power[(nu > 5000.)])

from scipy.optimize import curve_fit

def granulation(nu_gran, p_gran, t_gran):
    return p_gran/(1+(t_gran*1e-6*nu_gran)**2)

p0=np.array([pgran, tgran])

gran_opt, gran_cov = curve_fit(granulation, nu_gran, power_gran, p0) #curve_fit(function, x,y,first guess), then this thing iterates it and finds the value
#gran opt returns (pgran,tgran) but optimized, ignore gran cov


#defining the 3 types of noise and then fitting the data separately to each component
def activity(nu_act, p_act, t_act):
    return p_act /(1+(t_act*1e-6*nu_act)**2)

act_opt, act_cov = curve_fit(activity, nu_act, power_act, p0=np.array([pact, tact]))


def whitenoise(nu_b, b1):
    return b1

b_opt, b_cov = curve_fit(f = whitenoise, xdata = nu_b, ydata = power_b, p0 = (b_guess))


bg=  granulation(nu,gran_opt[0],gran_opt[1])+activity(nu,act_opt[0],act_opt[1]) + b_opt[0]

def background_noise(nu, p_gran, t_gran,p_act, t_act, b1):
    return p_gran/(1+(t_gran*1e-6*nu)**2) + p_act /(1+(t_act*1e-6*nu)**2) + b1


#adding all the components together and re-optimising
b_n_opt, b_n_cov = curve_fit(background_noise, nu, power, p0=np.array([gran_opt[0],gran_opt[1],act_opt[0],act_opt[1],b_opt[0]]))
combined_background=background_noise(nu,b_n_opt[0], b_n_opt[1], b_n_opt[2], b_n_opt[3], b_n_opt[4])
back_g=plt.plot(nu, combined_background,'g')


#printing the optimized parameters
print("granulation parameters= ", gran_opt) 
print("activity parameters = ", act_opt)
print("white noise = ", b_opt)
print("total fit= ", b_n_opt)

#if you want to plot lines individually with optimized values:
g = granulation(nu, b_n_opt[0], b_n_opt[1]) 
a = activity(nu, b_n_opt[2], b_n_opt[3])
b=whitenoise(nu_b,b_n_opt[4])



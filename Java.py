# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:50:30 2020

@author: alexs
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:37:45 2020

@author: alexs
"""

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np

search = lk.search_lightcurvefile('KIC3656476', cadence = 'short', mission = 'Kepler')
files = search[13:19].download_all()

lc = files.PDCSAP_FLUX.stitch()
lc = lc.remove_nans().remove_outliers().flatten(window_length=3351).remove_outliers().normalize(unit = 'ppm')

pg = lc.to_periodogram(method='lombscargle', normalization='psd') 
pg.plot(scale = 'log')


#PARSEVALS CORRECTION
flux = lc.flux
pow_ts = np.sum((flux-np.nanmean(flux))**2)/len(flux)
pow_psd = np.sum(pg.power.value)*(pg.frequency.value[2]-pg.frequency.value[1])
correction = pow_ts/pow_psd

print('psd power', pow_psd)
print('time series power', pow_ts)
print('correction =', correction)

correction = np.round(correction, decimals = 13)


pg    = (correction)*pg
nu    = pg.frequency.value
power = pg.power.value 

#noise fit taken from noisemodelJava 
p_gran, t_gran = [6.14704874e+00,  2.24076213e+03]
p_act, t_act =  [6.18192301e+01, -1.49698126e+05]
b =  [1.40219032e+00]

background_fit= p_gran/(1+(t_gran*1e-6*nu)**2) + p_act /(1+(t_act*1e-6*nu)**2) + b

plt.figure(1)
plt.plot(nu, background_fit)

#Using Hekker et al. to get Amax: 

#subbing in values
delta_nu = 93.194 #freq is already in micro Hz, from Lund et al.
delta_nu_upper = 0.018
delta_nu_lower = 0.02
nu_max_paper = 1925
nu_max_paper_upper = 7
nu_max_paper_lower = 6.3


#using the largest error
if delta_nu_upper > delta_nu_lower:
    delta_nu_error = delta_nu_upper
else: delta_nu_error = delta_nu_lower

if nu_max_paper_upper > nu_max_paper_lower:
    nu_max_paper_error = nu_max_paper_upper
else: nu_max_paper_error = nu_max_paper_lower


 #subtraction of signal and noise fit over pmodes
plt.rcParams['agg.path.chunksize'] = 1000  #increases number of calls we can make
pg_new= pg[(nu > nu_max_paper - 10*delta_nu) & (nu <nu_max_paper + 10*delta_nu)]
background_fit_new= background_fit[(nu > nu_max_paper - 10*delta_nu) & (nu <nu_max_paper + 10*delta_nu)]
diff = pg_new-background_fit_new


#lightkurve BoxKernel filter over 3delta_nu
ax = diff.plot() #creating graph to plot smoothed function over
s1 = diff.smooth(method='boxkernel', filter_width=3*delta_nu)
s1.plot(label='Smoothed', c='red', lw=2)

c=3.03
s2 = delta_nu*s1/(c)
s2.plot(ax = ax, label='Smoothed', c='red', lw=2, ylabel = 'Power (ppm$^2$)')

#displaying smoothing over range of +-5 delta nu to remove edge effects
ax=diff[(s1.frequency.value > nu_max_paper - 5*delta_nu) & (s1.frequency.value <nu_max_paper + 5*delta_nu)].plot()
freq_range=((s1.frequency.value > nu_max_paper - 5*delta_nu) & (s1.frequency.value <nu_max_paper + 5*delta_nu))
s2_new=s2[(s1.frequency.value > nu_max_paper - 5*delta_nu) & (s1.frequency.value <nu_max_paper + 5*delta_nu)]
s2_new.plot(ax=ax,label='KIC 806161 Smoothed', c='blue', lw=2)


#extracting A at nu_max from paper
nu=nu[(nu > nu_max_paper - 10*delta_nu) & (nu <nu_max_paper + 10*delta_nu)]
A_paper=(np.interp(nu_max_paper, nu, s2.power))

print("nu_max from paper = ",  nu_max_paper)
print("Amplitude at nu_max from paper = ", A_paper**0.5)

#extracting Amax and nu_max from our values
A_max = (np.amax(s2.power)) #need to square root
nu_max_code = nu[np.where(s2.power == A_max)]
print('A_max from our array = ', A_max**0.5)
print('nu_max from our array = ', nu_max_code)

#error bars from paper
s3  = s2[(nu >nu_max_paper - nu_max_paper_lower) & (nu < nu_max_paper + nu_max_paper_upper)]
A_lowerbound = ((np.amin(s3.power)))
A_upperbound = (np.amax(s3.power))
print('error bar A_paper: +', (A_upperbound.value**0.5-A_paper**0.5))
print('error bar A_paper: -', (A_paper**0.5 - A_lowerbound.value**0.5))


#error bars from code
s4 = (s2[(nu >(nu_max_code - 1.5*delta_nu)) & (nu < (nu_max_code +1.5*delta_nu))]).power
nu4 = (nu[(nu >(nu_max_code - 1.5*delta_nu)) & (nu < (nu_max_code +1.5*delta_nu))])
error = np.std(s4**0.5)
print('error on A_max_code: +/-', error)


#A_kp from scaling relation

Teff = 5668
Teff_error = 77
Teff_sun = 5777
L = 1.6274493
L_error = 0.008
A_bol = 2.53
A_bol_error = 0.11
S = 0.83
S_error = 0.002
T = 1.32
T_error = 0.02
nu_max_sun = 3090
delta_nu_sun = 135.1

#c_k correction
a1 = 1.349*10**(-4)
a2 = -3.12*10**(-9)
T_0 = 5934



#defining c_k and a_kp and propagating the errors
def c_k(t_eff):
    return 1 + 1.349*10**(-4)*(t_eff - 5934) - 3.12*10**(-9)*(t_eff - 5934)**2

Ck = c_k(Teff)
Ck_error = c_k(Teff+Teff_error)-Ck

print('ck = ', Ck, '+-', Ck_error)

def a_kp(l, nmax, deltan, t_eff, abol, ck, s, t):
    return (l**s) * (nmax/3090)**(-3*t) * (deltan/135.1)**(4*t) * (t_eff/5777)**((-3/2)*t -1) * abol/ck  

A_kp = a_kp(L, nu_max_paper, delta_nu, Teff, A_bol, Ck, S, T)

A_kp_error = np.sqrt( (a_kp((L + L_error), nu_max_paper, delta_nu, Teff, A_bol, Ck, S, T) - A_kp ) **2 \
+ (a_kp(L, nu_max_paper + nu_max_paper_error, delta_nu, Teff, A_bol, Ck, S, T) - A_kp)**2 \
+ (a_kp(L, nu_max_paper, delta_nu + delta_nu_error, Teff, A_bol, Ck, S, T) - A_kp)**2 \
+ (a_kp(L, nu_max_paper, delta_nu, Teff+ Teff_error, A_bol, Ck, S, T) - A_kp)**2 \
+ (a_kp(L, nu_max_paper, delta_nu, Teff, A_bol + A_bol_error, Ck, S, T) - A_kp)**2 \
+ (a_kp(L, nu_max_paper, delta_nu, Teff, A_bol, Ck + Ck_error, S, T) - A_kp)**2 \
+ (a_kp(L, nu_max_paper, delta_nu, Teff, A_bol, Ck, S + S_error, T) - A_kp)**2 \
+ (a_kp(L, nu_max_paper, delta_nu, Teff, A_bol, Ck, S, T + T_error) - A_kp)**2)

print('A_kp = ', A_kp, '+/-', A_kp_error)

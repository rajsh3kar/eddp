'''
SIMULATION PARAMTERS

'''
import numpy as np
import math
N_DBS = 4 #number of dbs
N_UE = 100
AREA = 500 # area
F = 2e+9 #carrier frequncy
a = 9.61 #environment dependable
beta = 0.28 #environment 
eLoS = 5 #dB excessive loss of LoS
eNLoS = 20 #dB excessive loss non LoS
sigma = -104 #dB Noise power
B = 5e+6 # Bandwidth MhZ
I = -95
c  = 3e+8 #speed of light
A_min = 50 #minimum altitude
A_max  = 200 #maximum altitude
Capacity = 1000000 #capacity of SOS 1M 
unit_move = 5 # distance it travel when action performed
tx = 20 #fixed transmission power
low_obs = np.array([0,0,50])
high_obs = np.array([AREA,AREA,150])
low_ue= np.array([0,0])
high_ue = np.array([AREA,AREA])
low_action = 0
high_action = 6.28319
sinr_th = 2
angle = math.pi/3
def pathLoS(distance):
    wavelength = c / F
    return 20 * math.log10((4 * math.pi * distance) / wavelength) + eLoS

def pathNLoS(distance):
    wavelength = c / F
    return 20 * math.log10((4 * math.pi * distance) / wavelength) + eNLoS


def angle_db(alti,distance):
    return np.arcsin(alti/distance)

#probability of LoS

def PLoS(alti,distance):
    angle = angle_db(alti,distance)
    return 1 / (a + (math.exp(-beta * (angle - a))))


def AVG_PATHLOSS(alti, distance):
    p_los = PLoS(alti, distance)
    return p_los * pathLoS(distance) + (1 - p_los) * pathNLoS(distance)



def SINR(alti, distance):
    return dBm2W(tx) * (10 ** -(AVG_PATHLOSS(alti, distance) / 10)) / + (dBm2W(I)+dBm2W(sigma))


def bandwidth(alti,distance):
    return Capacity /math.log(1+SINR(alti,distance))
def dBm2W(dBm):
    return 10**((dBm - 30)/10.)
#print(SINR(100,650))
#print(high_obs)



def radians_to_degrees(radians_list):
    degrees_list = []
    for radians in radians_list:
        degrees = radians * (180 / math.pi)
        degrees_list.append(degrees)
    return np.array(degrees_list, dtype=np.float32)
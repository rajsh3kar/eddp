"""import numpy as np
#simulation parameters

R_c = 500 #range of communication
alpha = 9.6
beta= 0.28
UAV_alti = 100
carrier_freq = 2e9 #hz
light_speed = 3e8 #m/s
ksi_los = 1  # Additional LOS path loss (dB)
ksi_nlos = 20  # Additional NLOS path loss (dB)
UAV_tp = 40 #dBm
Gauss_noise = -174 #dBm

def prob_los(horizonta_ue_UAV):
    theta = np.arctan(UAV_alti/horizonta_ue_UAV)
    return 1 / (1 + alpha * np.exp(-beta * (np.rad2deg(theta) - alpha)))

def distance(x,y):
    return np.sqrt((y[1]-x[1])**2+(y[0]-x[0])**2)

def Los(distance):
    return 20*np.log10(distance)+ 20* np.log10(carrier_freq) + 20*np.log10(4*np.pi/light_speed)+ksi_los
def NLos(distance):
    return 20*np.log10(distance)+ 20* np.log10(carrier_freq) + 20*np.log10(4*np.pi/light_speed)+ksi_nlos
def path_loss(x,y):
    horizonta_ue_UAV= distance(x,y)
    distance_ue_UAV = np.sqrt(horizonta_ue_UAV**2 + UAV_alti**2)
    prob = prob_los(horizonta_ue_UAV) 
    los = Los( distance_ue_UAV)
    nlos = NLos( distance_ue_UAV)
    return prob*los + (1-prob)*nlos
def dbm_to_watt(dbm):
   
    return 10 ** (dbm / 10.0) / 1000.0
def SINR(UAV_obj,UAV,UE):

    pl = path_loss(UAV.get_position(),UE.get_position())
    ptx = dbm_to_watt(UAV_tp)
    noise_power = dbm_to_watt(Gauss_noise)
    t1 = pl * ptx
    a = 0
    for k in UAV_obj: 
        if k != UAV: 
            a += path_loss(k.get_position(),UE.get_position())*ptx
    sinr = t1/(a+noise_power)
    sinr= 10 * np.log10(sinr)
    UE.set_sinr(UAV,sinr)
    return sinr
    


#print(path_loss([23,400],[300,550]))"""
import numpy as np

# Simulation parameters
R_c = 500  # Range of communication
alpha = 9.6
beta = 0.28
UAV_alti = 100
carrier_freq = 2e9  # Hz
light_speed = 3e8  # m/s
ksi_los = 1  # Additional LOS path loss (dB)
ksi_nlos = 20  # Additional NLOS path loss (dB)
UAV_tp = 40  # dBm
Gauss_noise = -174  # dBm

def prob_los(horizonta_ue_UAV):
    theta = np.arctan(UAV_alti / horizonta_ue_UAV)
    return 1 / (1 + alpha * np.exp(-beta * (np.rad2deg(theta) - alpha)))

def distance(x, y):
    return np.sqrt((y[1] - x[1]) ** 2 + (y[0] - x[0]) ** 2)

def Los(distance):
    return 20 * np.log10(distance) + 20 * np.log10(carrier_freq) + 20 * np.log10(4 * np.pi / light_speed) + ksi_los

def NLos(distance):
    return 20 * np.log10(distance) + 20 * np.log10(carrier_freq) + 20 * np.log10(4 * np.pi / light_speed) + ksi_nlos

def path_loss(x, y):
    horizonta_ue_UAV = distance(x, y)
    distance_ue_UAV = np.sqrt(horizonta_ue_UAV ** 2 + UAV_alti ** 2)
    prob = prob_los(horizonta_ue_UAV)
    los = Los(distance_ue_UAV)
    nlos = NLos(distance_ue_UAV)
    path_loss_db = prob * los + (1 - prob) * nlos
    return dbm_to_watt(-path_loss_db)  # Convert path loss to linear scale (Watt)

def dbm_to_watt(dbm):
    """Convert power from dBm to Watts."""
    return 10 ** (dbm / 10.0) / 1000.0

def SINR(UAV_obj, UAV, UE):
    pl = path_loss(UAV.get_position(), UE.get_position())
    ptx = dbm_to_watt(UAV_tp)
    noise_power = dbm_to_watt(Gauss_noise)
    signal_power = pl * ptx
    interference_power = 0

    for k in UAV_obj:
        if k != UAV:
            interference_power += path_loss(k.get_position(), UE.get_position()) * ptx

    sinr_linear = signal_power / (interference_power + noise_power)
    sinr_db = 10 * np.log10(sinr_linear)
    UE.set_sinr(UAV, sinr_db)

    return sinr_db

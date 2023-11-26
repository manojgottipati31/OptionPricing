import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt

########################################################
########################################################
# Least-Squares Method (V)

# Parameters
S0 = 100          # initial stock price
K = 100         # strike price
T = 1           # time to maturity
r = 0.05        # risk-free rate
sigma = 0.2     # volatility
M = 5       # number of time steps
dt = T / M      # time step
I = 1000       # number of paths
np.random.seed(1234)

# Simulating I paths with M time steps
def sim_Ipaths_MtimeSteps(M,I,sigma,dt,r):
    S = np.zeros((M + 1, I))
    S[0] = S0
    for t in range(1, M + 1):
        z = np.random.standard_normal(I)
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    return S


def calc_OptionVal(S,K,h,dt,r):
    V = np.copy(h)  # intrinsic values for put option
    for t in range(M - 1, 0, -1):
        in_the_money = K > S[t]  # Condition for in-the-money paths
        S_in_the_money = S[t][in_the_money]
        V_in_the_money = V[t + 1][in_the_money] * np.exp(-r * dt)

        if len(S_in_the_money) > 0:
            reg = np.polyfit(S_in_the_money, V_in_the_money, 6)
            C = np.polyval(reg, S[t]) 
            exercise = h[t] > C
            V[t] = np.where(exercise, h[t], V[t + 1] * np.exp(-r * dt))
            V[t + 1][exercise] = 0
    return V


#  Simulate stock paths
S = sim_Ipaths_MtimeSteps(M,I,sigma,dt,r)

# Initializing the early-exercise payoff
h = np.maximum(K - S, 0)

# Get option value at each time step in each path
V = calc_OptionVal(S,K,h,dt,r)

# present value mean
V0 = np.mean(V[1] * np.exp(-r * dt))

print("option price:",V0)


########################################################
S0 = 100          # initial stock price
K = 100         # strike price
T = 1           # time to maturity
r = 0.05        # risk-free rate
sigma = 0.2     # volatility
M = 5       # number of time steps
dt = T / M      # time step
I = 1000       # number of paths
np.random.seed(1234)
rates = np.linspace(0.02, 0.15, 500)
option_prices =[]
for i in range(len(rates)):
    S = sim_Ipaths_MtimeSteps(M,I,sigma,dt,rates[i])
    h = np.maximum(K - S, 0)
    V = calc_OptionVal(S,K,h,dt,r)
    V0 = np.mean(V[1] * np.exp(-r * dt))
    option_prices.append(V0)

plt.figure(figsize=(8, 4))
plt.plot(rates, option_prices, 'o', label='data points')
plt.xlabel('rates')
plt.ylabel('put price')
plt.show()


S0 = 100          # initial stock price
K = 100         # strike price
T = 1           # time to maturity
r = 0.05        # risk-free rate
sigma = 0.2     # volatility
M = 5       # number of time steps
dt = T / M      # time step
I = 1000       # number of paths
np.random.seed(1234)
sigmas = np.linspace(0.10, 0.40, 500)
option_prices =[]
for i in range(len(sigmas)):
    S = sim_Ipaths_MtimeSteps(M,I,sigmas[i],dt,r)
    h = np.maximum(K - S, 0)
    V = calc_OptionVal(S,K,h,dt,r)
    V0 = np.mean(V[1] * np.exp(-r * dt))
    option_prices.append(V0)

plt.figure(figsize=(8, 4))
plt.plot(sigmas, option_prices, 'o', label='data points')
plt.xlabel('volatility')
plt.ylabel('put price')
plt.show()
    


S0 = 100          # initial stock price
K = 100         # strike price
T = 1           # time to maturity
r = 0.05        # risk-free rate
sigma = 0.2     # volatility
M = 5       # number of time steps
dt = T / M      # time step
I = 1000       # number of paths
np.random.seed(1234)
Ks = np.linspace(80, 120, 500)
option_prices =[]
for i in range(len(Ks)):
    S = sim_Ipaths_MtimeSteps(M,I,sigma,dt,r)
    h = np.maximum(Ks[i] - S, 0)
    V = calc_OptionVal(S,Ks[i],h,dt,r)
    V0 = np.mean(V[1] * np.exp(-r * dt))
    option_prices.append(V0)

plt.figure(figsize=(8, 4))
plt.plot(Ks, option_prices, 'o', label='data points')
plt.xlabel('strike price, K')
plt.ylabel('put price')
plt.show()


########################################################
########################################################
# Broadie and Glasserman (IV)

# Parameters
S0 = 100       # Initial stock price
K = 100        # Strike price
T = 1          # Time to maturity in years
r = 0.05       # Risk-free rate
sigma = 0.2    # Volatility
N = 5          # Number of time steps
# Time increment
dt = T/N
np.random.seed(1234)


def sim_3branch_nrecomb_tree(N,sigma,dt,r):
    S = [[]]
    S[0] = [S0]
    for n in range(1,N+1):
        S_n = []
        for i in range(len(S[n-1])):
            for branch in range(3):
                z = np.random.standard_normal()
                S_n.append(S[n-1][i] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z))
        S.append(S_n)
    return S


# upper bound
def get_Upperbound(S,N,K,dt,r):
    EX = [[None] * len(sublist) for sublist in S]
    HOL = [[None] * len(sublist) for sublist in S]
    HOL[-1] =  [np.maximum(i -  K,0) for i in S[-1]]
    OPT = [sublist[:] for sublist in HOL]

    for n in range(N-1,-1,-1):
        for i in range(len(S[n])):
            ex_val = np.maximum(S[n][i]-K,0)
            hol_val = np.mean(OPT[n+1][i*3:i*3+3])*np.exp(-r*dt)
            EX[n][i] = ex_val
            HOL[n][i] = hol_val
            OPT[n][i] = np.maximum(ex_val,hol_val)
    return OPT,EX,HOL


# lower bound
def get_LowerBound(S,N,K,HOL,EX,dt,r):
    HOL[-1] =  [np.maximum(i -  K,0) for i in S[-1]]
    OPT_l = [sublist[:] for sublist in HOL]
    for n in range(N-1,-1,-1):
        for i in range(len(S[n])):
            ex_val = EX[n][i]
            j0_exp_hol_val = 0.5*(OPT_l[n+1][i*3 + 1] +  OPT_l[n+1][i*3 + 2])*np.exp(-r*dt)
            j1_exp_hol_val = 0.5*(OPT_l[n+1][i*3 + 0] +  OPT_l[n+1][i*3 + 2])*np.exp(-r*dt)
            j2_exp_hol_val = 0.5*(OPT_l[n+1][i*3 + 1] +  OPT_l[n+1][i*3 + 0])*np.exp(-r*dt)

            if  j0_exp_hol_val >= ex_val:
                j0_option_val = OPT_l[n+1][i*3 + 0] * np.exp(-r*dt)
            else:
                j0_option_val = ex_val

            if  j1_exp_hol_val >= ex_val:
                j1_option_val = OPT_l[n+1][i*3 + 1] * np.exp(-r*dt)
            else:
                j1_option_val = ex_val

            if  j2_exp_hol_val >= ex_val:
                j2_option_val = OPT_l[n+1][i*3 + 2] * np.exp(-r*dt)
            else:
                j2_option_val = ex_val

            OPT_l[n][i] = (j0_option_val+j1_option_val+j2_option_val)/3
    return OPT_l,HOL


# Simulate stock paths
S = sim_3branch_nrecomb_tree(N,sigma,dt,r)

# Get upper bound
OPT,EX,HOL = get_Upperbound(S,N,K,dt,r)
print("Upper bound for the option:", OPT[0][0])

# Get lower bound
OPT_l,HOL = get_LowerBound(S,N,K,HOL,EX,dt,r)
print("Lower bound for the option:", OPT_l[0][0])


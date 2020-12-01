import numpy as np
from scipy.integrate import odeint

def logistic_R0(t, R_0_start=2, k=0.2, x0=40, R_0_end=0.3):
    """
    R0 moduled as logistic function
    """
    
    return (R_0_start-R_0_end) / (1 + np.exp(-k*(-t+x0))) + R_0_end

def beta(t, gamma=1/7):
    """
    Computes beta at a given time `t`
    """

    return logistic_R0(t) * gamma

def sird_calc(y, t, N, gamma, alpha, rho, beta):
    """
    Computes SIRD model
    """

    S, I, R, D = y
    dSdt = -beta(t) * S * I / N
    dIdt = -dSdt - (1 - alpha) * gamma * I - alpha * rho * I
    dRdt = (1 - alpha) * gamma * I
    dDdt = alpha * rho * I
    return dSdt, dIdt, dRdt, dDdt

def sird(province,
         pop_prov_df,
         gamma=1/7,
         alpha=0.01,
         rho=1/9,
         days=101):
    """
    Create and compute a SIRD model

    Parameters
    ----------

    province : str
        The province name.
    
    pop_prov_df : pandas DataFrame
        The DataFrame with demographic data.
    
    gamma : float (default=1/7)
        Inverse of how many days the infection lasts.
    
    alpha : float (default=0.01)
        Death rate.
    
    rho : float (default=1/9)
        1 over number of days from infection until death.
    
    days : int (default=101)
        Total number of days to predict + 1.
    
    Return
    ------
    A numpy array of shape (4, days).
    """

    # Population
    N = pop_prov_df.loc[
        (pop_prov_df.Territorio == province) & 
        (pop_prov_df.Eta == "Total")
        ]['Value'].values[0]

    times = range(0, days)

    # S0, I0, R0, D0: initial conditions vector
    init = N-1, 1, 0, 0

    # Solve the model
    sirsol = odeint(sird_calc, init, times, args=(N, gamma, alpha, rho, beta))

    return sirsol.T
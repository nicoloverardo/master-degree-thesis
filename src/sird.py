import numpy as np
from scipy.integrate import odeint


def logistic_R0(t, R_0_start, k, x0, R_0_end):
    """
    R0 moduled as logistic function
    """

    return (R_0_start-R_0_end) / (1 + np.exp(-k*(-t+x0))) + R_0_end


def beta(t, R_0_start, k, x0, R_0_end, gamma):
    """
    Computes beta at a given time `t`
    """

    return logistic_R0(t, R_0_start, k, x0, R_0_end) * gamma


def sird_calc(y, t, N, gamma, alpha, R_0_start, k, x0, R_0_end, beta):
    """
    Computes SIRD model
    """

    S, I, R, D = y
    dSdt = -beta(t, R_0_start, k, x0, R_0_end, gamma) * S * I / N
    dIdt = -dSdt - (1 - alpha) * gamma * I - alpha * I
    dRdt = (1 - alpha) * gamma * I
    dDdt = alpha * I
    return dSdt, dIdt, dRdt, dDdt


def sird(province,
         pop_prov_df,
         gamma=1/7,
         alpha=0.01,
         days=101,
         R_0_start=2,
         k=0.2,
         x0=40,
         R_0_end=0.3):
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

    days : int (default=101)
        Total number of days to predict + 1.

    R_0_start : float (default=2)
        Starting value of RO

    k : float (default=0.2)
        How quickly R0 declines. Lower values of k will
        let R0 need more time to become lower.

    x0 : int (default=40)
        Value on the x-axis of the inflection point of R0.
        This can be interpreted as the day in which lockdown
        comes into effect.

    R_0_end : float (default=0.3)
        Final value of RO

    Returns
    -------
    A numpy array of shape (4, days).
    """

    # Population
    N = pop_prov_df.loc[
        (pop_prov_df.Territorio == province) &
        (pop_prov_df.Eta == "Total")
        ]['Value'].values[0]

    times = range(days)

    # S0, I0, R0, D0: initial conditions vector
    init = N-1, 1, 0, 0

    # Solve the model
    sirsol = odeint(sird_calc, init, times, args=(N, gamma,
                                                  alpha,
                                                  R_0_start, k,
                                                  x0, R_0_end,
                                                  beta
                                                  ))

    return sirsol.T


def Model(days, N, R_0_start, k, x0, R_0_end, alpha, gamma):
    y0 = N-1.0, 1.0, 0.0, 0.0,
    times = range(0, days)

    sirsol = odeint(sird_calc, y0, times, args=(
        N, gamma, alpha, R_0_start, k, x0, R_0_end, beta))

    S, I, R, D = sirsol.T
    R0_over_time = [beta(i, R_0_start, k, x0, R_0_end, gamma)/gamma
                    for i in range(len(times))]

    return times, S, I, R, D, R0_over_time

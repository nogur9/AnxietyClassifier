import numpy as np
from scipy.optimize import leastsq
import pylab as plt
from sklearn.linear_model import LinearRegression

def sine(data, plot=0):
    est_std_list = []
    est_phase_list = []
    est_mean_list = []
    for subject_data in data:
        N = len(subject_data)# number of data points
        t = np.linspace(0, 4*np.pi, N)
        guess_mean = np.nanmean(subject_data)
        guess_std = 3*np.nanstd(subject_data)/(2**0.5)
        guess_phase = 0

        # we'll use this to plot our first estimate. This might already be good enough for you
        data_first_guess = guess_std*np.sin(t+guess_phase) + guess_mean

        # Define the function to optimize, in this case, we want to minimize the difference
        # between the actual data and our "guessed" parameters
        optimize_func = lambda x: x[0]*np.sin(t+x[1]) + x[2] - subject_data
        est_std, est_phase, est_mean = leastsq(optimize_func, [guess_std, guess_phase, guess_mean])[0]

        # recreate the fitted curve using the optimized parameters
        est_std_list.append(est_std)
        est_phase_list.append(est_phase)
        est_mean_list.append(est_mean)

        if plot:
            data_fit = est_std * np.sin(t + est_phase) + est_mean
            plt.plot(subject_data, '.')
            plt.plot(data_fit, label='after fitting')
            #plt.plot(data_first_guess, label='first guess')
            plt.legend()
            plt.show()
    return [est_std_list, est_phase_list, est_mean_list]



def linear(data, plot=0):
    coeff1 = []
    coeff2 = []
    for subject_data in data:
        N = len(subject_data)# number of data points
        x = range(N)
        y = np.array(subject_data)
        z = np.polyfit(x, y, 1)
        coeff1.append(z[0])
        coeff2.append(z[1])
        if plot:
            p1 = np.poly1d(np.polyfit(x, y, 1))
            plt.scatter(range(N), subject_data, color='black')
            xp = np.linspace(-2, 6, 100)
            _ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p1(xp), '--')

            plt.show()
    return [coeff1,coeff2]
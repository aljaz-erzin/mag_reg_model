
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit
import scipy.stats as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from db_read_write import DIS_get_column, PREDPLOT_get_column, RPEDPLOT_get_trms_id
from matplotlib import pyplot
from numpy.random import normal
from numpy import hstack
from numpy import asarray
from numpy import exp
from sklearn.neighbors import KernelDensity
from common import chebyshev_inequality
def plot_distribution(table, column,left_init_cut, right_init_cut, left_cut, right_cut,  factor, second_factor, datefrom):


    perc = '65%'
    if factor == 2:
        perc = '95%'
    if factor > 2:
        perc = '99%'
    print("CURRENT ORIGINAL CUT = "+ perc)
    orig_perc = perc

    df = DIS_get_column(table, column, datefrom, None)
    data = df.values[:, 0].astype(float)
    print(data.shape)
    sum_thrown = len(data[(data <= left_init_cut)]) + len(data[(data >= right_init_cut)])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data, density=True, bins='auto', color='grey', histtype='stepfilled', alpha=0.2)
    tit = 'Density plot  |  (init_throw_left: '+str(len(data[(data <= left_init_cut)]))+', init_throw_right: '+str(len(data[(data >= right_init_cut)]))+')  |  (Original: ' + perc + ', Modified: '
    original_data = data.copy()
    data = data[(data > left_init_cut) & (data < right_init_cut)]

    original_data = np.sort(original_data, axis = 0)


    ax.hist(data, density=True, bins='auto', histtype='stepfilled', alpha=0.6)



    # ORIGINAL
    mean = np.mean(data)
    std_dev = np.std(data)
    x = np.linspace(np.min(data), np.max(data), 100)
    y = norm.pdf(x, mean, std_dev)
    ax.plot(x, y,
            'r-', lw=3, alpha=1, label=f'Original (Mean: {mean:.3f}, st_dev: {std_dev:.3f})')
    #ax.axvline(mean, color='red', linestyle='-')
    labela = ''
    n_cuts_left, n_cuts_right = 0, 0
    modified_data = data
    if left_cut:
        n_cuts_left = len(data[(data <= mean - factor*std_dev)])
        sum_thrown += n_cuts_left
        modified_data = modified_data[(modified_data > mean - factor*std_dev)]

    if right_cut:
        n_cuts_right = len(data[(data >= mean + factor*std_dev)])
        sum_thrown += n_cuts_right
        modified_data = modified_data[(modified_data < mean + factor*std_dev)]
        ax.axvline(mean + factor*std_dev, color='red', linestyle='--')
    if left_cut:
        ax.axvline(mean - factor * std_dev, color='red', linestyle='--',
               label=f'((L: {mean - factor*std_dev:.3f}, N:{n_cuts_left: .0f}), (R: {mean + factor*std_dev:.3f}, N: {n_cuts_right: .0f}))')



    perc = '65%'
    if second_factor == 2:
        perc = '95%'
    if second_factor > 2:
        perc = '99%'
    tit = tit + perc + ')'
    # standard deviaton
    mean = np.mean(modified_data)
    std_dev = np.std(modified_data)
    x = np.linspace(np.min(modified_data), np.max(modified_data), 100)

    y = norm.pdf(x, mean, std_dev)
    #data = data[(data > mean - std_dev)] # & (data < upper_bound)]
    n_cuts_left = len(data[(data <= mean - second_factor*std_dev)])
    n_cuts_right = len(data[(data >= mean + second_factor*std_dev)])
    sum_thrown += n_cuts_left + n_cuts_right
    ax.plot(x, y,
            'g-', lw=3, alpha=1, label=f'Modified (Mean: {mean:.3f}, st_dev: {std_dev:.3f})')
    #ax.axvline(mean, color='green', linestyle='-')
    ax.axvline(mean - second_factor*std_dev, color='green', linestyle='--', label=f'((L: {mean - second_factor*std_dev:.3f}, N:{n_cuts_left: .0f}), (R: {mean + second_factor*std_dev:.3f}, N: {n_cuts_right: .0f}))')
    ax.axvline(mean + second_factor*std_dev, color='green', linestyle='--')

    data_weights = norm.pdf(original_data, mean, std_dev)
    #print(list(zip(original_data, data_weights)))
    ax.set_xlim([-1, 1])
    ax.legend(loc='upper left')
    plt.title(tit + '  |  all_thrown: '+ str(sum_thrown))
    plt.xlabel('Data values')
    plt.ylabel('Density')
    plt.savefig('INSERT_PATH/Python/Plots/Density_' + table.split('.')[1] + '_' + column + '_' +datefrom+'.jpg')
    # Return results
    return

def plot_predictions_neto_helper(table, column, fig, ax, flag, color, alpha, type):
    df = PREDPLOT_get_column(table, column, type)
    data = df.values[:, 0].astype(float)
    ax.hist(data, density=True, bins='auto', color=color, histtype='stepfilled', alpha=alpha)
    # ORIGINAL
    if flag:
        mean = np.mean(data)
        std_dev = np.std(data)
        x = np.linspace(np.min(data), np.max(data), 100)
        y = norm.pdf(x, mean, std_dev)
        ax.plot(x, y,
                'r-', lw=3, alpha=1, label=f'(Mean: {mean:.3f}, st_dev: {std_dev:.3f})')
    return fig, ax, data
def Gauss(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y
def plot_predictions_neto(table, column, column_ratio_orignal, type, TRMS_ID):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig, ax, real  = plot_predictions_neto_helper(table, column_ratio_orignal, fig, ax, False, 'orange', 0.4, type)
    fig, ax, pred = plot_predictions_neto_helper(table, column, fig, ax, True, 'blue', 0.4, type)



    ax.set_xlim([-1, 1])
    ax.legend(loc='upper left')
    plt.title('Density plot')
    plt.xlabel('Data values')
    plt.ylabel('Density ' + type)
    plt.savefig('INSERT_PATH/Python/Plots/Density_' + table.split('.')[1] + '_TRMS_ID_' + TRMS_ID + '_' +type+ '.jpg')

    plt.clf()
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.scatter(real, pred, label="stars", color="green",
                marker="*", s=30)
    plt.title('BRUTO/NETO vs. PRED/NETO '+ type)
    plt.xlabel('BRUTO/NETO')
    plt.ylabel('PRED/NETO')
    plt.savefig('INSERT_PATH' + table.split('.')[1] + '_TRMS_ID_' + TRMS_ID + '_' +type+ '.jpg')



    # generate 2 2d grids for the x & y bounds
    y, x =  pred, real
    plt.clf()
    fig, ax = plt.subplots(figsize=(35, 35))
    z = np.sin(x * y)

    # Create heatmap
    fig, ax = plt.subplots()
    heatmap = ax.hist2d(x, y, bins=100, cmap='jet' )
    plt.colorbar(heatmap[3], ax=ax)


    plt.title('BRUTO/NETO vs. PRED/NETO ' + type)
    plt.xlabel('BRUTO/NETO')
    plt.ylabel('PRED/NETO')
    plt.savefig('INSERT_PATH' + table.split('.')[1] + '_TRMS_ID_' +TRMS_ID + '_' +type+ '.jpg')


    return



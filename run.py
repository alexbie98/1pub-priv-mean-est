import os
import math
import argparse
import random
import time

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import torch

from algos import multivariate_mean_iterative, L2

def compute_r(d, beta):
    '''
    Compute R based on Gaussian tailbound for 1 public sample
    '''
    return (d + 2*(d*math.log(1/beta))**0.5 + 2*math.log(1/beta))**0.5

def one_pub_coinpress(x_pub, X, c, t, Ps):
    '''
    Run CoinPress with one public sample x_pub
    '''
    Y = X - x_pub
    y_mean = multivariate_mean_iterative(Y, c, compute_r(X.shape[0], 0.01), t, Ps)
    return y_mean + x_pub

def run_simulations(k):
    # set experiment parameters
    d = 50
    mean = [k] * d
    cov = np.eye(d)
    p = 0.5 # target rho

    # coinpress settings
    c = [0] * d
    r = k * np.sqrt(d)
    t = 2
    Ps2 = [(1.0/4.0)*p, (3.0/4.0)*p]

    print(f'| Running simulations for k = {k} => R = {r}')
    non_pr_err = []
    t2_cp_err = []
    one_pub_cp_err = []

    n = np.linspace(1000, 10000, num=12)

    # for every sample size n_i, draw n_i samples and run each estimator 100 times
    for n_i in n:
        non_pr_err_i = []
        t2_cp_err_i = []
        one_pub_cp_err_i = []
        for i in range(100):
            # samples
            X = np.random.multivariate_normal(mean, cov, int(n_i))

            # non-private empirical mean
            non_pr_err_i.append(
                L2(np.mean(X.copy(), axis=0) - mean)
            )

            # coin press
            t2_cp_err_i.append(
                L2(multivariate_mean_iterative(X = X.copy(), c=c,r=r, t=t, Ps=Ps2) - mean)
            )

            # 1 pub + coin press 
            x_pub = np.random.multivariate_normal(mean, cov, 1)
            one_pub_cp_err_i.append(
                L2(one_pub_coinpress(x_pub=x_pub.copy(), X=X.copy(), c=c, t=t, Ps=Ps2) - mean)
            )

        non_pr_err.append(non_pr_err_i)
        t2_cp_err.append(t2_cp_err_i)
        one_pub_cp_err.append(one_pub_cp_err_i)

    return n, non_pr_err, t2_cp_err, one_pub_cp_err

def calc_trimmed_mean_std(err):
    return (
        scipy.stats.trim_mean(err, 0.1),
        scipy.stats.mstats.trimmed_std(err, limits=(0.1, 0.1), ddof=1)
    )

def draw_plot(n, err_stats, ylim, name, path, is_pdf):

    non_pr_err_stats, t2_cp_err_stats, one_pub_cp_err_stats = [
        list(zip(*err_stat)) for err_stat in err_stats
    ]

    fig, ax = plt.subplots(figsize=(5,4))
    ax.errorbar(
        n, non_pr_err_stats[0], yerr=non_pr_err_stats[1],
        marker=".", label='Non-private', color='#1f77b4'
    )
    ax.errorbar(
        n, t2_cp_err_stats[0], yerr=t2_cp_err_stats[1], 
        marker=".",  label='CoinPress (t=2)', color='#e377c2'
    )
    ax.errorbar(
        n, one_pub_cp_err_stats[0], yerr=one_pub_cp_err_stats[1], 
        marker=".", label='1 public + CoinPress (t=2)', color='#9467bd'
    )
    ax.set_xlabel('n')
    ax.set_ylabel('Mean Est L2 Error')
    ax.legend()
    ax.set_title('Multivariate Mean Estimation')
    if ylim is not None:
        ax.set_ylim(0, ylim)
    fig.tight_layout()
    fig.savefig(os.path.join(path, name + ('.pdf' if is_pdf else '.png')))

def set_fontsize():
    plt.rc('font', size=14)
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('legend', fontsize=10)
    plt.rc('figure', titlesize=14)

def fix_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', action='store_true', 
                        help='Generate output plots in .pdf format. Default is .png')
    parser.add_argument('--path', type=str, default='plots', 
                        help='Output path of plots. Default is ./plots/')
    args = parser.parse_args()

    print(f'| Outputting plots to {os.path.join(args.path, "*.pdf" if args.pdf else "*.png")}')

    set_fontsize()

    start = time.perf_counter()

    for k in [10, 100, 1000]:
        fix_seed()
        n, non_pr_err, t2_cp_err, one_pub_cp_err = run_simulations(k=k)

        err_stats = [
            [calc_trimmed_mean_std(err_i) for err_i in err]
            for err in (non_pr_err, t2_cp_err, one_pub_cp_err)
        ]

        draw_plot(n, err_stats, ylim=None, name=f'k{k}', path = args.path, is_pdf=args.pdf)
        if k!=10:
            draw_plot(n, err_stats, ylim=0.4, name=f'k{k}-zoomed', path = args.path, is_pdf=args.pdf)

    print (f'| Time elapsed: {time.perf_counter() - start}')

if __name__ == '__main__':
    main()

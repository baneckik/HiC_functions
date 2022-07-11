import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_transitivity(hic, link_function):
    n = hic.shape[0]
    a = []
    b = []
    c = []
    ab_bc = []
    ac = []
    dist = []
    is_ok = []
    for i in tqdm(range(0, n-2)):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                a.append(i)
                b.append(j)
                c.append(k)
                ab_bc_read = link_function(hic[i][j], hic[j][k])
                ab_bc.append(ab_bc_read)
                ac.append(hic[i][k])
                dist.append(k-i)
                is_ok.append(hic[i][k] >= ab_bc_read-1.01)
    df = pd.DataFrame()
    df["a"] = a
    df["b"] = b
    df["c"] = c
    df["ab_bc"] = ab_bc
    df["ac"] = ac
    df["dist"] = dist
    df["is_ok"] = is_ok
    return df


def get_trans_and_error_hic(hic, link_function):
    print("Calculating transitivity data frame...")
    df = get_transitivity(hic, link_function)
    print("Calculating error matrix...")
    error_hic = np.zeros((hic.shape[0], hic.shape[1]))
    for i in tqdm(range(df.shape[0])):
        if df["is_ok"].iloc[i] == 0:
            error_hic[df["a"].iloc[i]][df["b"].iloc[i]] += 1
            error_hic[df["b"].iloc[i]][df["a"].iloc[i]] += 1
            
            error_hic[df["b"].iloc[i]][df["c"].iloc[i]] += 1
            error_hic[df["c"].iloc[i]][df["b"].iloc[i]] += 1
            
            error_hic[df["a"].iloc[i]][df["c"].iloc[i]] -= 1
            error_hic[df["c"].iloc[i]][df["a"].iloc[i]] -= 1
    return df, error_hic
    

def print_comparison(hic_orig, hic_rec, name="test.png"):
    fig, ax = plt.subplots(3,2, figsize=(16,18))
    ax[0][0].set_title("Real Hi-C map", fontsize=15)
    ax[0][1].set_title("Hi-C from reconstructed structure", fontsize=15)
    sns.heatmap(hic_orig, cmap="summer", ax=ax[0][0])
    sns.heatmap(hic_rec, cmap="summer", ax=ax[0][1])
    
    def link_function(a, b):
        return a+b
    df1 = get_transitivity(hic_orig, link_function)
    df2 = get_transitivity(hic_rec, link_function)
    ax[1][0].set_xlabel("A~B + B~C", fontsize=15)
    ax[1][1].set_xlabel("A~B + B~C", fontsize=15)
    ax[1][0].set_ylabel("A~C", fontsize=15)
    ax[1][1].set_ylabel("A~C", fontsize=15)
#     sns.scatterplot(data=df1, x="a", y="b", hue="dist", ax=ax[1][0])
#     sns.scatterplot(data=df2, x="a", y="b", hue="dist", ax=ax[1][1])
    sns.kdeplot(data=df1, x="ab_bc", y="ac", fill=True, ax=ax[1][0])
    sns.kdeplot(data=df2, x="ab_bc", y="ac", fill=True, ax=ax[1][1])
    
    def link_function(a, b):
        return a-b
    df1 = get_transitivity(hic_orig, link_function)
    df2 = get_transitivity(hic_rec, link_function)
    ax[2][0].set_xlabel("A~B - B~C", fontsize=15)
    ax[2][1].set_xlabel("A~B - B~C", fontsize=15)
    ax[2][0].set_ylabel("A~C", fontsize=15)
    ax[2][1].set_ylabel("A~C", fontsize=15)
#     sns.scatterplot(data=df1, x="a", y="b", hue="dist", ax=ax[2][0])
#     sns.scatterplot(data=df2, x="a", y="b", hue="dist", ax=ax[2][1])
    sns.kdeplot(data=df1, x="ab_bc", y="ac", fill=True, ax=ax[2][0])
    sns.kdeplot(data=df2, x="ab_bc", y="ac", fill=True, ax=ax[2][1])
    
    #plt.savefig(name)
    plt.show()
    
    
def print_trans(hic, name="test.png"):
    fig, ax = plt.subplots(1, 3, figsize=(16,5))
    ax[0].set_title("Real Hi-C map", fontsize=15)
    sns.heatmap(hic, cmap="summer", ax=ax[0])
    
    def link_function(a, b):
        return a+b
    df = get_transitivity(hic, link_function)
    ax[1].set_xlabel("A~B + B~C", fontsize=15)
    ax[1].set_ylabel("A~C", fontsize=15)
    sns.scatterplot(data=df, x="ab_bc", y="ac", hue="is_ok", ax=ax[1])
#     sns.kdeplot(data=df1, x="a", y="b", fill=True, ax=ax[1])
    ax[1].grid()
    
    error_hic = np.zeros((hic.shape[0], hic.shape[1]))
    for i in range(df.shape[0]):
        if df["is_ok"].iloc[i] == 0:
            error_hic[df["a"].iloc[i]][df["b"].iloc[i]] += .5
            error_hic[df["b"].iloc[i]][df["a"].iloc[i]] += .5
            
            error_hic[df["b"].iloc[i]][df["c"].iloc[i]] += .5
            error_hic[df["c"].iloc[i]][df["b"].iloc[i]] += .5
            
            error_hic[df["a"].iloc[i]][df["c"].iloc[i]] -= 1
            error_hic[df["c"].iloc[i]][df["a"].iloc[i]] -= 1
            
    
    ax[2].set_title("Hi-C map of tr.in. violations", fontsize=15)
    sns.heatmap(error_hic, cmap="icefire", ax=ax[2])
    #plt.savefig(name)
    plt.show()
    

def print_error(hic_orig, df, error_hic, density_plot=False):
    fig, ax = plt.subplots(1, 3, figsize=(16,5))
    ax[0].set_title("Hi-C map", fontsize=15)
    sns.heatmap(hic_orig, cmap="YlOrBr", ax=ax[0])
    
    ax[1].set_xlabel("A~B + B~C", fontsize=15)
    ax[1].set_ylabel("A~C", fontsize=15)
    if density_plot:
        sns.kdeplot(data=df, x="ab_bc", y="ac", fill=True, ax=ax[1])
    else:
        sns.scatterplot(data=df, x="ab_bc", y="ac", hue="is_ok", ax=ax[1])
    
    ax[1].grid()
    
    ax[2].set_title("Map of tr.in. violations", fontsize=15)
    sns.heatmap(error_hic, cmap="icefire", ax=ax[2])
    #plt.savefig(name)
    plt.show()
    

def print_error2(hic_orig, error_hic):
    fig, ax = plt.subplots(1, 2, figsize=(16,6))
    ax[0].set_title("Hi-C map", fontsize=15)
    sns.heatmap(hic_orig, cmap="YlOrBr", ax=ax[0])
    
    ax[1].set_title("Map of trian. ineq. violations", fontsize=15)
    sns.heatmap(error_hic, cmap="icefire", ax=ax[1])
    #plt.savefig(name)
    plt.show()
    
    
def normalize(map):
    result = map
    np.fill_diagonal(result, np.median(result))
    return (result - result.min()) / (result.max() - result.min())


def f(x):
    return np.arctan(x / np.median(x)) / np.pi * 2 


def adjust_errors(hic, error_hic):
    n = hic.shape[0]
    hic2 = hic.copy()
    error_max = np.max(error_hic)
    error_min = np.min(error_hic)
    error_med = np.median(hic)
    for i in range(n):
        for j in range(i, n):
            if error_hic[i][j]>0:
                hic2[i][j] *= (1-error_hic[i][j]/error_max/10)
            elif error_hic[i][j]<0 and hic2[i][j]!=0:
                hic2[i][j] *= (1+error_hic[i][j]/error_min/2)
            elif error_hic[i][j]<0:
                hic2[i][j] += error_hic[i][j]/error_min*error_med
            hic2[j][i] = hic2[i][j]
    return hic2
    
    
    
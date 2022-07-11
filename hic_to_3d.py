import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import imageio
from scipy.spatial.distance import pdist, squareform


def dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)


def f(hic):
    return np.arctan(hic/np.median(hic))/np.pi*2


def g(x):
    return 1-np.arctan(x)/np.pi*2


def normalize(hic):
    return (hic - hic.min())/(hic.max()-hic.min())


def get_hic(points):
    return g(squareform(pdist(points, 'euclidean')))


# ---------------- optimizing ---------------------

def opt_fun(hic, hic2, i=None):
    if i==None:
        return sum(sum(hic*hic2))**2/sum(sum(hic**2))/sum(sum(hic2**2))
    else:
        return sum(hic[i]*hic2[i])**2/sum(hic[i]**2)/sum(hic2[i]**2)


def update_hic(hic, points, i, coord, delta):
    """
    Updates the whole Hi-C map from 
    """
    n = hic.shape[0]
    
    if coord=="x":
        new_point = (points[i][0]+delta, points[i][1], points[i][2])
    elif coord=="y":
        new_point = (points[i][0], points[i][1]+delta, points[i][2])
    elif coord=="z":
        new_point = (points[i][0], points[i][1], points[i][2]+delta)
    
    for j in range(n):
        if j != i:
            d = g(dist(new_point, points[j]))
            hic[i][j] = d
            hic[j][i] = d
    return hic


def update_hic2(hic, points, i):
    for j in range(hic.shape[0]):
        if i!=j:
            d = g(dist(points[i], points[j]))
            hic[i][j] = d
            hic[j][i] = d
    return hic


def get_mse(hic1, hic2):
    n = hic1.shape[0]
    error_sum = 0
    for i in range(n):
        for j in range(n):
            error_sum += (hic1[i][j]-hic2[i][j])**2
    return error_sum/n**2


def optimize(orig_hic, orig_structure=None, init_structure=None, r=1, iterations=100, delta=0.1, lr=1, margin=0.1, seconds=5, seconds_pause=1, fps=20, gif_path='./test.gif', elev=35, azim=35, elev2=35, azim2=35, lim=None, corr_thresh=0.9999):
    
    n = orig_hic.shape[0]
    if True:
        hic = f(orig_hic)
    
    if init_structure==None:
#         theta = np.linspace(0, 6, n)
#         init_structure = [ (r*np.cos(t*2), r*np.cos(t), r*np.sin(t)) for t in theta]
        embedding = MDS(n_components=3, max_iter=10000)
        init_structure = embedding.fit_transform(np.exp(-hic))
    
    curr_structure = init_structure.copy()
    curr_hic = get_hic(curr_structure)
    
    # plot dimensions
    if orig_structure is not None:
        max_dim = np.max(orig_structure)
        min_dim = np.min(orig_structure)
        range_dim = max_dim-min_dim
        xlim = (min_dim-margin*range_dim, max_dim+margin*range_dim)
    else:
        max_dim = np.max(init_structure)
        min_dim = np.min(init_structure)
        range_dim = max_dim-min_dim
        if lim==None:
            xlim = (min_dim-margin*range_dim, max_dim+margin*range_dim)
        else:
            xlim = lim
    
    mse = []
    corr = []
    trajectory = [init_structure]
    plots = []
    plots.append(plot_for_gif(trajectory[0], orig_structure, hic, 0, elev=elev, azim=azim, elev2=elev2, azim2=azim2, xlim=xlim, ylim=xlim, zlim=xlim))
    for it in tqdm(range(iterations)):
        for i in range(0, n):
            hic2 = update_hic(curr_hic, curr_structure, i, "x", -delta)
            fun_x1 = opt_fun(hic, hic2, i)
            hic2 = update_hic(curr_hic, curr_structure, i, "x", delta)
            fun_x2 = opt_fun(hic, hic2, i)
            hic2 = update_hic(curr_hic, curr_structure, i, "y", -delta)
            fun_y1 = opt_fun(hic, hic2, i)
            hic2 = update_hic(curr_hic, curr_structure, i, "y", delta)
            fun_y2 = opt_fun(hic, hic2, i)
            hic2 = update_hic(curr_hic, curr_structure, i, "z", -delta)
            fun_z1 = opt_fun(hic, hic2, i)
            hic2 = update_hic(curr_hic, curr_structure, i, "z", delta)
            fun_z2 = opt_fun(hic, hic2, i)
            grad = [(fun_x2-fun_x1)/delta, (fun_y2-fun_y1)/delta, (fun_z2-fun_z1)/delta]
            curr_structure[i] = (curr_structure[i][0]+lr*grad[0], 
                                 curr_structure[i][1]+lr*grad[1], 
                                 curr_structure[i][2]+lr*grad[2])
            new_hic = update_hic2(curr_hic, curr_structure, i)
            curr_hic = new_hic
            
        mse.append(get_mse(hic, new_hic))
        corr.append(opt_fun(hic, curr_hic))
           
        xlim2 = list(xlim).copy()
        if it<20 or it%(max(1, iterations//seconds//fps))==0 or it==iterations-1:
            trajectory.append(curr_structure.copy())
            max_dim = np.max(curr_structure)
            min_dim = np.min(curr_structure)
            range_dim = max_dim-min_dim
            if min_dim<xlim[0]-0.1*range_dim:
                xlim2[0] = min_dim   
            if max_dim>xlim[1]+0.1*range_dim:
                xlim2[1] = max_dim
            
            plots.append(plot_for_gif(trajectory[-1], orig_structure, hic, it+1, elev=elev, azim=azim, elev2=elev2, azim2=azim2, xlim=xlim2, ylim=xlim2, zlim=xlim2))
        if corr[-1]>corr_thresh:
            break
    plots = [plots[0]]*(seconds_pause*fps) + plots + [plots[-1]]*(seconds_pause*fps)
    print("saving...")
    imageio.mimsave(gif_path, plots, fps=fps)
    print("Saved")
    print(f"Final MSE: {round(mse[-1],4)}\nFinal Correlation: {round(corr[-1],4)}")
    return trajectory, mse, corr
    
    
def plot_for_gif(structure, orig_structure, orig_hic, i, elev=35, azim=35, elev2=35, azim2=35, xlim=(-1,1), ylim=(-1,1), zlim=(-1,1)):
    
    fig, ax = plt.subplots(2,2, figsize=(12,10))

    ax[0][0].set_title("Original structure", fontsize=20)
    ax[0][1].set_title("Iteration "+str(i), fontsize=20)

    ax[1][0].axis('off')
    ax[1][1].axis('off')
    
    g = sns.heatmap(orig_hic, linewidth=0, cmap="YlGnBu", ax=ax[0][0])
    g = sns.heatmap(get_hic(structure), linewidth=0, cmap="YlGnBu", ax=ax[0][1])
    
    if orig_structure != None:
        ax3 = fig.add_subplot(2,2,3, projection='3d')
        ax3.plot([p[0] for p in orig_structure], 
                   [p[1] for p in orig_structure], 
                   [p[2] for p in orig_structure], 
                marker='o')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('z')
        ax3.set_xlim(xlim)
        ax3.set_ylim(ylim)
        ax3.set_zlim(zlim)
        ax3.view_init(elev=elev, azim=azim)
    
    ax4 = fig.add_subplot(2,2,4, projection='3d')
    ax4.plot([p[0] for p in structure], 
               [p[1] for p in structure], 
               [p[2] for p in structure], 
                marker='o')
    
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')
    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim)
    ax4.set_zlim(zlim)
    ax4.view_init(elev=elev2, azim=azim2)
    
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return image


def amplify(structure, scale=10):
    return [(s[0]*scale, s[1]*scale, s[2]*scale) for s in structure]

    

### this script calculates performance metrics and creates plots



### import packages

import numpy as np
import matplotlib.pyplot as plt
import skimage.measure

from os import listdir
from os.path import isfile, join



### get parameters

with open('settings.json', 'r') as f:
  settings = json.load(f)

output_path = settings['output_directory']
training_sets = settings['training_sets']
test_sets_dams = settings['test_sets_dams']
test_sets_no_dams = settings['test_sets_no_dams']
test_sets_all = test_sets_no_dams + test_sets_dams + ['all']
resolutions = settings['eval_resolutions']
coarsing = settings['coarsing']
quantiles = settings['quantiles']
n_per_file = settings['n_images_shard'] * 10
cross_validations = [1,2,3]



### function for counting TP, FP, FN per image that include dams, for subsets of dams
def counting_positives(pred, truth1, truth2, thresh, coarsing, training_data, test_data, cross_val, merc):
    
    # get number of prediction arrays
    n = len(pred)
    
    # create a array in which to store TP, FP, ...
    d = np.zeros((n*len(thresh)*len(coarsing), 10))
    
    # loop over prediction arrays
    for j in range(n):
        
        # get prediction
        p_i = pred[j][:, :, 0].copy()
        
        # get ground truth heatmap for subset of dams dams (t_w) and all dams (t_uw))
        t_w = truth1[j][:, :].copy()
        t_uw = truth2[j][:, :].copy()
        
        # take only location where value is one (dam locations)
        t_w[t_w < 1] = 0
        t_uw[t_uw < 1] = 0
        
        # loop over extents (called coarsing)
        for c_enum, c in enumerate(coarsing):
            
            # apply block_reduce to get the require extent
            p_i = skimage.measure.block_reduce(p_i, (c, c), np.max)
            t_w = skimage.measure.block_reduce(t_w, (c, c), np.max)
            t_uw = skimage.measure.block_reduce(t_uw, (c, c), np.max)
            
            # create copies so that the former array is kept
            p_j = p_i.copy()
            t_w_j = t_w.copy()
            t_uw_j = t_uw.copy()
            
            # loop over thresholds, starting from largest one (that way, we only have to check for newly added predicted locations)
            for t_enum, t in enumerate(thresh):
                
                # get predicted dam locations by applying threshold
                dams = np.where(p_j >= t)

                # remove the predicted locations from prediction heatmap to later lower threshold
                p_j[p_j >= t] = 0
                
                # loop over predicted locations
                for k in range(len(dams[0])):
                    
                    # if there is a dam of the height that is evaluated: increase TP by 1
                    if (t_w_j[dams[0][k], dams[1][k]] == 1):
                        d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 0] += 1
                        # dam found: remove ground truth location to later count FN
                        t_w_j[dams[0][k], dams[1][k]] = 0
                    # if there is no dam at the location: increase FP by 1 (for evaluation of all dams)
                    elif (t_uw_j[dams[0][k], dams[1][k]] != 1):
                        d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 1] += 1
                        
                # sum TP and FP with TP and FP of higher thresholds
                if t != thresh[0]:
                    d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 0:2] += d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum - 1, 0:2]
                
                # get FN
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 2] = np.sum(t_w_j)
                
                # save threshold
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 3] = t_enum
                
                # save extent
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 4] = c_enum
                
                # save training set identifier
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 5] = training_data
                
                # save test set identifier
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 6] = test_data
                
                # save cross validation fold
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 7] = cross_val
                
                # save mercator coordinates of array
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 8] = merc[j, 1]
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 9] = merc[j, 2]
                
    return d # columns of output array: TP, FP, FN, threshold, coarsing, training_data, test_data, cross_validation fold, mercator coordinates



### equivalent function of the previous function that counts TP, FP, FN per image that includes dams, but for all dams
def counting_all(pred, truth1, thresh, coarsing, training_data, test_data, cross_val, merc):
    n = len(pred)
    d = np.zeros((n*len(thresh)*len(coarsing), 10))
    for j in range(n):
        p_i = pred[j][:, :, 0].copy()
        t_w = truth1[j][:, :].copy()
        t_w[t_w < 1] = 0
        for c_enum, c in enumerate(coarsing):
            p_i = skimage.measure.block_reduce(p_i, (c, c), np.max)
            t_w = skimage.measure.block_reduce(t_w, (c, c), np.max)
            p_j = p_i.copy()
            t_w_j = t_w.copy()
            for t_enum, t in enumerate(thresh): 
                dams = np.where(p_j >= t)
                p_j[p_j >= t] = 0
                for k in range(len(dams[0])):
                    if (t_w_j[dams[0][k], dams[1][k]] == 1):
                        d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 0] += 1
                        t_w_j[dams[0][k], dams[1][k]] = 0
                    else:
                        d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 1] += 1
                if t != thresh[0]:
                    d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 0:2] += d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum - 1, 0:2]
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 2] = np.sum(t_w_j)
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 3] = t_enum
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 4] = c_enum
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 5] = training_data
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 6] = test_data
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 7] = cross_val
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 8] = merc[j, 1]
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 9] = merc[j, 2]
    return d # columns of output array: TP, FP, FN, threshold, coarsing, training_data, test_data, cross_validation fold, mercator coordinates




### function for counting TP, FP, FN per image that doesn't include dams
def counting_negatives(pred, thresh, coarsing, training_data, test_data, cross_val, merc):
    n = len(pred)
    d = np.zeros((n*len(thresh)*len(coarsing), 10))
    for j in range(n):
        p_i = pred[j][:, :, 0].copy()
        for c_enum, c in enumerate(coarsing):
            p_i = skimage.measure.block_reduce(p_i, (c, c), np.max)
            p_j = p_i.copy()
            for t_enum, t in enumerate(thresh): 
                dams = np.where(p_j >= t)
                p_j[p_j >= t] = 0
                
                # columns 0 and 2 of output array are 0 (TP and FN since they cannot happen)
                # get false positives (since there shouldn't be any dams in image)
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 1] += len(dams[0])
                if t != thresh[0]:
                    d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 0:2] += d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum - 1, 0:2]
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 3] = t_enum
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 4] = c_enum
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 5] = training_data
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 6] = test_data
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 7] = cross_val
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 8] = merc[j, 1]
                d[j *(len(thresh) * len(coarsing)) + c_enum * len(thresh) + t_enum, 9] = merc[j, 2]
    return d # columns of output array: TP, FP, FN, threshold, coarsing, training_data, test_data, cross_validation fold, mercator coordinates



### this function creates multiple heatmaps of the performance
def multiplot_heatmap(tab):
    
    # headings
    names = ["(a)", "(b)", "(c)"]
    # create fig and axs objects
    fig, axs = plt.subplots(1, 3, figsize=(20,9), sharey=True)
    
    # loop over resolutions
    for r_num, resolution in enumerate(resolutions):
        # empty array for results
        data = np.zeros((5,5))
        # loop over training sets
        for tr_num, training_set in enumerate(training_sets):
            # loop over test sets of different dam sizes
            for te_num in range(len(test_sets_no_dams), len(test_sets_all)):
                # create array to store results for different thresholds
                thr_tab = np.zeros(thresholds.shape[2])
                # loop over thresholds
                for thr_num, threshold in enumerate(thresholds[tr_num, 0]):
                    # filter results table
                    res_tab = results_summary[results_summary[:,0] == tr_num,:]
                    res_tab = res_tab[res_tab[:,1] == te_num,:]
                    res_tab = res_tab[res_tab[:,2] == thr_num,:]
                    res_tab = res_tab[res_tab[:,3] == r_num,:]
                    # calculate mean F score of cross validation folds
                    thr_tab[thr_num] = np.mean(res_tab[:,10])

                # put all dams training dataset first in plot (pure swap)
                if te_num < 6:
                    te_row = te_num + 1
                elif te_num == 6:
                    te_row = 2
                data[tr_num, te_row-len(test_sets_no_dams)] = np.max(thr_tab)
        # create heatmap from F scores
        heatmap = axs[r_num].pcolor(data, vmin = 0, vmax = 1, cmap = "Purples")
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                axs[r_num].text(x + 0.5, y + 0.5, '%.2f' % data[y, x],horizontalalignment='center', verticalalignment='center', fontsize = 20)
        axs[r_num].set_xticks([])
        axs[r_num].set_yticks([])
        axs[r_num].text(0.4, -0.3, 'All dams', ha='center', fontsize = 14)
        axs[r_num].text(1.4, -0.3, '$\geq$15m', ha='center', fontsize = 14)
        axs[r_num].text(2.4, -0.3, '10-<15m', ha='center', fontsize = 14)
        axs[r_num].text(3.4, -0.3, '5-<10m', ha='center', fontsize = 14)
        axs[r_num].text(4.4, -0.3, '<5m', ha='center', fontsize = 14)
    axs[0].text(-0.3, 0.4, 'All dams', va='center', rotation=90, fontsize = 14)
    axs[0].text(-0.3, 1.4, '$\geq$15m', va='center', rotation=90, fontsize = 14)
    axs[0].text(-0.3, 2.4, '10-<15m', va='center', rotation=90, fontsize = 14)
    axs[0].text(-0.6, 2.4, 'Evaluated on', va='center', rotation=90, fontsize = 16)
    axs[0].text(-0.3, 3.4, '5-<10m', va='center', rotation=90, fontsize = 14)
    axs[0].text(-0.3, 4.4, '<5m', va='center', rotation=90, fontsize = 14)
    axs[0].text(2.4, -0.6, 'Trained on', ha='center', fontsize = 16)
    axs[1].text(2.4, -0.6, 'Trained on', ha='center', fontsize = 16)
    axs[2].text(2.4, -0.6, 'Trained on', ha='center', fontsize = 16)
    axs[0].text(2.4, 5.3, '(a)', ha='center', fontsize = 35)
    axs[1].text(2.4, 5.3, '(b)', ha='center', fontsize = 35)
    axs[2].text(2.4, 5.3, '(c)', ha='center', fontsize = 35)
    cbar_ax = fig.add_axes([0.02, 0.05, 0.02, 0.9])
    cbar = fig.colorbar(heatmap, cax=cbar_ax, orientation="vertical")
    font_size = 20 # Adjust as appropriate.
    cbar.ax.tick_params(labelsize=font_size)
    
    # save and close
    fig.savefig('plots/heatmap.pdf')
    fig.clf()



### this function creates a heatmap with performance across the US (input argument: table of output)
def multiplot_map(tab):
    
    # heading for plot 
    names = ["(a)", "(b)", "(c)"]
    
    # create fig, axs objects
    fig, axs = plt.subplots(1, 3, figsize=(20,9), sharex=True, sharey=True)
    
    # loop over cross validation folds
    for c_num in [0,1,2]:
        
        # subset table for: cross validation fold, training set all dams, third threshold (0.999 quantile) and test set rivers, lakes OR all dams
        tab_small = tab[np.where((tab[:,4] == c_num) & (tab[:,5] == 0) & (tab[:,3] == 2) & ((tab[:,6] == 6) | (tab[:,6] == 1) | (tab[:,6] == 0))),][0]
        
        # calculate limits of plot
        x_lims = [np.min(tab_small[:,8]), np.max(tab_small[:,8])]
        y_lims = [np.min(tab_small[:,9]), np.max(tab_small[:,9])]
        
        # get longer and maller side
        larger_diff = np.max([x_lims[1] - x_lims[0], y_lims[1] - y_lims[0]])
        smaller_diff = np.min([x_lims[1] - x_lims[0], y_lims[1] - y_lims[0]])
        
        # max. 50 bins
        x_steps = 50
        y_steps = np.int32(np.ceil(smaller_diff/larger_diff*x_steps))
        
        # create array for map data
        map_arr = np.zeros([x_steps, y_steps])
        map_arr[:] = np.nan
        
        # loop over all map bins
        for x in range(x_steps):
            for y in range(y_steps):
                
                # filter by mercator coordinates to get data of bin
                tab_mappart = tab_small[np.where((tab_small[:,8] > (x_lims[0] + x * larger_diff/x_steps)) & (tab_small[:,8] < (x_lims[0] + (x + 1) * larger_diff/x_steps)) & (tab_small[:,9] > (y_lims[0] + y * larger_diff/x_steps)) & (tab_small[:,9] < (y_lims[0] + (y + 1) * larger_diff/x_steps))),:][0]
                tab_negatives = tab_mappart[np.where((tab_mappart[:,6] == 0) | (tab_mappart[:,6] == 1)),:][0]
                tab_positives = tab_mappart[np.where((tab_mappart[:,6] == 6)),:][0]
                
                # calculate weighted TP, FP, FN
                TP = np.sum(tab_positives[:,0]) * tab_negatives.shape[0]/tab_positives.shape[0]
                FP = np.sum(tab_positives[:,1]) * tab_negatives.shape[0]/tab_positives.shape[0] + np.sum(tab_negatives[:,1])
                FN = np.sum(tab_positives[:,2]) * tab_negatives.shape[0]/tab_positives.shape[0] + np.sum(tab_negatives[:,2])
                
                # calculate F score, if possible
                if (TP + FP + FN == 0):
                    map_arr[x, y_steps -y - 1] = np.nan
                elif tab_positives.shape[0] == 0:
                    map_arr[x, y_steps - y - 1] = np.nan
                elif tab_negatives.shape[0] == 0:
                    map_arr[x, y_steps - y - 1] = np.nan            
                else:
                    map_arr[x, y_steps - y - 1] = 2 * TP / (2 * TP + FP + FN)
                    
        # for visual reasons, swap axes
        map_arr = np.swapaxes(map_arr, 0, 1)
        
        # create colormap
        im = axs[c_num].imshow(map_arr, cmap = 'Purples', vmin = 0, vmax = 1)

        # title + no axis
        axs[c_num].set_title(names[c_num], y=-0.2, fontsize = 35)
        axs[c_num].axis('off')

    # finalize plot
    cbar_ax = fig.add_axes([0.05, 0.8, 0.9, 0.05])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    font_size = 20 # Adjust as appropriate.
    cbar.ax.tick_params(labelsize=font_size)
    fig.tight_layout()
    
    # save and clear
    fig.savefig('plots/map.pdf')
    fig.clf()



### function that creates a precision/recall curve for one training set and one evaluation resolution, but multiple tests sets (different lines) and multiple thresholds (different points on line)
### input arguments: data, training dataset, resolution, testing dataset, names for testing dataset, color for testing dataset, filename
def precision_recall_single(data, cut1, cut2, cut3, names_cut3, colors_cut3, filename):
    
    # subset data for training set and resolution
    dat = data[data[:,0] == cut1,:]
    dat = dat[dat[:,3] == cut2,:]
    
    # different subsets, one for each cross-validation
    c1 = dat[dat[:,4] == 0]
    c2 = dat[dat[:,4] == 1]
    c3 = dat[dat[:,4] == 2]
    
    # loop over testing datasets
    for i, cut in enumerate(cut3):
        
        # subset
        part1 = c1[c1[:,1] == cut,:]
        part1 = part1[part1[:, 2].argsort()]
        part2 = c2[c2[:,1] == cut,:]
        part2 = part2[part2[:, 2].argsort()]
        part3 = c3[c3[:,1] == cut,:]
        part3 = part3[part3[:, 2].argsort()]
        
        # calculate mean, min, max, std of precision and recall, over the three cross validation folds
        precision = np.stack([part1[:,9], part2[:,9], part3[:,9]], axis = 1)
        recall = np.stack([part1[:,8], part2[:,8], part3[:,8]], axis = 1)
        precision_mean = np.nanmean(precision, axis = 1)
        precision_min = np.nanmin(precision, axis = 1) 
        precision_max = np.nanmax(precision, axis = 1)
        precision_std = np.nanstd(precision, axis = 1)/np.sqrt(3)
        recall_mean = np.nanmean(recall, axis = 1)
        recall_min = np.nanmin(recall, axis = 1)
        recall_max = np.nanmax(recall, axis = 1)
        recall_std = np.nanstd(recall, axis = 1)/np.sqrt(3)
        
        # create precision and recall plot, using different colors for each testing dataset
        plt.plot(recall_mean, precision_mean, color = colors_cut3[i], label = names_cut3[i])
        plt.plot(recall_mean - recall_std, precision_mean - precision_std, color = colors_cut3[i], linewidth=0.5, ls = 'dashed')
        plt.plot(recall_mean + recall_std, precision_mean + precision_std, color = colors_cut3[i], linewidth=0.5, ls = 'dashed')
        
    # finalize plot
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend()
    
    # save and clear
    plt.savefig(filename)
    plt.clf()



### function that creates a single plot, used for multiplotting
### input arguments: data, training dataset, resolution, testing dataset, names for testing dataset, color for testing dataset, axs object
def plot_helper_function(data, cut1, cut2, cut3, names_cut3, colors_cut3, axs):
    
    # subset data for training set and resolution
    dat = data[data[:,0] == cut1,:]
    dat = dat[dat[:,3] == cut2,:]
    
    # different subsets, one for each cross-validation
    c1 = dat[dat[:,4] == 0]
    c2 = dat[dat[:,4] == 1]
    c3 = dat[dat[:,4] == 2]

    # loop over testing datasets
    for i, cut in enumerate(cut3):

        # subset
        part1 = c1[c1[:,1] == cut,:]
        part1 = part1[part1[:, 2].argsort()]
        part2 = c2[c2[:,1] == cut,:]
        part2 = part2[part2[:, 2].argsort()]
        part3 = c3[c3[:,1] == cut,:]
        part3 = part3[part3[:, 2].argsort()]


        # calculate mean, min, max, std of precision and recall, over the three cross validation folds
        precision = np.stack([part1[:,9], part2[:,9], part3[:,9]], axis = 1)
        recall = np.stack([part1[:,8], part2[:,8], part3[:,8]], axis = 1)
        precision_mean = np.nanmean(precision, axis = 1)
        precision_min = np.nanmin(precision, axis = 1) 
        precision_max = np.nanmax(precision, axis = 1)
        precision_std = np.nanstd(precision, axis = 1)/np.sqrt(3)
        recall_mean = np.nanmean(recall, axis = 1)
        recall_min = np.nanmin(recall, axis = 1)
        recall_max = np.nanmax(recall, axis = 1)
        recall_std = np.nanstd(recall, axis = 1)/np.sqrt(3)

        # solid line for mean, dashed lines for uncertainty bands
        if cut1 == 0 and cut2 == 0:
            axs[cut1,cut2].plot(recall_mean, precision_mean, color = colors_cut3[i], label = names_cut3[i], linewidth = 1)
        else:
            axs[cut1,cut2].plot(recall_mean, precision_mean, color = colors_cut3[i], linewidth = 1)            
        axs[cut1,cut2].plot(recall_mean - recall_std, precision_mean - precision_std, color = colors_cut3[i], linewidth=0.5, ls = 'dashed')
        axs[cut1,cut2].plot(recall_mean + recall_std, precision_mean + precision_std, color = colors_cut3[i], linewidth=0.5, ls = 'dashed')

    # set axs limits
    axs[cut1,cut2].set_xlim([0,1])
    axs[cut1,cut2].set_ylim([0,1])



### function that creates multiple precision recall curves in one figure
def precision_recall_multi(data, cut_training, cut_training_names, cut_resolution, cut_resolution_names, cut3, names_cut3, colors_cut3, filename):
    
    # create fig, axs objects
    fig, axs = plt.subplots(len(cut_training),len(cut_resolution), sharex=True, sharey=True)
    
    # loop over training sets and resolutions
    for c_train in cut_training:
        for c_res in cut_resolution:
            # create single plot, using helper function
            plotting3(data = data, cut1 = c_train, cut2 = c_res, cut3 = cut3, names_cut3 = names_cut3, colors_cut3 = colors_cut3, axs = axs)

    # finalize figure
    fig.legend(ncol = 5)
    fig.text(0.5, 0.04, 'Recall', ha='center')
    fig.text(0.04, 0.5, 'Precision', va='center', rotation=90)
    fig.text(0.92, 0.82, 'All dams', va='center', rotation=270)
    fig.text(0.92, 0.66, '$\geq$15m', va='center', rotation=270)
    fig.text(0.92, 0.5, '10-<15m', va='center', rotation=270)
    fig.text(0.92, 0.34, '5-<10m', va='center', rotation=270)
    fig.text(0.92, 0.18, '<5m', va='center', rotation=270)
    fig.text(0.95, 0.5, 'Trained on', va='center', rotation=270)
    fig.text(0.23, 0.9, '280m', ha='center')
    fig.text(0.075, 0.95, 'Evaluated on', ha='center')
    fig.text(0.52, 0.9, '1120m', ha='center')
    fig.text(0.79, 0.9, '4480m', ha='center')
    
    # save figure
    fig.savefig(filename)



### main code

# get all output files
files = [f for f in listdir(output_path) if isfile(join(output_path, f))]

# create table for TP, FP, ...
tab = np.zeros((2*n_per_file*len(files)*len(quantiles)*len(coarsing), 10))

# empty array to calculate thresholds from quantiles
thresholds = np.zeros((len(training_sets), len(cross_validations), len(quantiles)))

# loop over images without a dam
# loop over different models, evaluation datasets, cross validations, get all files
# loop over all files and calculate TP, FP, FN
for training_num, training_set in enumerate(training_sets):
    for cross_num, cross_validation in enumerate(cross_validations):
        predictions = [s for s in files if '_' + training_set + '_' + str(cross_validation) in s and "pred" in s]
        predictions.sort()
        prediction = np.load(output_path + predictions[0])
        # calculate quantiles, based on 2500 samples
        for n_quantile, quantile in enumerate(quantiles):
            thresholds[training_num, cross_num, n_quantile] = np.nanquantile(prediction, quantile)


# counter
counter = 0

# loop over images without a dam
# loop over different models, evaluation datasets, cross validations, get all files
# loop over all files and calculate TP, FP, FN
for training_num, training_set in enumerate(training_sets):
    for test_num, test_set in enumerate(test_sets_no_dams):
        for cross_num, cross_validation in enumerate(cross_validations):
            # get predictions
            predictions = [s for s in files if s.startswith(test_set + '_' + training_set + '_' + str(cross_validation)) and "pred" in s]
            predictions.sort()
            # get mercator information
            mercators = [s for s in files if s.startswith(test_set + '_' + training_set + '_' + str(cross_validation)) and "mercator" in s]
            mercators.sort()
            for i in range(len(predictions)):
                prediction = np.load(output_path + predictions[i])
                mercator = np.load(output_path + mercators[i])
                result = counting_negatives(prediction, thresholds[training_num,cross_num,:], coarsing, training_num, test_num, cross_num, mercator)
                tab[counter:(counter+result.shape[0]),:] = result
                counter += result.shape[0]

# loop for images with a dam
# loop over different models, evaluation datasets, cross validations, get all files, loop over all files and calculate TP, FP, FN
for training_num, training_set in enumerate(training_sets):
    for test_num, test_set in enumerate(test_sets_dams):
        for cross_num, cross_validation in enumerate(cross_validations):
            # get predictions
            predictions = [s for s in files if s.startswith(test_set + '_' + training_set + '_' + str(cross_validation)) and "pred" in s]
            predictions.sort()
            # get ground truth of associated dam height
            truth_files_1 = [s for s in files if s.startswith(test_set + '_' + training_set + '_' + str(cross_validation)) and s.endswith(test_sets_dams[test_num] + '.npy')]
            truth_files_1.sort()
            # get ground truth of dams of all heights
            truth_files_2 = [s for s in files if s.startswith(test_set + '_' + training_set + '_' + str(cross_validation)) and s.endswith('_all.npy')]
            truth_files_2.sort()
            # get mercator information
            mercators = [s for s in files if s.startswith(test_set + '_' + training_set + '_' + str(cross_validation)) and "mercator" in s]
            mercators.sort()
            # loop over all files
            for i in range(len(predictions)):
                # open the different files
                prediction = np.load(output_path + predictions[i])
                truth_file_1 = np.load(output_path + truth_files_1[i])
                truth_file_2 = np.load(output_path + truth_files_2[i])
                mercator = np.load(output_path + mercators[i])
                
                # calculate TP, FP, ...
                result = counting_positives(prediction, truth_file_1, truth_file_2, thresholds[training_num,cross_num,:], coarsing, training_num, test_num + len(test_sets_no_dams), cross_num, mercator)
                # store results in table
                tab[counter:(counter+result.shape[0]),:] = result
                counter += result.shape[0]
                result = counting_all(prediction, truth_file_2, thresholds[training_num,cross_num,:], coarsing, training_num, len(test_sets_dams) + len(test_sets_no_dams), cross_num, mercator)
                tab[counter:(counter+result.shape[0]),:] = result
                counter += result.shape[0]

# remove empty rows
tab = tab[0:counter,:]



# create table for results
results_table = np.zeros((len(test_sets_all)*len(training_sets)*len(quantiles)*len(coarsing)*len(cross_validations), 10))

# counter
counter = 0

# loop over all dimensions (training set, test set, evaluation resolution, cross validations, thresholds) and calculate TP, FP, FN by summing
# store together with identifiers in table
for tr_num, training_set in enumerate(training_sets):
    tab_part = tab[np.where((tab[:,5] == tr_num)),:][0]
    for te_num, test_set in enumerate(test_sets_all):
        tab_part_1 = tab_part[np.where((tab_part[:,6] == te_num)),:][0]
        for r_num, resolution in enumerate(resolutions):
            tab_part_2 = tab_part_1[np.where((tab_part_1[:,4] == r_num)),:][0]
            for c_num, cross_validation in enumerate(cross_validations):
                tab_part_3 = tab_part_2[np.where((tab_part_2[:,7] == c_num)),:][0]
                for thr_num, threshold in enumerate(thresholds[tr_num, c_num]):
                    tab_part_4 = tab_part_3[np.where((tab_part_3[:,3] == thr_num)),:][0]
                    results_table[counter, 0] = tr_num
                    results_table[counter, 1] = te_num
                    results_table[counter, 2] = thr_num
                    results_table[counter, 3] = r_num
                    results_table[counter, 4] = c_num
                    results_table[counter, 5] = np.sum(tab_part_4[:,0])
                    results_table[counter, 6] = np.sum(tab_part_4[:,1])
                    results_table[counter, 7] = np.sum(tab_part_4[:,2])
                    results_table[counter, 8] = tab_part_4.shape[0]
                    counter += 1

results_summary = np.zeros(((len(test_sets_all) - len(test_sets_no_dams))*len(training_sets)*len(quantiles)*len(coarsing)*len(cross_validations), 11))

# counter
counter = 0

# loop over training set, test set, esolution, cross_validation, threshold and calculate weighted TP, FP, FN, recall, precision, F score
for tr_num, training_set in enumerate(training_sets):
    for te_num in range(len(test_sets_no_dams), len(test_sets_all)):
        for r_num, resolution in enumerate(resolutions):
            for c_num, cross_validation in enumerate(cross_validations):
                for thr_num, threshold in enumerate(thresholds[tr_num, c_num]):
                    res_tab = results_table[results_table[:,0] == tr_num,:]
                    res_tab = res_tab[res_tab[:,2] == thr_num,:]
                    res_tab = res_tab[res_tab[:,3] == r_num,:]
                    res_tab = res_tab[res_tab[:,4] == c_num,:]
                    # weigh FP, TP, FN by the ratio of images without dam and with dam for images with dams
                    # this leads to equal weights for both images with and without dams
                    FP = res_tab[res_tab[:,1] == te_num,6]*(res_tab[res_tab[:,1] == 0,8]+res_tab[res_tab[:,1] == 1,8])/res_tab[res_tab[:,1] == te_num,8]+res_tab[res_tab[:,1] == 0,6]+res_tab[res_tab[:,1] == 1,6]
                    TP = res_tab[res_tab[:,1] == te_num,5]*(res_tab[res_tab[:,1] == 0,8]+res_tab[res_tab[:,1] == 1,8])/res_tab[res_tab[:,1] == te_num,8]
                    FN = res_tab[res_tab[:,1] == te_num,7]*(res_tab[res_tab[:,1] == 0,8]+res_tab[res_tab[:,1] == 1,8])/res_tab[res_tab[:,1] == te_num,8]
                    results_summary[counter, 0] = tr_num
                    results_summary[counter, 1] = te_num
                    results_summary[counter, 2] = thr_num
                    results_summary[counter, 3] = r_num
                    results_summary[counter, 4] = c_num
                    results_summary[counter, 5] = TP
                    results_summary[counter, 6] = FP
                    results_summary[counter, 7] = FN
                    results_summary[counter, 8] = TP/(TP + FN) # recall
                    results_summary[counter, 9] = TP/(TP + FP) # precision
                    results_summary[counter, 10] = 2*TP/(2*TP + FP + FN) # f score
                    counter += 1

# save the results         
np.save('results.npy', results_table)
np.save('summary.npy', results_summary)
np.save('results_detailed.npy', tab)



    
# create heatmap
multiplot_heatmap(tab)
    
# create map
multiplot_map(tab)

# create multiplot    
precision_recall_multi(results_summary, [0,1, 2, 3, 4], ['$\geq$15m', '10-<15m', '5-<10m', '<5m', 'All'], [0,1,2], ['280m', '1120m', '4480m'], [2,3,4,5,6], names_cut3 = ['$\geq$15m', '10-<15m', '5-<10m', '<5m', 'All'], colors_cut3 = ['blue', 'red', 'green', 'yellow', 'purple'], filename = 'plots/multi.pdf')
    
# create single precision recall curves
precision_recall_single(results_summary, 0, 0, [2,3,4,5,6], names_cut3 = ['$\geq$15m', '10-<15m', '5-<10m', '<5m', 'All'], colors_cut3 = ['blue', 'red', 'green', 'yellow', 'purple'], filename = 'plots/large_280m.pdf')
precision_recall_single(results_summary, 0, 1, [2,3,4,5,6], names_cut3 = ['$\geq$15m', '10-<15m', '5-<10m', '<5m', 'All'], colors_cut3 = ['blue', 'red', 'green', 'yellow', 'purple'], filename = 'plots/large_1120m.pdf')
precision_recall_single(results_summary, 0, 2, [2,3,4,5,6], names_cut3 = ['$\geq$15m', '10-<15m', '5-<10m', '<5m', 'All'], colors_cut3 = ['blue', 'red', 'green', 'yellow', 'purple'], filename = 'plots/large_4480m.pdf')
precision_recall_single(results_summary, 1, 0, [2,3,4,5,6], names_cut3 = ['$\geq$15m', '10-<15m', '5-<10m', '<5m', 'All'], colors_cut3 = ['blue', 'red', 'green', 'yellow', 'purple'], filename = 'plots/medium_280m.pdf')
precision_recall_single(results_summary, 1, 1, [2,3,4,5,6], names_cut3 = ['$\geq$15m', '10-<15m', '5-<10m', '<5m', 'All'], colors_cut3 = ['blue', 'red', 'green', 'yellow', 'purple'], filename = 'plots/medium_1120m.pdf')
precision_recall_single(results_summary, 1, 2, [2,3,4,5,6], names_cut3 = ['$\geq$15m', '10-<15m', '5-<10m', '<5m', 'All'], colors_cut3 = ['blue', 'red', 'green', 'yellow', 'purple'], filename = 'plots/medium_4480m.pdf')
precision_recall_single(results_summary, 2, 0, [2,3,4,5,6], names_cut3 = ['$\geq$15m', '10-<15m', '5-<10m', '<5m', 'All'], colors_cut3 = ['blue', 'red', 'green', 'yellow', 'purple'], filename = 'plots/small_280m.pdf')
precision_recall_single(results_summary, 2, 1, [2,3,4,5,6], names_cut3 = ['$\geq$15m', '10-<15m', '5-<10m', '<5m', 'All'], colors_cut3 = ['blue', 'red', 'green', 'yellow', 'purple'], filename = 'plots/small_1120m.pdf')
precision_recall_single(results_summary, 2, 2, [2,3,4,5,6], names_cut3 = ['$\geq$15m', '10-<15m', '5-<10m', '<5m', 'All'], colors_cut3 = ['blue', 'red', 'green', 'yellow', 'purple'], filename = 'plots/small_4480m.pdf')
precision_recall_single(results_summary, 3, 0, [2,3,4,5,6], names_cut3 = ['$\geq$15m', '10-<15m', '5-<10m', '<5m', 'All'], colors_cut3 = ['blue', 'red', 'green', 'yellow', 'purple'], filename = 'plots/verysmall_280m.pdf')
precision_recall_single(results_summary, 3, 1, [2,3,4,5,6], names_cut3 = ['$\geq$15m', '10-<15m', '5-<10m', '<5m', 'All'], colors_cut3 = ['blue', 'red', 'green', 'yellow', 'purple'], filename = 'plots/verysmall_1120m.pdf')
precision_recall_single(results_summary, 3, 2, [2,3,4,5,6], names_cut3 = ['$\geq$15m', '10-<15m', '5-<10m', '<5m', 'All'], colors_cut3 = ['blue', 'red', 'green', 'yellow', 'purple'], filename = 'plots/verysmall_4480m.pdf')
precision_recall_single(results_summary, 4, 0, [2,3,4,5,6], names_cut3 = ['$\geq$15m', '10-<15m', '5-<10m', '<5m', 'All'], colors_cut3 = ['blue', 'red', 'green', 'yellow', 'purple'], filename = 'plots/all_280m.pdf')
precision_recall_single(results_summary, 4, 1, [2,3,4,5,6], names_cut3 = ['$\geq$15m', '10-<15m', '5-<10m', '<5m', 'All'], colors_cut3 = ['blue', 'red', 'green', 'yellow', 'purple'], filename = 'plots/all_1120m.pdf')
precision_recall_single(results_summary, 4, 2, [2,3,4,5,6], names_cut3 = ['$\geq$15m', '10-<15m', '5-<10m', '<5m', 'All'], colors_cut3 = ['blue', 'red', 'green', 'yellow', 'purple'], filename = 'plots/all_4480m.pdf')

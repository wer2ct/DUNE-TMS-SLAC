#This script builds time segments and dbscan clusters

#Kieran Wall - University of Virginia - August 2025

#Run - python3 TMSDetectorEffects.py "detsim-file" "output-directory" "file-number" "configuration .json"

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Imports
import numpy as np
import matplotlib.pyplot as plt
import collections
import ROOT as root
import awkward as ak
import uproot
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
import sys
import json
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#JSON for Globals
def load_config(file_path):
    global_vars = globals()
    try:
        with open(file_path, 'r') as config_file:
            config = json.load(config_file)
            for key, value in config.items():
                global_vars[key] = value
                print(f"Set global variable: {key} = {value}")
    except FileNotFoundError:
        print(f"Error: The configuration file '{file_path}' was not found.")
        # Handle the error appropriately, e.g., use default values or exit
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        # Handle the error appropriately
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


#Functions

def BuildInputArray(merged_tree):
    nns = merged_tree['MergedNeutrinoNumber'].array()
    hns = merged_tree['MergedHitNumber'].array()
    trackids = merged_tree['MergedTrackid'].array()
    detsimx = merged_tree['MergedDetSimHitX'].array()
    detsimz = merged_tree['MergedDetSimHitZ'].array()
    detsimt = merged_tree['MergedDetSimHitT'].array()
    detsimPE = merged_tree['MergedDetSimHitPE'].array()
    spill_nos = merged_tree['MergedSpillNumber'].array()
    file_number = merged_tree['MergedFileNumber'].array()
    array = np.column_stack([nns, hns, trackids, detsimx, detsimz, detsimt, detsimPE, file_number, spill_nos])
    sorted_array = array[np.argsort(array[:,5])]
    return(sorted_array)


#This function creates time segments using a Kernel Density Estimate, outputs the hits with the cluster labels appended. 
#This is generic, meaning that we can feed in basically any data sequence containing time series information and have it segmented. 
def MakeKDSegments(hits, bandwidth = 15, mesh_points = 10000, plot = False):
    
    hit_ts = hits[:,5].to_numpy() #extract just the time series data. 
    
    hit_ts_reshaped = hit_ts.reshape(-1, 1)

    # KDE fitting using a gaussian kernel. 
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(hit_ts_reshaped)

    # Evaluate KDE on a fine grid - project our function onto a fine grid to look for local minima
    #This is a necessary step to be able to find our local minima numerically. Basically the kde gives us our functional form, we need to plot it on points to get out the pdf. 
    #The plotting with the points is accomplished by pulling the score_samples method, which returns log likelihoods. We then convert these into actually probabilitiy densities. 
    t_plot = np.linspace(hit_ts.min() - 2, hit_ts.max() + 2, mesh_points).reshape(-1, 1) 
    log_dens = kde.score_samples(t_plot)
    dens = np.exp(log_dens)

    # Find valleys in KDE curve (ie, local minima)
    #Can use scipy peak finding, but invert data so we are really finding the local minima! Returns indices
    inverted = -dens
    valleys, _ = find_peaks(inverted)
    valley_positions = t_plot[valleys].flatten() #this step maps our indices onto actual times. 

    # Assign cluster index based on which region the point falls into
    def assign_cluster(t, boundaries):
        for i, b in enumerate(boundaries): #will loop through boundaries and the last one assigned will be the one it fits into.
            if t < b:
                return(i)
        return(len(boundaries))

    labels = np.array([assign_cluster(t, valley_positions) for t in hit_ts]) #assign labels to our hits. 

    
    if (plot == True): #output a KDE plot
        plt.plot(t_plot[:, 0], dens, label="KDE")
        plt.scatter(hit_ts, np.zeros_like(hit_ts), c=labels, cmap='tab20', s=50, label="Time Segments")
        for v in valley_positions:
            plt.axvline(v, color='gray', linestyle='--', linewidth=1)
        plt.title("KDE-based hit Clustering")
        plt.xlabel("Hit Time")
        plt.yticks([])
        plt.legend()
        plt.show()

    print(f"We found {len(np.unique(labels))} clusters")
    clusters_added = np.column_stack((hits, labels)) #save our data with cluster labels added to the end of each hit row. 
    
    return(clusters_added)

#Creates time segments using the KDE method. Appends segment label to end of the hit info row. 
def CreateTimeSegments(time_sorted_hits):
    spills_segmented = [] #this list will store all of the hits with fine segmentation grouped by spill. 
    spills = np.unique(time_sorted_hits[:,-1]) #grab spill numbers (should always be 0,12 for current sim)
    
    for spill_no in spills:
        selected_spill = spill_no
        hits_in_spill = time_sorted_hits[time_sorted_hits[:,-1] == selected_spill ] #grabbing our desired spill

        #We can include a routine here to scan and eliminate long-lived neutrons.
        hit_time_range = (max(hits_in_spill[:,5]) - min(hits_in_spill[:,5]))
        if hit_time_range > 1200000000: #set a threshold at hits occuring a full spill step late. 
            rel_time = min(hits_in_spill[:,5]) 
            ceiling_time = rel_time + 1200000000
            indices = np.where([hits_in_spill[:,5] > ceiling_time])[1][0] #grab index of hits greater than ceiling time.
            hits_in_spill = np.delete(hits_in_spill, indices, axis = 0)
            print(f'deleted index {indices} from spill {spill_no}')
        
        time_segmented = MakeKDSegments(hits_in_spill, bandwidth = KDE_bandwidth, mesh_points = KDE_meshpoints, plot = False)
        spills_segmented.append(time_segmented)

    return(np.vstack(spills_segmented))

#DBSCAN code
def BiasedDBSCAN(seg_hits_): #scale_vec is a tuple like (z_scale, x_scale, t_scale)
    z_scale = dbscan_scalevec[0]
    x_scale = dbscan_scalevec[1]
    t_scale = dbscan_scalevec[2]
    nns = seg_hits_[:,0]
    hit_xs = seg_hits_[:,3]
    hit_zs = seg_hits_[:,4]
    hit_ts = seg_hits_[:,5]
    if len(np.unique(hit_xs)) > 1 and len(np.unique(hit_zs)) > 1 and len(np.unique(hit_ts)) > 1 : #this serves to check if there is more than 3 unique dimensions in our vector. If none, just assign same label 0 for whole group. 
        min_x = min(hit_xs)
        max_x = max(hit_xs)
        x_range = max_x - min_x
        normalized_xs = (hit_xs + abs(min_x)) / x_range
        scaled_xs = x_scale * normalized_xs

        #Normalize z
        min_z = min(hit_zs)
        max_z = max(hit_zs)
        z_range = max_z - min_z
        normalized_zs = (hit_zs - min_z) / (z_range)
        scaled_zs = z_scale * normalized_zs

        #Normalize t 
        min_t = min(hit_ts)
        max_t = max(hit_ts)
        t_range = max_t - min_t
        normalized_ts = (hit_ts - min_t) / (t_range)
        scaled_ts = t_scale * normalized_ts

        scaled_hit_vecs = []
        for i in range(len(scaled_xs)):
            scaled_hit_vec = [scaled_zs[i], scaled_xs[i], scaled_ts[i]]
            scaled_hit_vecs.append(scaled_hit_vec)

        scaled_hit_vec_array = np.array(scaled_hit_vecs)

        labels = (DBSCAN(eps = dbscan_epsilon, min_samples = dbscan_minpoints).fit(scaled_hit_vec_array)).labels_
    else:
        labels = np.zeros_like(hit_xs)

    
    return(np.column_stack((seg_hits_, labels)))

#This function runs DBSCAN on a full file worth of time segments. 
def FileRunDBSCAN(hits_segmented_):
    spills = np.unique(hits_segmented_[:,-2])
    labeled_segments = []
    for spill_no in spills:
        #the hits in a given spill
        spill_hits = hits_segmented_[(hits_segmented_[:,-2] == spill_no)]
        segments = np.unique(spill_hits[:,-1])
        for segment in segments:
            #the hits in a given segment of a spill
            segment_hits = spill_hits[spill_hits[:,-1] == segment]
            labeled_hits = BiasedDBSCAN(segment_hits)
            labeled_segments.append(labeled_hits)

    return(np.vstack(labeled_segments))

#how well did our time clustering do?
def time_segment_eval(dbscanned, neutrino_info_set):
    time_segment_metric_array = np.zeros((len(neutrino_info_set),3))

    for i in range(len(neutrino_info_set)):
        time_segment_metric_array[i][0] = i

    segment_occupancies = []
    for spill_ in (np.unique(dbscanned[:,-3])):
        spill_hits = dbscanned[dbscanned[:,-3] == spill_]
        for segment in np.unique(spill_hits[:,-2]):
            segment_hits = spill_hits[spill_hits[:,-2] == segment]
            segment_occupancies.append(len(np.unique(segment_hits[:,0])))
            for nn in np.unique(segment_hits[:,0]): #grab all the nns in the segment
                nn_hits = segment_hits[segment_hits[:,0] == nn]

                #grab segment splits
                seen_hits = len(nn_hits)
                seen_PEs = np.sum(nn_hits[:,6])

                #grab totals
                nn_vis_hits = neutrino_info_set[int(nn)][0]
                nn_vis_PEs = neutrino_info_set[int(nn)][1]

                #Update containment array
                if (time_segment_metric_array[int(nn)][1]) <= (seen_hits / nn_vis_hits) :
                    time_segment_metric_array[int(nn)][1] = seen_hits / nn_vis_hits #hits contained

                if (time_segment_metric_array[int(nn)][2]) <= (seen_PEs / nn_vis_PEs) :
                    time_segment_metric_array[int(nn)][2] = seen_PEs / nn_vis_PEs #PEs contained

    hit_content_mask = neutrino_info_set[:,0] != 0
    time_segment_metric_array[hit_content_mask]
    return((time_segment_metric_array[hit_content_mask], segment_occupancies))

#how well did dbscan do?
def dbscan_eval(dbscanned, neutrino_info_set):
    dbscan_metric_array = np.zeros((len(neutrino_info_set),3))

    for i in range((len(neutrino_info_set))):
        dbscan_metric_array[i][0] = i

    dbscan_occupancies = []
    for spill_ in (np.unique(dbscanned[:,-3])):
        spill_hits = dbscanned[dbscanned[:,-3] == spill_]
        for segment in np.unique(spill_hits[:,-2]):
            segment_hits = spill_hits[spill_hits[:,-2] == segment]
            for cluster in np.unique(segment_hits[:,-1]):
                cluster_hits = segment_hits[segment_hits[:,-1] == cluster]
                dbscan_occupancies.append(len(np.unique(cluster_hits[:,0])))
                for nn in np.unique(cluster_hits[:,0]): #grab all the nns in the cluster
                    nn_hits = cluster_hits[cluster_hits[:,0] == nn]

                    #grab segment splits
                    seen_hits = len(nn_hits)
                    seen_PEs = np.sum(nn_hits[:,6])

                    #grab totals
                    nn_vis_hits = neutrino_info_set[int(nn)][0]
                    nn_vis_PEs = neutrino_info_set[int(nn)][1]

                    #Update containment array
                    if (dbscan_metric_array[int(nn)][1]) <= (seen_hits / nn_vis_hits) :
                        dbscan_metric_array[int(nn)][1] = seen_hits / nn_vis_hits #hits contained

                    if (dbscan_metric_array[int(nn)][2]) <= (seen_PEs / nn_vis_PEs) :
                        dbscan_metric_array[int(nn)][2] = seen_PEs / nn_vis_PEs #PEs contained

    hit_content_mask = neutrino_info_set[:,0] != 0
    return((dbscan_metric_array[hit_content_mask], dbscan_occupancies))

#Forms an array transfering our constituent hit information. One entry for each constituent hit. NO entries for core merged hits with no constituents. 
def FormConstituentArray(merged_tree):
    const_nns = merged_tree["constituent_neutrino_numbers"].array()
    const_hns = merged_tree["constituent_hit_numbers"].array()
    const_trackids = merged_tree["constituent_hit_trackids"].array()
    constituency_array_list = []
    for i in range(len(const_nns)):
        merged_group_index = i
        merged_group_nns = const_nns[i]
        merged_group_hns = const_hns[i]
        merged_group_trackids = const_trackids[i]
        core_nn = merged_group_nns[0] #core is definitionally first element of the vector 
        core_hn = merged_group_hns[0]
        for j in range(1,len(merged_group_nns)): #loop over remaining indices, these are constituents. Only runs if more than one entry. 
            constituency_info = [core_nn, core_hn, merged_group_nns[j], merged_group_hns[j], merged_group_trackids[j]]
            constituency_array_list.append(constituency_info)
            
    constituency_array = np.vstack(constituency_array_list)
    return(constituency_array)

#Saves our clustered hits to a .npz file
def SaveToNPZ(dbscan_clustered_hits, dbscan_epsilon, file_number_, out_dir):
    outpath = out_dir + 'hits_clustered_' + 'epsilon_' + str(dbscan_epsilon) + '_' + str(file_number_) + '.npz' 
    np.savez(outpath, first = dbscan_clustered_hits)
    return(outpath)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Main Function 
def main():
    #add command line arguments for the clustering parameters

    #Load the configuration
    config_file_path = sys.argv[4]
    load_config(config_file_path)
    
    detsim_file = uproot.open(sys.argv[1])
    output_directory = sys.argv[2]
    file_number = sys.argv[3]
    merged_hits_tree = detsim_file["MergedTree"]
    neutrino_tree = detsim_file["NeutrinoVertexTree"]

    #Extract our root info into an array and sort by time. 
    time_sorted_hits_ = BuildInputArray(merged_hits_tree)
    print("Sorted hits by time")
    
    #Segment by time. 
    time_segmented_hits = CreateTimeSegments(time_sorted_hits_)
    print("Completed KDE Segmentation")
    
    #DBSCAN
    dbscanned_hits = FileRunDBSCAN(time_segmented_hits) 
    print("Completed DBSCAN Segmentation")

    #Eval
    print("Beginning Eval")
    neutrino_info_set = np.column_stack( (neutrino_tree["VisibleHits"].array(), neutrino_tree["VisiblePEs"].array() ) ).to_numpy()
    dbscan_metrics, dbscan_occ = dbscan_eval(dbscanned_hits, neutrino_info_set)
    time_seg_metrics, time_seg_occ = time_segment_eval(dbscanned_hits, neutrino_info_set)

    #Forming Constituents
    print("Beginning to create constituent hit array")
    constituent_hit_array = FormConstituentArray(merged_hits_tree)
    
    #Save
    outpath = output_directory + 'hits_clustered_' + 'epsilon_' + str(dbscan_epsilon) + '_' + str(file_number) + '.npz' 
    np.savez(outpath, dbscan_hits = dbscanned_hits, dbscan_metric_array = dbscan_metrics, dbscan_occupancy = dbscan_occ, timeseg_metric_array = time_seg_metrics, timeseg_occupancy = time_seg_occ, constituents_array = constituent_hit_array)
    print(f"Saved clustered hits to path {outpath}")
    
main()
    
    
        









    


    
#This script takes as input an edep-sim file and outputs a ROOT file of TMS hit instances with TMS detector effects applied. This is in need of modularization at the class level at least. 

#Kieran Wall - University of Virginia - August 2025

#Run - python3 TMSDetectorEffects.py "edep-sim-file" "output-directory" "file-number"

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Imports
#I apologize for using both root and uproot at different parts. Building trees is so much easier with PyROOT that I just had to.
import uproot
import matplotlib.pyplot as plt
import awkward as ak
import numpy as np
import math
import ROOT as root
from array import array
from numba import jit
from collections import defaultdict
import sys
import ctypes
from numba import njit, types
from numba.typed import Dict, List
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Classes

#the TMS hit class - pulls directly from the edep-sim hitsegments vector. 

class TMS_Hit:
    def __init__(self, tms_hit_seg, neutrino_number, hit_number, geo, spill_number, trackid):
        self.tms_hit_seg = tms_hit_seg
        self.neutrino_number = neutrino_number #associated with what neutrino vertex
        self.hit_number = hit_number #which TMS hit in the event
        self.trackid = trackid
        self.spill_number = spill_number
        self.geo = geo
        self.x_diff = 0.
        self.y_diff = 0.
        self.z_diff = 0.
        self.Dx = 0.
        self.widths, self.bar_orientation, self.bar_no, self.layer_no, self.bar_positions = ModuleFinder(self.GetHitTrueX(), self.GetHitTrueY(), self.GetHitTrueZ() , self.geo)

    #returns hit neutrino number (an initialization input)
    def GetNeutrinoNumber(self):
        return(self.neutrino_number)

    #returns hit number (an initialization input)
    def GetHitNumber(self):
        return(self.hit_number)

    def GetSpillNumber(self):
        return(self.spill_number)

    def GetHitTrackid(self):
        return(self.trackid)
        
    #returns averaged x position of edep-sim hit segments within the bar    
    def GetHitTrueX(self):
        x_pos = (self.tms_hit_seg.GetStart()[0] + self.tms_hit_seg.GetStop()[0]) / 2 
        return(x_pos)

    #returns averaged y position of edep-sim hit segments within the bar 
    def GetHitTrueY(self):
        y_pos = (self.tms_hit_seg.GetStart()[1] + self.tms_hit_seg.GetStop()[1]) / 2
        return(y_pos)

    #returns averaged z position of edep-sim hit segments within the bar 
    def GetHitTrueZ(self):
        z_pos = (self.tms_hit_seg.GetStart()[2] + self.tms_hit_seg.GetStop()[2]) / 2
        return(z_pos)

    #returns averaged t of edep-sim hit segments within the bar 
    def GetHitTrueT(self):
        time = (self.tms_hit_seg.GetStart()[3] + self.tms_hit_seg.GetStop()[3]) / 2
        return(time)

    #returns the true Dx 
    def GetTrueDx(self):
        self.x_diff = (self.tms_hit_seg.GetStart()[0] - self.tms_hit_seg.GetStop()[0])
        self.y_diff = (self.tms_hit_seg.GetStart()[1] - self.tms_hit_seg.GetStop()[1])
        self.z_diff = (self.tms_hit_seg.GetStart()[2] - self.tms_hit_seg.GetStop()[2])
        self.Dx = ((self.x_diff)**2 + (self.y_diff)**2 + (self.z_diff)**2)**(1/2)
        return(self.Dx)

    #returns the true primary deposit (in MeV)
    def GetTruePrimaryDeposit(self):
        PrimaryDeposit = self.tms_hit_seg.GetEnergyDeposit()
        return(PrimaryDeposit)

    #returns the PE with detector effects applied (in PE)
    def GetDetSimPE(self):
        #note, the 0th index here is what we generally want, the other two are for detector sim calls. 
        initial_PE = self.GetTruePrimaryDeposit() * 50.0  #E to PE with conversion -- this is light yield. TODO - change this function to simply accept an energy deposition?
        suppressed_PE = BirkSuppress( self.GetTruePrimaryDeposit(), self.GetTrueDx(), initial_PE ) #apply BirkSuppression
        detsim_PE, short_pe, long_pe = FiberLengthSim(suppressed_PE, self)
        return(detsim_PE, short_pe, long_pe)
        
    #Returns pedestal subtracted status -- using default pedestal subtraction threshold, could take this as a parameter.
    def GetPedestalSubtractedStatus(self): #true if hit is pedestal subtracted
        if (self.GetDetSimPE()[0] < 3.0):
            return(True)

        else:
            return(False)
        
    #Attributes which are dependent on the bar segmentation

    #Returns our DetSim Position 
    def GetBarHitPos(self):
        return(self.bar_positions)

    #Returns our Bar Number
    def GetBarNo(self):
        return(self.bar_no)

    #Returns our Layer Number
    def GetBarLayer(self):
        return(self.layer_no)  

    #Returns our bar orientation
    def GetBarOrientation(self):
        return(self.bar_orientation)
        
    #Returns our DetSimTime
    def GetDetSimHitT(self):
        return(HitTimingSim(self))


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#All Functions

#Save neutrino vertex information. Saves all neutrino vertices, even ones not leaving TMS hits, such that we can use row index as nn when needed 
#All Functions

#Save neutrino vertex information. Saves all neutrino vertices, even ones not leaving TMS hits, such that we can use row index as nn when needed 
def CreateVtxContainers(tracker_tree):
    edep_true_neutrino_vtx = [] 
    vtxs = array('d', [0.0]*5) 
    tracker_tree.SetBranchAddress("EvtVtx", vtxs)
    for i in range(tracker_tree.GetEntries()):
        tracker_tree.GetEntry(i)
        #don't forget to scale positions (tracker inexplicably uses m)
        interaction_time = vtxs[3] / (10**9) #in seconds, to figure out spill
        spill = interaction_time // 1.2 #based on a 1.2 second spill separation. This will need to be changed if beam changes. 
        vtx_data = {
            "neutrino_num": i,
            "x": vtxs[0]*1000,
            "y": vtxs[1]*1000,
            "z": vtxs[2]*1000,
            "t": vtxs[3],
            "spill": spill,
        }
        edep_true_neutrino_vtx.append(vtx_data)

    #Create an array with vertex information - for saving as output.
    neutrino_number = []
    neutrino_x = []
    neutrino_y = []
    neutrino_z = []
    neutrino_t = []
    neutrino_spill = []
    for entry in edep_true_neutrino_vtx:
        neutrino_number.append(entry['neutrino_num'])
        neutrino_x.append(entry['x'])
        neutrino_y.append(entry['y'])
        neutrino_z.append(entry['z'])
        neutrino_t.append(entry['t'])
        neutrino_spill.append(entry['spill'])

    neutrino_vertex_array = np.column_stack((neutrino_number,neutrino_x,neutrino_y,neutrino_z,neutrino_t,neutrino_spill))

    return(neutrino_vertex_array)
    
#Handles the geometry mapping. 
def ModuleFinder(x,y,z,geom):
    module_names = {'modulelayervol1' : 'U' ,'modulelayervol2' : 'V' ,'modulelayervol3': 'X' ,'modulelayervol4' : 'Y'}
    node = geom.FindNode(x, y, z)
    orientation = 'null'
    bar_number = 'null'
    layer_number = 'null'
    widths = [0,0,0]
    bar_positions = array('d', [x, y, z]) #hold bar positions as just hit x,y,z for now bar_positions = array('d', [0.0, 0.0, 0.0])
    while node:
        if 'scinBox' in node.GetName(): #this is like a rlly stupid line but let's just hope it works
            bar_number = node.GetNumber()
            box = geom.GetCurrentVolume().GetShape()
            local = array('d', [0.0, 0.0, 0.0])  # 'd' = C double
            geom.GetCurrentMatrix().LocalToMaster(local, bar_positions) #assigning the center of the bar to the hit -- basically just PyROOTED dune-tms code.
            widths[0] = 2 * box.GetDX()
            widths[1] = 2 * box.GetDY()
            widths[2] = 2 * box.GetDZ()
        if "modulelayervol" in node.GetName():
            layer_number = node.GetNumber()
            for module_name in module_names.keys():
                if module_name in node.GetName():
                    orientation = module_names[module_name]
                    #print(module_name)
        if 'volDetEnclosure' in node.GetName():
            break
        geom.CdUp()
        node = geom.GetCurrentNode()

    if orientation == 'X':
        bar_positions[0] = -99999000  # remove X information -- irrelevant given geometry
        xw, yw = yw, xw               # flip widths --apparently root handles incorrectly for x-bars
    elif orientation in ['U', 'V', 'Y']:
        bar_positions[1] = -99999000  # remove Y information -- irrelevant given geometry
    else:
        bar_positions = [-99999000, -99999000, -99999000]
    return (widths, orientation, bar_number, layer_number, bar_positions)


# ------ Functions pertaining to the Optical Simulation, Effects PE ------- #

rand_seed = 42 #this is technically a global variable -- TODO, place in the global variables section. 
def GetTrueDistanceFromReadout(TMS_hit):
    TMS_Start_Bars_Only = [-3350, -2950]
    TMS_End_Bars_Only = [3350, 240]
    #XBars are weird
    if (TMS_hit.GetBarOrientation == 'X'):
        if (TMS_hit.GetHitTrueX() < 0):
            return(TMS_hit.GetHitTrueX() - TMS_Start_Bars_Only[0])
        else:
            return(TMS_End_Bars_Only[0] - TMS_hit.GetHitTrueX())
    #All other bar orientations (U,V,Y)
    else:
        return(TMS_End_Bars_Only[1] - TMS_hit.GetHitTrueY())

def GetTrueLongDistanceFromReadout(TMS_hit):
    TMS_Start_Bars_Only = [-3350, -2950]
    TMS_End_Bars_Only = [3350, 240]
    additional_length = 0.
    #XBars
    if (TMS_hit.GetBarOrientation == 'X'):
        additional_length = 2 * TMS_End_Bars_Only[0] #2 * XBar Length
    #All other bar orientations (U,V,Y)
    else:
        additional_length = 2 * (TMS_End_Bars_Only[1] - TMS_Start_Bars_Only[1]) #2* YBar Length
    return( additional_length - GetTrueDistanceFromReadout(TMS_hit) )

def GetTrueDistanceFromMiddle(TMS_hit):
    TMS_Start_Bars_Only = [-3350, -2950]
    TMS_End_Bars_Only = [3350, 240]
    additional_length = 0.
    #XBars
    if (TMS_hit.GetBarOrientation == 'X'):
        additional_length = 0.5 * TMS_End_Bars_Only[0]
    #All other bar orientations (U,V,Y)
    else:
        additional_length = 0.5 * (TMS_End_Bars_Only[1] - TMS_Start_Bars_Only[1])
    return( GetTrueDistanceFromReadout(TMS_hit) - additional_length )
        
def GetTrueLongDistanceFromMiddle(TMS_hit):
    TMS_Start_Bars_Only = [-3350, -2950]
    TMS_End_Bars_Only = [3350, 240]
    additional_length = 0.
    #XBars
    if (TMS_hit.GetBarOrientation == 'X'):
        additional_length = 0.5 * TMS_End_Bars_Only[0]
    #All other bar orientations (U,V,Y)
    else:
        additional_length = 0.5 * (TMS_End_Bars_Only[1] - TMS_Start_Bars_Only[1])
    return( GetTrueLongDistanceFromReadout(TMS_hit) - additional_length )

#Models the effect of birk suppression on the total PE
def BirkSuppress(de, dx, pe): #de is energy
    birks_constant = 0.0905
    dedx = 0
    if (dx > 1e-8):
        dedx = de / dx
    else: 
        dedx = de / 1.0;
    return(pe * (1.0 / (1.0 + birks_constant * dedx)))

#Selects PEs as traveling in the long or short directions - then models the effect on the PE for traveling the long or short distance. Returns corrected PE
def FiberLengthSim(pe, TMS_hit):
    #Constant used
    wsf_decay_constant = 1 / 4.160
    wsf_length_multiplier = 1.8 
    wsf_fiber_reflection_eff = 0.95
    readout_coupling_eff = 1.0
    
    #Select the PEs traveling in either direction. 
    rng = np.random.default_rng(seed = rand_seed)
    pe_short = pe #what is number of pe's that travel the short path
    pe_long = 0 #what is the number of pe's that travel the long path
    pe = rng.poisson(pe)
    pe_short = rng.binomial(n = pe, p = 0.5)
    pe_long = pe - pe_short
    
    #these next calculations are done in meters to make it a bit easier, hence the conversions, models attenuation. 
    distance_from_end = GetTrueDistanceFromReadout(TMS_hit) * 1e-3
    long_way_distance_from_end = GetTrueLongDistanceFromReadout(TMS_hit) * 1e-3
    distance_from_end *= wsf_length_multiplier
    long_way_distance_from_end *= wsf_length_multiplier

    #attentuation
    pe_short = pe_short * math.exp(-wsf_decay_constant * distance_from_end)
    pe_long = pe_long * math.exp(-wsf_decay_constant * long_way_distance_from_end) * wsf_fiber_reflection_eff
    
    #could couple to additional optical fiber but we are going to go ahead and neglect that - default setting is to neglect. 
    pe_short *= readout_coupling_eff
    pe_long *= readout_coupling_eff

    #return the PE post attenuation, as well as some other useful bits of information for other functions 
    return( ( (pe_short + pe_long), pe_short, pe_long ) )


# ------ Functions pertaining to the Hit-Wise Timing Simulation ------- #

def HitTimingSim(TMS_hit):
    rng = np.random.default_rng(seed=rand_seed)
    #Constant
    SPEED_OF_LIGHT = 0.2998  # m/ns
    FIBER_N = 1.5
    SPEED_OF_LIGHT_IN_FIBER = SPEED_OF_LIGHT / FIBER_N
    scintillator_decay_time = 3.0  # ns
    wsf_decay_time = 20.0  # ns
    wsf_length_multiplier = 1.8 

    # Gaussian noise: mean = 0.0, std dev = 1.0 ns
    noise_distribution = lambda: rng.normal(loc=0.0, scale=1.0)

    # Coin flip: returns 0 or 1
    coin_flip = lambda: rng.integers(low=0, high=2)  # high=2 is exclusive

    # Exponential decay distributions (rate = 1 / decay time)
    exp_scint = lambda: rng.exponential(scale=scintillator_decay_time)  # mean = 3 ns
    exp_wsf = lambda: rng.exponential(scale=wsf_decay_time)             # mean = 20 ns

    #begin determining timing effect t here is the added time. 
    t = 0.
    t += noise_distribution()
    distance_from_middle = (GetTrueDistanceFromMiddle(TMS_hit) * 1e-3) * wsf_length_multiplier
    long_way_distance = (GetTrueLongDistanceFromMiddle(TMS_hit) * 1e-3) * wsf_length_multiplier
    time_correction = distance_from_middle / SPEED_OF_LIGHT_IN_FIBER;
    time_correction_long_way = long_way_distance / SPEED_OF_LIGHT_IN_FIBER;

    pe_short_path = TMS_hit.GetDetSimPE()[1]
    pe_long_path = TMS_hit.GetDetSimPE()[2]
    minimum_time_offset = 1e100

    MAX_PE_THROWS = 300

    #serves to cap the number of throws we do (little difference past 300)
    if (pe_short_path > MAX_PE_THROWS):
        pe_short_path = MAX_PE_THROWS
    while (pe_short_path > 0):
        time_offset = time_correction
        time_offset += exp_scint()
        time_offset += exp_wsf()
        minimum_time_offset = min(time_offset, minimum_time_offset) #take whatever is less
        pe_short_path -= 1

    #serves to cap the number of throws that we do (little difference past 300)
    if (pe_long_path > MAX_PE_THROWS):
        pe_long_path = MAX_PE_THROWS
    
    while (pe_long_path > 0):
        time_offset = time_correction_long_way
        time_offset += exp_scint()
        time_offset += exp_wsf()
        minimum_time_offset = min(time_offset, minimum_time_offset) #take whatever is less
        pe_long_path -= 1

    t += minimum_time_offset
    return (t + TMS_hit.GetHitTrueT()) #return the true hit time + the determined travel effects.

    #known issue here, 1e100 is returned as the time for hits w/ 0 PE post evaluating the poisson throw. These will be tossed anyways by the pedestal suppression but in case someone sees something weird at this step. 

# ------ Functions pertaining to Coincident Hit Merging ------- #

#Sorts hits by bar
def SortByBar(hit_info_array):
    N = hit_info_array.shape[0]

    # Group hits by (layerno, barno)
    group_dict = Dict.empty(
        key_type=types.UniTuple(types.float64, 2),
        value_type=types.ListType(types.int64)
    )

    for i in range(N):
        layer = hit_info_array[i, 1]
        bar = hit_info_array[i, 0]
        key = (layer, bar)
        if key not in group_dict:
            group_dict[key] = List.empty_list(types.int64)
        group_dict[key].append(i)
        
    return(group_dict)

#This function will take our group dictionary, detsim_hit_array, and a readout window length, and sorts hits in bars into timing groups
def SortByBar(hit_info_array):
    N = hit_info_array.shape[0]

    # Group hits by (layerno, barno)
    group_dict = Dict.empty(
        key_type=types.UniTuple(types.float64, 2),
        value_type=types.ListType(types.int64)
    )

    for i in range(N):
        layer = hit_info_array[i, 1]
        bar = hit_info_array[i, 0]
        key = (layer, bar)
        if key not in group_dict:
            group_dict[key] = List.empty_list(types.int64)
        group_dict[key].append(i)
        
    return(group_dict)

#This function will take our group dictionary, detsim_hit_array, and a readout window length, and sorts hits in bars into timing groups
def CreateTimeGroups(hit_array, bar_dictionary, readout_window = 120):
    all_groups = []
    for key in bar_dictionary:
        index_list = bar_dictionary[key]
        
        times = [hit_array[i][2] for i in index_list]
        sorted_idx = np.argsort(times) #returns indixes that would sort a list. - not matched to hits
        sorted_times = [times[i] for i in sorted_idx] #times sorted by times
        sorted_indices = [index_list[i] for i in sorted_idx] #return sorted list of hit indices
        start = 0
        time_index_array = np.column_stack((sorted_times, sorted_indices)) #perhaps a useful object
        groups = []
        current_group = [int(time_index_array[0][1])]
        t_start = time_index_array[0][0]

        for i in range(1, time_index_array.shape[0]):
            t = time_index_array[i][0]
            idx = int(time_index_array[i][1])

            if t - t_start <= readout_window:
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]
                t_start = t

        # Add the final group
        if current_group:
            groups.append(current_group)

        # Print and append result
        for i, group in enumerate(groups):
            #print(f"Key {key} - Group {i}: {group}")

            all_groups.append(group)
            
    return(all_groups)

# ------ Functions pertaining to creating output trees ------- #

#Unmerged Tree
def CreateUnmergedTree(detsim_hits_generator, root_output_file, file_number_):
    root_output_file.cd()
    UnmergedTree = root.TTree("UnmergedTree", "All non-pedestal subtracted TMS hits w/ true and detsim information")
    
    # all our branch variables
    neutrino_number = ctypes.c_long(0)
    hit_number = ctypes.c_long(0)
    spill_number = ctypes.c_long(0)
    file_number = ctypes.c_long(0)
    trackid = ctypes.c_long(0)
    
    TrueHitX = ctypes.c_double(0)
    TrueHitY = ctypes.c_double(0)
    TrueHitZ = ctypes.c_double(0)
    TrueHitT = ctypes.c_double(0)
    TrueHitDx = ctypes.c_double(0)
    TruePrimDep = ctypes.c_double(0)
    
    DetSimHitX = ctypes.c_double(0)
    DetSimHitY = ctypes.c_double(0)
    DetSimHitZ = ctypes.c_double(0)
    DetSimHitT = ctypes.c_double(0)
    DetSimHitPE = ctypes.c_double(0)
    
    DetSimHitBarNo = ctypes.c_long(0)
    DetSimHitLayerNo = ctypes.c_long(0)
    DetSimHitBarOrient = ctypes.c_long(0)
    
    # Declaring all our branches
    UnmergedTree.Branch("neutrino_number", neutrino_number, "neutrino_number/L")
    UnmergedTree.Branch("hit_number", hit_number, "hit_number/L")
    UnmergedTree.Branch("spill_number", spill_number, "spill_number/L")
    UnmergedTree.Branch("trackid", trackid, "trackid/L")
    UnmergedTree.Branch("file_number", file_number, "file_number/L")
    UnmergedTree.Branch("TrueHitX", TrueHitX, "TrueHitX/D")
    UnmergedTree.Branch("TrueHitY", TrueHitY, "TrueHitY/D")
    UnmergedTree.Branch("TrueHitZ", TrueHitZ, "TrueHitZ/D")

    UnmergedTree.Branch("TrueHitT", TrueHitT, "TrueHitT/D")
    UnmergedTree.Branch("TrueHitDx", TrueHitDx, "TrueHitDx/D")
    UnmergedTree.Branch("TruePrimDep", TruePrimDep, "TruePrimDep/D")
    UnmergedTree.Branch("DetSimHitX", DetSimHitX, "DetSimHitX/D")
    UnmergedTree.Branch("DetSimHitY", DetSimHitY, "DetSimHitY/D")
    UnmergedTree.Branch("DetSimHitZ", DetSimHitZ, "DetSimHitZ/D")
    UnmergedTree.Branch("DetSimHitT", DetSimHitT, "DetSimHitT/D")
    UnmergedTree.Branch("DetSimHitPE", DetSimHitPE, "DetSimHitPE/D")
    UnmergedTree.Branch("DetSimHitBarNo", DetSimHitBarNo, "DetSimHitBarNo/L") 
    UnmergedTree.Branch("DetSimHitLayerNo", DetSimHitLayerNo, "DetSimHitLayerNo/L") 
    UnmergedTree.Branch("DetSimHitBarOrient", DetSimHitBarOrient, "DetSimHitBarOrient/L") 

    # Now loop to fill our branches
    for hit in detsim_hits_generator:
        if hit.GetBarNo() == 'null':
            #print("Had to skip due to a bad bar number") 
            #these are peculiar hits - nodes connect to a layer but not a scintillator bar, ie they aren't associated with a scintillator volume. 
            #I would put money on this being an issue in dune-tms as well whether its been spotted or not.
            #just going to log as a known issue and continue onwards. 
            continue
        bar_orientation = hit.GetBarOrientation()
        bar_orientation_coded = -1
        if bar_orientation == 'U':
            bar_orientation_coded = 0
        elif bar_orientation == 'V':
            bar_orientation_coded = 1
        elif bar_orientation == 'X':
            bar_orientation_coded = 2
        else:
            bar_orientation_coded = -1
        
        # Set values using .value
        file_number.value = file_number_
        neutrino_number.value = hit.GetNeutrinoNumber()
        hit_number.value = hit.GetHitNumber()
        spill_number.value = int(hit.GetSpillNumber())
        trackid.value = hit.GetHitTrackid()
        TrueHitX.value = hit.GetHitTrueX()
        TrueHitY.value = hit.GetHitTrueY()
        TrueHitZ.value = hit.GetHitTrueZ()
        TrueHitT.value = hit.GetHitTrueT()
        TrueHitDx.value = hit.GetTrueDx()
        TruePrimDep.value = hit.GetTruePrimaryDeposit()
        DetSimHitX.value = hit.GetBarHitPos()[0]
        DetSimHitY.value = hit.GetBarHitPos()[1]
        DetSimHitZ.value = hit.GetBarHitPos()[2]
        DetSimHitT.value = hit.GetDetSimHitT()
        DetSimHitPE.value = hit.GetDetSimPE()[0] 
        DetSimHitBarNo.value = hit.GetBarNo()
        DetSimHitLayerNo.value = hit.GetBarLayer()
        DetSimHitBarOrient.value = bar_orientation_coded
        
        UnmergedTree.Fill()
        
    root_output_file.Write()
    root_output_file.Close()

#Merged Tree
def CreateMergedTree(Unmerged_Tree_Uproot, root_output_file, merged_index_groups, file_number_):
    root_file = root.TFile(root_output_file, "UPDATE")
    root_file.cd()
    MergedTree = root.TTree("MergedTree", "TMS Hits with hit merging applied")

    # The variables to be filled from a single merged hit
    MergedNeutrinoNumber = ctypes.c_long(0)
    MergedHitNumber = ctypes.c_long(0)
    MergedTrackid = ctypes.c_long(0)
    MergedSpillNumber = ctypes.c_long(0)
    MergedFileNumber = ctypes.c_long(0)
    
    MergedTrueHitX = ctypes.c_double(0)
    MergedTrueHitY = ctypes.c_double(0)
    MergedTrueHitZ = ctypes.c_double(0)
    MergedTrueHitT = ctypes.c_double(0)
    MergedTrueHitDx = ctypes.c_double(0)
    MergedTruePrimDep = ctypes.c_double(0)
    
    MergedDetSimHitX = ctypes.c_double(0)
    MergedDetSimHitY = ctypes.c_double(0)
    MergedDetSimHitZ = ctypes.c_double(0)
    MergedDetSimHitT = ctypes.c_double(0)
    MergedDetSimHitPE = ctypes.c_double(0)
    MergedDetSimHitBarNo = ctypes.c_long(0)
    MergedDetSimHitLayerNo = ctypes.c_long(0)
    MergedDetSimHitBarOrient = ctypes.c_long(0)

    # The vectors to store constituent info
    constituent_neutrino_numbers = root.std.vector('int')()
    constituent_hit_numbers = root.std.vector('int')()
    constituent_hit_trackids = root.std.vector('int')()
    constituent_hitTs = root.std.vector('double')()
    constituent_hitPEs = root.std.vector('double')()
    
    # Declare the branches for MergedTree
    MergedTree.Branch("MergedNeutrinoNumber", MergedNeutrinoNumber, "MergedNeutrinoNumber/L")
    MergedTree.Branch("MergedHitNumber", MergedHitNumber, "MergedHitNumber/L")
    MergedTree.Branch("MergedSpillNumber", MergedSpillNumber, "MergedSpillNumber/L")
    MergedTree.Branch("MergedTrackid", MergedTrackid, "MergedTrackid/L")
    MergedTree.Branch("MergedFileNumber", MergedFileNumber, "MergedFileNumber/L")
    MergedTree.Branch("MergedTrueHitX", MergedTrueHitX, "MergedTrueHitX/D")
    MergedTree.Branch("MergedTrueHitY", MergedTrueHitY, "MergedTrueHitY/D")
    MergedTree.Branch("MergedTrueHitZ", MergedTrueHitZ, "MergedTrueHitZ/D")
    MergedTree.Branch("MergedTrueHitT", MergedTrueHitT, "MergedTrueHitT/D")
    MergedTree.Branch("MergedTrueHitDx", MergedTrueHitDx, "MergedTrueHitDx/D")
    MergedTree.Branch("MergedTruePrimDep", MergedTruePrimDep, "MergedTruePrimDep/D")
    MergedTree.Branch("MergedDetSimHitX", MergedDetSimHitX, "MergedDetSimHitX/D")
    MergedTree.Branch("MergedDetSimHitY", MergedDetSimHitY, "MergedDetSimHitY/D")
    MergedTree.Branch("MergedDetSimHitZ", MergedDetSimHitZ, "MergedDetSimHitZ/D")
    MergedTree.Branch("MergedDetSimHitT", MergedDetSimHitT, "MergedDetSimHitT/D")
    MergedTree.Branch("MergedDetSimHitPE", MergedDetSimHitPE, "MergedDetSimHitPE/D")
    MergedTree.Branch("MergedDetSimHitBarNo", MergedDetSimHitBarNo, "MergedDetSimHitBarNo/L")
    MergedTree.Branch("MergedDetSimHitLayerNo", MergedDetSimHitLayerNo, "MergedDetSimHitLayerNo/L")
    MergedTree.Branch("MergedDetSimHitBarOrient", MergedDetSimHitBarOrient, "MergedDetSimHitBarOrient/L")

    MergedTree.Branch("constituent_neutrino_numbers", constituent_neutrino_numbers)
    MergedTree.Branch("constituent_hit_numbers", constituent_hit_numbers)
    MergedTree.Branch("constituent_hit_trackids", constituent_hit_trackids)
    MergedTree.Branch("constituent_hitTs", constituent_hitTs)
    MergedTree.Branch("constituent_hitPEs", constituent_hitPEs)

    # Use Uproot to fetch data for all branches at once
    # This is much faster and cleaner than GetEntry() in a loop
    unmerged_hit_data = Unmerged_Tree_Uproot.arrays(
        ["neutrino_number", "hit_number", "trackid", "spill_number", "TrueHitX", "TrueHitY", "TrueHitZ", "TrueHitT", 
         "TrueHitDx", "TruePrimDep", "DetSimHitX", "DetSimHitY", "DetSimHitZ", "DetSimHitT", 
         "DetSimHitPE", "DetSimHitBarNo", "DetSimHitLayerNo", "DetSimHitBarOrient"],
        library="np"
    )

    # Loop over the merged hit index groups
    for index_group in merged_index_groups:
        # Get the core hit information from the first index in the group
        core_index = index_group[0]
        
        # Populate merged tree variables from the core hit
        MergedFileNumber.value = file_number_
        MergedNeutrinoNumber.value = unmerged_hit_data["neutrino_number"][core_index]
        MergedHitNumber.value = unmerged_hit_data["hit_number"][core_index]
        MergedSpillNumber.value = unmerged_hit_data["spill_number"][core_index]
        MergedTrackid.value = unmerged_hit_data["trackid"][core_index]
        MergedTrueHitX.value = unmerged_hit_data["TrueHitX"][core_index]
        MergedTrueHitY.value = unmerged_hit_data["TrueHitY"][core_index]
        MergedTrueHitZ.value = unmerged_hit_data["TrueHitZ"][core_index]
        MergedTrueHitT.value = unmerged_hit_data["TrueHitT"][core_index]
        MergedTrueHitDx.value = unmerged_hit_data["TrueHitDx"][core_index]
        MergedDetSimHitX.value = unmerged_hit_data["DetSimHitX"][core_index]
        MergedDetSimHitY.value = unmerged_hit_data["DetSimHitY"][core_index]
        MergedDetSimHitZ.value = unmerged_hit_data["DetSimHitZ"][core_index]
        MergedDetSimHitT.value = unmerged_hit_data["DetSimHitT"][core_index]
        MergedDetSimHitBarNo.value = unmerged_hit_data["DetSimHitBarNo"][core_index]
        MergedDetSimHitLayerNo.value = unmerged_hit_data["DetSimHitLayerNo"][core_index]
        MergedDetSimHitBarOrient.value = unmerged_hit_data["DetSimHitBarOrient"][core_index]

        # Tally the PEs and true deposits and fill constituent vectors
        PrimDeps = 0.0
        PEs = 0.0
        constituent_neutrino_numbers.clear()
        constituent_hit_numbers.clear()
        constituent_hit_trackids.clear()
        constituent_hitTs.clear()
        constituent_hitPEs.clear()

        for index in index_group:
            PrimDeps += unmerged_hit_data["TruePrimDep"][index]
            PEs += unmerged_hit_data["DetSimHitPE"][index]

            constituent_neutrino_numbers.push_back(int(unmerged_hit_data["neutrino_number"][index]))
            constituent_hit_numbers.push_back(int(unmerged_hit_data["hit_number"][index]))
            constituent_hit_trackids.push_back(int(unmerged_hit_data["trackid"][index]))
            constituent_hitTs.push_back(unmerged_hit_data["DetSimHitT"][index])
            constituent_hitPEs.push_back(unmerged_hit_data["DetSimHitPE"][index])

        MergedTruePrimDep.value = PrimDeps
        MergedDetSimHitPE.value = PEs

        MergedTree.Fill()
    
    root_file.Write()
    root_file.Close()


#Neutrino Vertex Tree
def CreateNeutrinoTree(vertex_array, root_output_file, file_number_, neutrino_info_array):
    root_file = root.TFile(root_output_file, "UPDATE")
    root_file.cd()
    NeutrinoVertexTree = root.TTree("NeutrinoVertexTree", "Neutrino Vertices")

    # The variables to be filled from a single merged hit
    NeutrinoNumber = ctypes.c_long(0)
    FileNumber = ctypes.c_long(0)
    NeutrinoSpill = ctypes.c_long(0)
    
    NeutrinoVtxX = ctypes.c_double(0)
    NeutrinoVtxY = ctypes.c_double(0)
    NeutrinoVtxZ = ctypes.c_double(0)
    NeutrinoVtxT = ctypes.c_double(0)

    VisibleHits = ctypes.c_long(0)
    VisiblePEs = ctypes.c_double(0)
    
    
    # Declare the branches for MergedTree
    NeutrinoVertexTree.Branch("NeutrinoVtxX", NeutrinoVtxX, "NeutrinoVtxX/D")
    NeutrinoVertexTree.Branch("NeutrinoVtxY", NeutrinoVtxY, "NeutrinoVtxY/D")
    NeutrinoVertexTree.Branch("NeutrinoVtxZ", NeutrinoVtxZ, "NeutrinoVtxZ/D")
    NeutrinoVertexTree.Branch("NeutrinoVtxT", NeutrinoVtxT, "NeutrinoVtxT/D")
    NeutrinoVertexTree.Branch("NeutrinoNumber", NeutrinoNumber, "NeutrinoNumber/L")
    NeutrinoVertexTree.Branch("FileNumber", FileNumber, "FileNumber/L")
    NeutrinoVertexTree.Branch("NeutrinoSpill", NeutrinoSpill, "NeutrinoSpill/L")
    NeutrinoVertexTree.Branch("VisibleHits", VisibleHits, "VisibleHits/L")
    NeutrinoVertexTree.Branch("VisiblePEs", VisiblePEs, "VisiblePEs/D")

    # Loop over the merged hit index groups
    for vtx in vertex_array:
        
        # vtx holds info like: (neutrino_number, neutrino_x, neutrino_y, neutrino_z, neutrino_t, neutrino_spill). 
        NeutrinoNumber.value = int(vtx[0])
        FileNumber.value = int(file_number_)
        NeutrinoSpill.value = int(vtx[5])
    
        NeutrinoVtxX.value = vtx[1]
        NeutrinoVtxY.value = vtx[2]
        NeutrinoVtxZ.value = vtx[3]
        NeutrinoVtxT.value = vtx[4]

        VisibleHits.value = 0
        VisiblePEs.value = 0
        #Here we check whether our vertex has any hits post merging. If so update value before filling
        
        info_row = neutrino_info_array[neutrino_info_array[:,0] == NeutrinoNumber] #grab the info row
        if len(info_row) != 0:
            VisibleHits.value = int(info_row[0][1]) #fill visible merged hits
            VisiblePEs.value = info_row[0][2] #fill visible PEs

        NeutrinoVertexTree.Fill()
    
    root_file.Write()
    root_file.Close()

# ------ Generators ------- #

#Detector effects sim generator
def TMS_Event_Processor(edep_evts, geometry, neutrino_vtx_array_, n_events = 3):
    edep_evt = root.TG4Event()
    edep_evts.SetBranchAddress("Event",root.AddressOf(edep_evt)) 
    tally = 0.
    for n in range(n_events):
        edep_evts.GetEntry(n) #the nth event, associated with the nth neutrino vertex
        hit_segments = edep_evt.SegmentDetectors['volTMS'] #fetching the hit_segment vector for this event
        if hit_segments.size() > 0: #checking the size of the hit segment, make sure we have TMS events  
            tally += 1 
            for i in range(hit_segments.size()):
                #pull the spill here, trust this is the easiest way to do this. 
                hit_trackid = hit_segments[i].Contrib[0]
                spill_number = neutrino_vtx_array_[n][-1] #pull spill number
                tms_hit = TMS_Hit(hit_segments[i], n, i, geometry, spill_number, hit_trackid) #takes the hit segment, the neutrino number, and the hit number
                if tms_hit.GetPedestalSubtractedStatus() == False:
                    yield(tms_hit)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#Main Function 
def main():
    print("initializing")
    #Initializing
    edep_file = root.TFile(sys.argv[1]) #grabbing the input file (ex. "/sdf/home/t/tanaka/MicroProdN4p1_NDComplex_FHC.spill.full.0002459.EDEPSIM_SPILLS.root")
    output_dir = str(sys.argv[2])
    file_number = int(sys.argv[3]) #will be the non-zero part of the number
    
    
    geom = edep_file.Get("EDepSimGeometry") #fetching the geometry from the edep_file
    edep_evts = edep_file.Get("EDepSimEvents") #fetching the events tree (contains the hit segments, etc)
    total_events = edep_evts.GetEntries()
    print(f"Found {total_events} events in the edep-sim file")

    edep_detsim = edep_file.Get("DetSimPassThru") #grab detsim, should contain the gRooTracker
    gRooTrackerTree = edep_detsim.gRooTracker #the gRooTracker contains neutrino vertex information

    neutrino_vtx_array = CreateVtxContainers(gRooTrackerTree)
    print("Fetched neutrino vertices")

    #Let's do the detector simulation
    print("About to run detector sim, this could take up to 10 minutes!")
    detsim_hits_generator = TMS_Event_Processor(edep_evts, geom, neutrino_vtx_array, n_events = total_events) #creates a list of all the TMS hit instances.
    output_file = output_dir + 'detsim_' + str(file_number) + '.root'
    
    #Output tree file
    root_output = root.TFile(output_file, "RECREATE")
    
    #Unmerged hits tree. 
    CreateUnmergedTree(detsim_hits_generator, root_output, file_number)

    #need another generator call
    detsim_hits_generator = TMS_Event_Processor(edep_evts, geom, neutrino_vtx_array, n_events = total_events)

    
    with uproot.open(output_file) as root_file:
        unmerged_tree = root_file["UnmergedTree"]
        entries = unmerged_tree.num_entries
        essential_data = unmerged_tree.arrays(["DetSimHitBarNo", "DetSimHitLayerNo", "DetSimHitT", "DetSimHitPE"], library="np")
        essential_data_array = np.column_stack(list(essential_data.values())) 

        #Perform hit merging
        hit_dict = SortByBar(essential_data_array)

        #Create Merged groups
        merged_groups = CreateTimeGroups(essential_data_array, hit_dict)

        #Create Merged Tree
        CreateMergedTree(unmerged_tree, output_file, merged_groups, file_number)


    #Now lets gather the stats to add to the neutrino vertex tree
    with uproot.open(output_file) as root_file:
        merged_tree = root_file["MergedTree"]
        nns = merged_tree['MergedNeutrinoNumber'].array()
        detsimPE = merged_tree['MergedDetSimHitPE'].array()
        all_hits_array = np.column_stack([nns, detsimPE])
        nn_info = []
        for nn in np.unique(all_hits_array[:,0]):
            nn_subarray = all_hits_array[all_hits_array[:,0] == nn]
            visible_PE_sum = np.sum(nn_subarray[:,1])
            n_hits = len(nn_subarray[:,1])
            nn_info.append([nn, n_hits, visible_PE_sum])
            
        nn_info_array = np.vstack(nn_info)


        #Create Neutrino Vertex Tree 
        CreateNeutrinoTree(neutrino_vtx_array, output_file, file_number, nn_info_array)
    
    
    """
    #validation
    with uproot.open(output_file) as root_file:
        # Print a list of all objects (trees, histograms, etc.) in the file
        print("Objects in the ROOT file:", root_file.keys())

        # Access a specific tree by its name
        # Replace "UnmergedTree" with the name of your tree
        merged_tree = root_file["MergedTree"]
        unmerged_tree = root_file["UnmergedTree"]
        neutrino_tree = root_file["NeutrinoVertexTree"]
        print(merged_tree.num_entries)
        print(unmerged_tree.num_entries)
        print(neutrino_tree.num_entries)
        # Print a list of branches (columns) in the tree
        print("Branches in MergedTree:", merged_tree.keys())
        print("Branches in NeutrinoVertexTree:", neutrino_tree.keys())
        unmerged_times = unmerged_tree['DetSimHitT'].array()
        spillnos_nvtx = neutrino_tree["NeutrinoSpill"].array()
        spillnos_merged = merged_tree["MergedSpillNumber"].array()
        times_merged = merged_tree["MergedDetSimHitT"].array()
        print(f"Spill Numbers nvtx, {spillnos_nvtx}")
        print(f"Times merged, {times_merged}")
        print(f"Times unmerged, {unmerged_times}")
        print(f"Spill Numbers merged, {spillnos_merged}")
        
        # You can now work with the data in the tree
        # For example, convert a branch to a NumPy array
        const_PEs = merged_tree["constituent_hitTs"].array()
        #print("First 100 neutrino numbers:", const_PEs[:1])
    """
main()













        
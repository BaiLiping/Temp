"""
%% ----------------------------------- Poisson Multi-Bernoulli Mixture(PMBM) filter ------------------------------ %%
This Python code is reproduction for the "point target PMBM filter" originally proposed in paper [1]. 
The original Matlab code for "point target PMBM filter" could be available from authors page:
https://github.com/Agarciafernandez/MTT
Corresponding video explains MBM, PMBM, TPMBM, TPMB in detail can be seen: https://www.youtube.com/playlist?list=PLadnyz93xCLjl51PzSoFhLLSp2hAYDY0H
%% ----------------------------------- Reference Papers ------------------------------------------ %%
% [1] A. F. García-Fernández, J. L. Williams, K. Granström, and L. Svensson, “Poisson multi-Bernoulli mixture filter: direct 
        derivation and implementation”, IEEE Transactions on Aerospace and Electronic Systems, 2018.
  [2] 2006. B.-N. Vo, W.-K. Ma, "The Gaussian Mixture Probability Hypothesis Density Filter", IEEE Transactions on Signal
        Processing
%% ----------------------------------- Data Structure  ------------------------------------------ %%
Beware filter_pred and filter_upd are two dictionary with the following fields(Beware filter_pred has the same items as filter_upd, 
thus here we only show the examples with filter_upd):
Poisson Components:
    filter_upd['weightPois']:
        weights of the Poisson point process (PPP). It is a list with N_miss_detected_targets elements, each element is a scalar value.
    filter_upd['meanPois']:
        means of the Gaussian components of the PPP. It is a list with N_miss_detected_targets elements, each element is a vector with size (4, 1).
    filter_upd['covPois']:
        covariances of the Gaussian components of the PPP. It is a list with N_miss_detected_targets elements, each element is a matrix with size (4, 4).
MBM Components:
    filter_upd['globHyp']:
        the "matrix"(actually it is a list of list) whose number of rows is the number of global hypotheses and the number of columns is the number of 
        Bernoulli components(Note: the number of Bernoulli components here equal to "number of surviving track which were detected previously + the 
        number of new measurements", the corresponding details will be explained at "step 2: update" in the code.). Each element in a particular row 
        indicates the index of the single target hypothesis for each Bernoulli component in the particular global hypothesis. It is zero if this global 
        hypothesis does not have a particular Bernoulli component.
    filter_upd['globHypWeight']:
        the list of N_GlobalHypothese elements, each element is a scalar which is the weight of a particular global hypothesis.
    filter_upd['tracks']:
        a list of N_BernoulliComponents elements, each element is a Bernoulli component(and each Bernoulli component is actually a dictionary which contains following items).
        filter_upd['tracks'][i]:
            a dictionary contains several corresponding information of the ith Bernoulli component.
        filter_upd['tracks'][i]['t_ini']:
            a scalar value which stands for the time of birth of i-th Bernoulli component.
        filter_upd['tracks'][i]['meanB']:
            a list contains means of several Gaussian components corresponding to i-th Bernoulli component, each Gaussian componenet stands for each single target hypothesis corresponding to i-th Bernoulli component.
            e.g. filter_upd['tracks'][i]['meanB'][j] contains the mean value of j-th Gaussian component(j-th single target hypothesis) corresponding to i-th Bernoulli component.
        filter_upd['tracks'][i]['covB']:
            a list contains covariance matrices of several Gaussian components corresponding to i-th Bernoulli component, each Gaussian componenet stands for each single target hypothesis corresponding to i-th Bernoulli component.
            e.g. filter_upd['tracks'][i]['covB'][j] contains the covariance matrix of j-th Gaussian component(j-th single target hypothesis) corresponding to i-th Bernoulli component.
        filter_upd['tracks'][i]['eB']:
            a list contains existence probabilities of all Gaussian components(single target hypotheses) correponding to the i-th Bernoulli component.
            e.g. filter_upd['tracks'][i]['eB'][j] is a scalar value which stands for existence probability of j-th Gaussian component(j-th single target hypothesis) corresponding to i-th Bernoulli component.
        filter_upd['tracks'][i]['aHis']:
            a list contains the history information of data associations (indices to measurements or 0 if undetected) for the all Gaussian components(all single target hypotheses) corresponding to i-th Bernoulli component.
            e.g. filter_upd['tracks'][i]['aHis'][j] is a list which contains the history info(from the time of birth until current time stamp, one time stamp data association information as one scalar element of this list) 
        filter_upd['tracks'][i]['weightBLog']:
            a list contains the log weight information of data associations (indices to measurements or 0 if undetected) for the all Gaussian components(all single target hypotheses) corresponding to i-th Bernoulli component.
            e.g. filter_upd['tracks'][i]['weightBLog'][j] is a list which contains the log weight info(from the time of birth until current time stamp, one time stamp data association information as one scalar element of this list) 
"""
"""
The current version of this code, is updated in 20210808.
"""

import numpy as np
import copy
import math
from murty import Murty
from util import mvnpdf, CardinalityMB
from functools import reduce
import operator
from itertools import chain

"""
PMBM Point Target Filter.
Beware in PMBM point target filter, there are basically two types of components:
1. Poisson Point Process/PPP component. This component is used to model "new birth targets" and "existing/surviving miss-detected targets".
2. Multi-Bernoulli Mixture/MBM component. This component is used to model "existing/surviving detected targets".
"""

class PMBM_Filter:
    def __init__(self, model, bayesian_filter_type, motion_model_type):
        self.model = model # use generated model which is configured for all parameters used in PMBM filter model for tracking the multi-targets.
        self.bayesian_filter_type = bayesian_filter_type # Bayesian filter type, i.e. Kalman, EKF, Particle filter
        self.motion_model_type = motion_model_type # Motion model type, i.e. Constant Velocity(CV), Constant Accelaration(CA), Constant Turning(CT), Interacting Multiple Motion Model(IMM)

    """
    Step 1: Prediction. Section V-B of [1]
    For prediction step, there are three parts:
    1.1. Prediction for (existing/surviving)previously miss-detected targets(using Poisson Point Process/PPP to model).
        -- Beware Poisson Point Process/PPP is described by intensity lambda.
    1.2. Prediction for new birth targets(using Poisson Point Process/PPP to model), it will be incorporated into "prediction for miss-detected targets PPP component".
    1.3. Prediction for existing/surviving previously detected targets (Multi-Bernoulli Mixture/MBM to model).
    """
    def predict(self, filter_pruned):
        # Get pre-defined parameters.
        F = self.model['F_k']   # Transition matrix, F.
        Q = self.model['Q_k']   # Process noise, Q.
        Ps = self.model['p_S']  # Probability of target survival, Ps.
        number_of_new_birth_targets = self.model['number_of_new_birth_targets']

        # Get components information from filter_pruned.
        number_of_surviving_previously_miss_detected_targets = len(filter_pruned['weightPois'])
        number_of_surviving_previously_detected_targets=len(filter_pruned['tracks'])
        
        # Initate data structure for predicted step.
        # Data structure for Poisson Components.
        filter_predicted = {}
        filter_predicted['weightPois']=[]
        filter_predicted['meanPois']=[]
        filter_predicted['covPois']=[]
        # Data structure for Bernoulli Components.
        if number_of_surviving_previously_detected_targets > 0:
            filter_predicted['tracks'] = [{} for i in range(number_of_surviving_previously_detected_targets)]
            # For each track of surviving previously detected target, initiate the dictionary with the right keys and an empty list
            for previously_detected_target_index in range(number_of_surviving_previously_detected_targets):
                filter_predicted['tracks'][previously_detected_target_index]['meanB']=[]
                filter_predicted['tracks'][previously_detected_target_index]['covB']=[]
                filter_predicted['tracks'][previously_detected_target_index]['eB']=[]
                filter_predicted['tracks'][previously_detected_target_index]['aHis']=[]
                filter_predicted['tracks'][previously_detected_target_index]['t_ini']=copy.deepcopy(filter_pruned['tracks'][previously_detected_target_index]['t_ini'])
            # global Hypptheses & global Hypotheses Weight remain unchanged through the prediction step
            filter_predicted['globHyp'] = copy.deepcopy(filter_pruned['globHyp'])
            filter_predicted['globHypWeight'] = copy.deepcopy(filter_pruned['globHypWeight'])
        else:
            filter_predicted['tracks'] = []
            filter_predicted['globHyp'] = []
            filter_predicted['globHypWeight'] = []
        """
        Step 1.1 : Prediction for surviving previously miss detected targets(i.e. the targets were undetected at previous frame and survive into current frame) by using PPP.
        """        
        # Compute where it would have been should this track have been detected in previous step.
        if number_of_surviving_previously_miss_detected_targets > 0:
            for PPP_component_index in range(number_of_surviving_previously_miss_detected_targets):
                filter_predicted['weightPois'].append(Ps * filter_pruned['weightPois'][PPP_component_index])       # Equation (25) in [2].
                filter_predicted['meanPois'].append(F.dot(filter_pruned['meanPois'][PPP_component_index])) # Equation (26) in [2] Calculate means of the Gaussian components of miss detected targets PPP.
                filter_predicted['covPois'].append(Q + F.dot(filter_pruned['covPois'][PPP_component_index]).dot(np.transpose(F)))   # Equation (27) in [1] Calculate covariance of the Gaussian component of each miss detected target PPP. 

        """
        Step 1.2 : Prediction for new birth targets by using PPP.
        """
        # Only generate new birth target if there are no existing miss detected targets
        # Incorporate New Birth intensity into PPP. 
        anchor_birth_position = [self.model['x_new_birth'][0],self.model['x_new_birth'][1]] 
        for new_birth_target_index in range(number_of_new_birth_targets):
            filter_predicted['weightPois'].append(self.model['w_birthsum']/number_of_new_birth_targets)  # Create the weight of PPP using the weight of the new birth PPP
            filter_predicted['meanPois'].append([[anchor_birth_position[0][0]+10*new_birth_target_index], [anchor_birth_position[1][0]-10*new_birth_target_index],[self.model['x_new_birth'][2][0]],[self.model['x_new_birth'][3][0]]])   # Create the mean of PPP using the mean of the new birth PPP
            filter_predicted['covPois'].append(self.model['P_new_birth'])    # Create the variance of PPP using the variance of the new birth PPP
        """
        Step 1.3 : Prediction for existing/surviving previously detected targets(i.e. targets were detected at previous frame and survive into current frame) by using Bernoulli components, or so called Multi-Bernoulli RFS.
        """
        if number_of_surviving_previously_detected_targets > 0:
            for previously_detected_target_index in range(number_of_surviving_previously_detected_targets):
                # Get the number of single target hypotheses of the previously_detected_target_index-th Bernoulli component(single target).
                number_of_single_target_hypothesis_from_previous_step = len(filter_pruned['tracks'][previously_detected_target_index]['eB'])
                # Get time of birth of this Bernoulli component (single target).
                filter_predicted['tracks'][previously_detected_target_index]['t_ini'] = filter_pruned['tracks'][previously_detected_target_index]['t_ini']
                # We go through all single target hypotheses:
                for single_target_hypothesis_index in range(number_of_single_target_hypothesis_from_previous_step):
                    filter_predicted['tracks'][previously_detected_target_index]['meanB'].append(F.dot(filter_pruned['tracks'][previously_detected_target_index]['meanB'][single_target_hypothesis_index])) # Calculate mean of this guassian distribution
                    filter_predicted['tracks'][previously_detected_target_index]['covB'].append(F.dot(filter_pruned['tracks'][previously_detected_target_index]['covB'][single_target_hypothesis_index]).dot(np.transpose(F)) + Q) # Calculate variance of this guassian distribution
                    filter_predicted['tracks'][previously_detected_target_index]['eB'].append(Ps * filter_pruned['tracks'][previously_detected_target_index]['eB'][single_target_hypothesis_index]) # Calculate existence probability of this guassian distribution
                    filter_predicted['tracks'][previously_detected_target_index]['aHis'].append([filter_pruned['tracks'][previously_detected_target_index]['aHis'][single_target_hypothesis_index]]) # Register the association history of this guassian distribution

        return filter_predicted

    def predict_initial_step(self):
        """
        Compute the predicted intensity of new birth targets for the initial step (first frame).
        """
        # Create an empty dictionary filter_predicted which will be filled in by calculation and output from this function.
        filter_predicted = {}
        filter_predicted['weightPois']=[]
        filter_predicted['meanPois']=[]
        filter_predicted['covPois']=[]
        filter_predicted['tracks'] = []
        filter_predicted['globHyp'] = []
        filter_predicted['globHypWeight'] = []
        # Get the number of new birth targets.
        number_of_new_birth_targets_init = self.model['number_of_new_birth_targets_init']
        # Only birth for the initial step. Notice the birth is w_birthinit instead of w_birthsum. 
        # Beware we only output none-Bernoulli components(e.g. PPP) in predict_initial_step, as there is not any surviving detected targets Multi-Bernoulli RFS at beginning.
        anchor_birth_position = [self.model['x_new_birth'][0],self.model['x_new_birth'][1]] 
        for new_birth_target_index in range(number_of_new_birth_targets_init):
            filter_predicted['weightPois'].append(self.model['w_birthsuminit']/number_of_new_birth_targets_init)  # Create the weight of PPP using the weight of the new birth PPP
            birth = [[anchor_birth_position[0][0]+10*new_birth_target_index], 
                     [anchor_birth_position[1][0]-10*new_birth_target_index],
                     [self.model['x_new_birth'][2][0]],
                     [self.model['x_new_birth'][3]][0]]
            filter_predicted['meanPois'].append(birth)   # Create the mean of PPP using the mean of the new birth PPP
            filter_predicted['covPois'].append(self.model['P_new_birth'])    # Create the variance of PPP using the variance of the new birth PPP
        return filter_predicted

    """
    Step 2: Update
    2.1. For the previously miss detected targets and new birth targets(both represented by PPP) which are still undetected at current frame, just update the weight of PPP but mean 
            and covarince remains same.
    2.2.1. For the previously miss detected targets and new birth targets(both represented by PPP) which are now associated with detections(detected) at current frame, corresponding 
            Bernoulli RFS is converted from PPP normally by updating (PPP --> Bernoulli) for each of them.
    2.2.2. For the measurements(detections) which can not be in the gating area of any previously miss detected target or any new birth target(both represented by PPP), corresponding 
            Bernoulli RFS is created by filling most of the parameters of this Bernoulli as zeors (create Bernoulli with zero existence probability, stands for detection is originated 
            from clutter) for each of them.
    2.3.1. For the previously detected targets which are now undetected at current frame, just update the eB of the distribution but mean and covarince remains same for each of them.
    2.3.2. For the previously detected targets which are now associated with detection(detected) at current frame, the parameters of the distribution is updated for each of them.
    """
    def update(self, Z_k, filter_predicted, nth_scan):
        # Get pre-defined parameters.
        p_D =self.model['p_D']
        R_k = self.model['R_k']
        H_k = self.model['H_k']
        gating_threshold = self.model['gating_threshold']
        clutter_intesntiy = self.model['clutter_intensity']

        # Get components information from filter_predicted.
        number_of_previously_miss_detected_targets_and_new_birth_targets = len(filter_predicted['weightPois'])
        number_of_previously_detected_targets = len(filter_predicted['tracks'])
        number_of_global_hypothesis = len(filter_predicted['globHyp'])
        number_of_measurements_at_current_frame = len(Z_k)
        # At the extreme case, all the measurements could be originated from previously miss-detected targets, new birth targets and clutter only, and none 
        # of the measurements originated from previously detected targets(i.e. in such extreme case, all the previously detected targets are miss-detected 
        # at current frame.)
        number_of_previously_undetected_targets_and_new_birth_targets_plus_number_of_clutters = number_of_measurements_at_current_frame
        number_of_potential_detected_targets_at_current_frame_after_update = number_of_previously_detected_targets + number_of_previously_undetected_targets_and_new_birth_targets_plus_number_of_clutters

        # Initialize data structures for filter_update
        filter_updated = {}
        filter_updated['weightPois'] = []
        filter_updated['meanPois'] = []
        filter_updated['covPois'] = []

        if number_of_previously_detected_targets==0:
            filter_updated['globHyp']=[[int(x) for x in np.zeros(number_of_measurements_at_current_frame)]] # The global hypothesis is all associated with missed detection
            filter_updated['globHypWeight']=[1] # There would be one global hypothesis, each measurement is associated with itself.
            if number_of_measurements_at_current_frame == 0:
                filter_updated['tracks'] = [] 
            else: 
                filter_updated['tracks']=[{} for n in range(number_of_measurements_at_current_frame)] # Initiate the data structure with right size of dictionaries
                for i in range(number_of_measurements_at_current_frame): # Initialte the dictionary with empty list.
                    filter_updated['tracks'][i]['eB']= []
                    filter_updated['tracks'][i]['covB']= []
                    filter_updated['tracks'][i]['meanB']= []
                    filter_updated['tracks'][i]['weightBLog']= []
                    filter_updated['tracks'][i]['aHis']= []
        else:
            filter_updated['globHyp'] = []
            filter_updated['globHypWeight'] = []
            if number_of_measurements_at_current_frame == 0:
                filter_updated['tracks'] = [] 
            else:
                filter_updated['tracks']=[{} for n in range(number_of_potential_detected_targets_at_current_frame_after_update)] # Initiate the data structure with right size of dictionaries
                # Initiate data structure for indexing 0 to number of detected target index
                for previously_detected_target_index in range(number_of_previously_detected_targets):
                    number_of_single_target_hypothesis_from_previous_step = len(filter_predicted['tracks'][previously_detected_target_index]['eB'])
                    filter_updated['tracks'][previously_detected_target_index]['t_ini'] = filter_predicted['tracks'][previously_detected_target_index]['t_ini']
                    # The intent here is to register a matrix with a vector. Alternatively, you can use a matrix for this or just use vector cat.
                    number_of_new_single_target_hypotheses_after_update = number_of_single_target_hypothesis_from_previous_step*(number_of_measurements_at_current_frame+1) # need to leave room for m_0 which is missed detection
                    # Initiated Data Structure
                    # Notice the length of the list is number_of_new_single_target_hypotheses_after_update, which is the number of previous targets plus measurements of this frame.
                    # This is because a list is used instead of a matrix. The indexing issue would be confusing later in the code.
                    # alternative, we can initiate the data structure as a matrix as below, this alternative format would make things easy and less confusing, particularly in the later part of the code
                    '''
                    filter_updated['tracks'][previously_detected_target_index]['meanB'] = [[] for i in range(number_of_new_single_target_hypotheses_after_update)] 
                    filter_updated['tracks'][previously_detected_target_index]['covB'] = [[] for i in range(number_of_new_single_target_hypotheses_after_update)]
                    filter_updated['tracks'][previously_detected_target_index]['eB'] = np.zeros(number_of_new_single_target_hypotheses_after_update)
                    filter_updated['tracks'][previously_detected_target_index]['aHis'] = [[] for i in range(number_of_new_single_target_hypotheses_after_update)]   
                    filter_updated['tracks'][previously_detected_target_index]['weightBLog_k'] = [[] for i in range(number_of_new_single_target_hypotheses_after_update)]
                    '''
                    filter_updated['tracks'][previously_detected_target_index]['meanB'] = [] 
                    filter_updated['tracks'][previously_detected_target_index]['covB'] = []
                    filter_updated['tracks'][previously_detected_target_index]['eB'] = []
                    filter_updated['tracks'][previously_detected_target_index]['aHis'] = []   
                    filter_updated['tracks'][previously_detected_target_index]['weightBLog_k'] = []

                # Initializing data structure for index from number of previously detected targets to number of previosly detected targets + number of measuremetns                  
                for i in range(number_of_previously_undetected_targets_and_new_birth_targets_plus_number_of_clutters): # Initialte the dictionary with empty list.
                    filter_updated['tracks'][number_of_previously_detected_targets + i]['eB']= []
                    filter_updated['tracks'][number_of_previously_detected_targets + i]['covB']= []
                    filter_updated['tracks'][number_of_previously_detected_targets + i]['meanB']= []
                    filter_updated['tracks'][number_of_previously_detected_targets + i]['weightBLog']= []
                    filter_updated['tracks'][number_of_previously_detected_targets + i]['aHis']= []
        """
        Step 2.1. for update: 
        Update step for "the targets which were miss detected previosly and still remain undetected at current frame, and new birth targets got undetected
        at current frame. Remain PPP. 
        """
        # Miss detected target and new birth target are modelled by using Poisson Point Process(PPP). This is the same as the miss detected target modelling part in [2].
        # Notice the reason mean and covariance remain the same is because if there is no detection, there would be no update.
        for PPP_component_index in range(number_of_previously_miss_detected_targets_and_new_birth_targets):
            filter_updated['weightPois'].append((1-p_D)*filter_predicted['weightPois'][PPP_component_index])
            filter_updated['meanPois'].append(filter_predicted['meanPois'][PPP_component_index])
            filter_updated['covPois'].append(filter_predicted['covPois'][PPP_component_index])

        """
        Step 2.2. for update: Generate number_of_measurements_at_current_frame new Bernoulli components(Parts of new Bernoulli components are converted from PPP, others are 
                    created originally.). Section V-C1 of [1]
        2.2.1: Convert Poisson Point Processes to Bernoulli RFSs. Update the targets which were miss detected previosly but now get detected at current frame, by updating with 
                    the valid measurement within gating area.
        2.2.2: Create new Bernoulli RFSs. For the measurements not falling into gating area of any PPP component, it is assumed to be originated from clutter. Create a Bernoulli 
                    RSF by filling parameters with zeros for each of them anyway for data structure purpose.
        """
        for measurement_index in range(number_of_measurements_at_current_frame):    
            associated_tracks = []
            # Go through all Poisson components(previously miss-detected targets and new birth targets) and perform gating
            for PPP_component_index in range(number_of_previously_miss_detected_targets_and_new_birth_targets):
                mean_PPP_component = filter_predicted['meanPois'][PPP_component_index]
                cov_PPP_component = filter_predicted['covPois'][PPP_component_index]
                
                # Compute Kalman Filter Elements
                eta_pred_PPP_component = H_k.dot(mean_PPP_component).astype('float64')                     
                S_pred_PPP_component = R_k + H_k.dot(cov_PPP_component).dot(np.transpose(H_k)).astype('float64')
                Si = copy.deepcopy(S_pred_PPP_component)
                invSi = np.linalg.inv(np.array(Si, dtype=np.float64))
                K_pred_PPP_component = cov_PPP_component.dot(np.transpose(H_k)).dot(invSi).astype('float64')
                position_difference = [eta_pred_PPP_component[0] - Z_k[measurement_index][0],eta_pred_PPP_component[1] - Z_k[measurement_index][1]] 
                
                mean_updated_PPP_component = mean_PPP_component + K_pred_PPP_component.dot(position_difference) # it is a column vector with lenghth 4
                cov_updated_PPP_component = cov_PPP_component - K_pred_PPP_component.dot(H_k).dot(cov_PPP_component).astype('float64')

                mahananobis_distance_between_current_measurement_and_current_PPP_component = np.transpose(position_difference).dot(invSi).dot(position_difference)
                # Mark the current PPP component when current measurement fall into the gating area of current PPP component.
                if mahananobis_distance_between_current_measurement_and_current_PPP_component < gating_threshold:
                    associated_tracks.append(PPP_component_index)  
            '''
            2.2.1: If current measurements is associated with PPP component(previously miss-detected target or new birth target), use this measurement to update the target, 
                    thus convert corresponding PPP into Bernoulli RFS.
            '''
            if len(associated_tracks)>0: # If there are PPP components could be associated with current measurement.
                meanB_sum = np.zeros((len(self.model['H_k'][0]),1))
                covB_sum = np.zeros((len(self.model['H_k'][0]),len(self.model['H_k'][0])))
                weightB_sum = 0
                for associated_track_index in associated_tracks:
                    # Update the target by using current measurement, convert PPP into Bernoulli RFS(currently detected target is represented using Bernoulli).
                    weight_miss_detected_track = p_D*filter_predicted['weightPois'][associated_track_index]*mvnpdf(Z_k[measurement_index], eta_pred_PPP_component,S_pred_PPP_component)
                    weightB_sum += weight_miss_detected_track
                    meanB_sum += weight_miss_detected_track*(mean_updated_PPP_component)
                    covB_sum += weight_miss_detected_track*cov_updated_PPP_component + weight_miss_detected_track*(mean_updated_PPP_component.dot(np.transpose(mean_updated_PPP_component)))

                # Gaussian mixture reduction
                # If current measurement is associated with more than one targets(PPP components, now already converted to Bernoullu components), merging all the guassians.
                meanB = meanB_sum/weightB_sum
                covB = covB_sum/weightB_sum - (meanB*np.transpose(meanB))
                eB = weightB_sum/(weightB_sum+clutter_intesntiy) # Notice this is the same as the normalization step in PHD existence/existence+clutter
                
                # Fill in the data structure
                filter_updated['tracks'][number_of_previously_detected_targets + measurement_index]['meanB'].append(meanB)
                filter_updated['tracks'][number_of_previously_detected_targets + measurement_index]['covB'].append(covB)
                filter_updated['tracks'][number_of_previously_detected_targets + measurement_index]['t_ini']=nth_scan # the track is initiated at time nth_scan
                filter_updated['tracks'][number_of_previously_detected_targets + measurement_index]['aHis'].append(measurement_index) # register history 
                filter_updated['tracks'][number_of_previously_detected_targets + measurement_index]['weightBLog'].append(np.log(weightB_sum)) # weightB is used for cost matrix computation
                filter_updated['tracks'][number_of_previously_detected_targets + measurement_index]['eB'].append(eB) # Notice the meaning of of eB, is for cardinality computation, weightB is the hypothesis probability, which are two different things

            else: 
                '''
                2.2.2
                If there is not any PPP component(previously miss-detected target or new birth target) could be associated with current measurement, assume this measurement is originated from clutter. 
                We still need to create a Bernoulli component for it, since we need to guarantee that ever measurement generate a Bernoulli RFS.
                The created Bernoulli component has existence probability zero (denote it is clutter). It will be removed by pruning.
                '''
                weightB = clutter_intesntiy #This measurement is a clutter
                # in the global hypothesis generating part of the code, this option would be registered as h_0 this track does not exist
                filter_updated['tracks'][number_of_previously_detected_targets + measurement_index]['eB'].append(0)
                filter_updated['tracks'][number_of_previously_detected_targets + measurement_index]['weightBLog'].append(np.log(weightB))
                filter_updated['tracks'][number_of_previously_detected_targets + measurement_index]['meanB'].append(np.zeros((len(self.model['H_k'][0]),1))) #TODO figure out why zero
                filter_updated['tracks'][number_of_previously_detected_targets + measurement_index]['covB'].append(np.zeros((len(self.model['H_k'][0]), len(self.model['H_k'][0]))))
                filter_updated['tracks'][number_of_previously_detected_targets + measurement_index]['t_ini']=nth_scan # If it is a potential detection, then it would start from Nth scan.
                filter_updated['tracks'][number_of_previously_detected_targets + measurement_index]['aHis'].append(measurement_index)

        """
        Step 2.3. for update: Section V-C2 of [1]
        Update for targets which got detected at previous frame.
        """
        for previously_detected_target_index in range(number_of_previously_detected_targets):
            '''
            number_of_single_target_hypothesis_from_previous_step = len(filter_predicted['tracks'][previously_detected_target_index]['eB'])
            filter_updated['tracks'][previously_detected_target_index]['t_ini'] = filter_predicted['tracks'][previously_detected_target_index]['t_ini']
            # The intent here is to register a matrix with a vector. Alternatively, you can use a matrix for this or just use vector cat.
            number_of_new_single_target_hypotheses_after_update = number_of_single_target_hypothesis_from_previous_step*number_of_measurements_at_current_frame
            '''
            # Go through all single target hypotheses.
            for single_target_hypothesis_index in range(number_of_single_target_hypothesis_from_previous_step):
                # Get the data from filter_predicted
                mean_single_target_hypothesis = filter_predicted['tracks'][previously_detected_target_index]['meanB'][single_target_hypothesis_index]
                cov_single_target_hypothesis = filter_predicted['tracks'][previously_detected_target_index]['covB'][single_target_hypothesis_index]
                eB_single_target_hypothesis = filter_predicted['tracks'][previously_detected_target_index]['eB'][single_target_hypothesis_index]
                aHis_single_target_hypothesis = filter_predicted['tracks'][previously_detected_target_index]['aHis'][single_target_hypothesis_index]
                # weightBLog_k_single_target_hypothesis = filter_predicted['tracks'][previously_detected_target_index]['weightBLog_k'][single_target_hypothesis_index]
                t_ini_single_target_hypothesis = filter_predicted['tracks'][previously_detected_target_index]['t_ini']
                filter_updated['tracks'][previously_detected_target_index]['t_ini']= t_ini_single_target_hypothesis
                
                # Compute Kalman Filter Elements
                S_pred_single_target_hypothesis = R_k + H_k.dot(cov_single_target_hypothesis).dot(np.transpose(H_k)).astype('float64')
                eta_pred_single_target_hypothesis = H_k.dot(mean_single_target_hypothesis).astype('float64')
                '''
                alternatively, seems that inverse can be computed this way. 
                Si = copy.deepcopy(S_pred_single_target_hypothesis)
                invSi = np.linalg.inv(np.array(Si, dtype=np.float64))
                '''
                Vs = np.linalg.cholesky(S_pred_single_target_hypothesis) 
                log_det_S_pred_single_target_hypothesis = 2*np.log(reduce(operator.mul, np.diag(Vs), 1)) 
                inv_sqrt_S = np.linalg.inv(Vs) 
                invSi = inv_sqrt_S.dot(np.transpose(inv_sqrt_S))
                K_pred_single_target_hypothesis = cov_single_target_hypothesis.dot(np.transpose(H_k)).dot(invSi).astype('float64')
                
                mean_updated_single_target_hypothesis = mean_single_target_hypothesis + K_pred_single_target_hypothesis.dot(position_difference) # it is a column vector with lenghth 4
                cov_updated_single_target_hypothesis = cov_single_target_hypothesis - K_pred_single_target_hypothesis.dot(H_k).dot(cov_single_target_hypothesis).astype('float64')
                """
                Step 2.3.1. for update:
                Update the targets got detected previously but get undetected at current frame.
                """
                eB_undetected = eB_single_target_hypothesis*(1-p_D)/(1-eB_single_target_hypothesis+eB_single_target_hypothesis*(1-p_D))  # the existence probability of this track
                # this is according to page 10 of [1] notice that it does not exactly follow the equation.
                # it should be:
                # weightBLog_k = np.log(weightBLog_K_single_target_hypothesis*(1-eB_single_target_hypothesis+eB_single_target_hypothesis*(1-p_D))
                weightBLog_k = np.log(1-eB_single_target_hypothesis+eB_single_target_hypothesis*(1-p_D)) # does not exist plus exist but not detected, how likely is this hypothesis
                new_single_target_hypothesis_index = single_target_hypothesis_index + 0 * number_of_single_target_hypothesis_from_previous_step # naming things this way to make it more consistant
                # The first element of the data structure registers missed detection scenario.
                # Because this target has no measurement associated with it, there would be no update. Everything remain the same.
                filter_updated['tracks'][previously_detected_target_index]['meanB'].append(mean_single_target_hypothesis)
                filter_updated['tracks'][previously_detected_target_index]['covB'].append(cov_single_target_hypothesis)
                # Existence probability of "undetected target which is detected previously" = "The probability such target existed in the previous frame,
                # survival at this frame but undetected" / ("The probability such target existed in the previous frame, survival at this frame but undetected" 
                # + "The probability such target doesn't survive at this frame".)
                filter_updated['tracks'][previously_detected_target_index]['eB'].append(eB_undetected)
                filter_updated['tracks'][previously_detected_target_index]['aHis'].append(aHis_single_target_hypothesis.append(0)) # 0 means missed detection
                filter_updated['tracks'][previously_detected_target_index]['weightBLog_k'].append(weightBLog_k)
                
                """
                Step 2.3.2. for update:
                Update the targets got detected previously and still get detected at current frame.
                Beware what we do here is to update all the possible single target hypotheses and corresponding cost value for every single target hypothesis(each 
                target-measurement possible association pair). The single target hypothese which can happen at the same time will form a global hypothesis(joint 
                event), and all the global hypotheses will be formed exhaustively later by using part of "all the possible single target hypotheses". 
                """
                # Go through all the measurements.
                # Notice that the length of filter_updated['tracks'][previously_detected_target_index]['meanB'] would become number_of_single_target_hypothesis_from_previous_step*number_of_measurements_at_current_frame, since we
                # will update every single target hypothesis by every measurement exhaustively.
                for measurement_index in range(number_of_measurements_at_current_frame+1)[1:]: # starting from m_1, since m_0 means missed detection
                    # The following is a conversion from matrix form into vector form. [single_target_hypothesis, number_of_measurements]
                    # Notice this is a very convoluted way to go about things, could have been a lot easier and can be a point of confusion.
                    # the first column is m_0 which indicated missed detection
                    # the following rows are accessed via new_single_target_hypothesis_index
                    new_single_target_hypothesis_index = single_target_hypothesis_index + number_of_single_target_hypothesis_from_previous_step*measurement_index
                    position_difference = [eta_pred_single_target_hypothesis[0] - Z_k[measurement_index-1][0], eta_pred_single_target_hypothesis[1] - Z_k[measurement_index-1][1]] 
                    mahananobis_distance_between_current_surviving_previously_detected_target_under_previous_single_target_hypothesis_and_current_measurement = np.transpose(position_difference).dot(invSi).dot(position_difference)
                    if mahananobis_distance_between_current_surviving_previously_detected_target_under_previous_single_target_hypothesis_and_current_measurement < gating_threshold: 
                        # If this measurement can be associated to current "previously detected target under previous single target hypothesis", use this measurement to perform Kalman Filter update of this target.
                        filter_updated['tracks'][previously_detected_target_index]['meanB'].append(mean_updated_single_target_hypothesis)
                        filter_updated['tracks'][previously_detected_target_index]['covB'].append(cov_updated_single_target_hypothesis)
                        filter_updated['tracks'][previously_detected_target_index]['eB'].append(1) # Notice this is a MBM_01 implementation of the code. The reason Gasia choose this is because his estimator 2 need cardinality estimation. eB is for calculating cardinality, only integer is acceptable.
                        filter_updated['tracks'][previously_detected_target_index]['aHis'].append(aHis_single_target_hypothesis.append(measurement_index)) # Add current measurement to association history
                        filter_updated['tracks'][previously_detected_target_index]['weightBLog_k'].append(np.log(eB_single_target_hypothesis*p_D)+(-0.5*mahananobis_distance_between_current_surviving_previously_detected_target_under_previous_single_target_hypothesis_and_current_measurement )-1/2*log_det_S_pred_single_target_hypothesis-len(H_k)*np.log(2*math.pi)/2) # weightBLog_k is the cost value for current single target hypothesis(pair of current measurement and current target).
                        # according to equation in page 10 of [1] this is simply w_ij*e_ij*p_ig*mvnpdf(measurement, track)
                        # another way for this is the following: 
                        #filter_updated['tracks'][previously_detected_target_index]['weightBLog_k'][new_single_target_hypothesis_index]=np.log(weightBLog_k_for_single_target_hypothesis * eB_single_target_hypothesis*p_D*mvnpdf(z_k[measurement_index], mean_updated_single_target_hypothesis, cov_updated_single_target_hypothesis))
                    else: # Set the cost value as -inf for current single target hypothesis, if the current measurement in the gating area of current target.
                        filter_updated['tracks'][previously_detected_target_index]['meanB'].append(np.zeros((len(self.model['H_k'][0]),1)))
                        filter_updated['tracks'][previously_detected_target_index]['covB'].append(np.zeros((len(self.model['H_k'][0]),len(self.model['H_k'][0]))))
                        filter_updated['tracks'][previously_detected_target_index]['eB'].append(0) # Notice this is a MBM_01 implementation of the code. The reason Gasia choose this is because his estimator 2 need cardinality estimation. eB is for calculating cardinality, only integer is acceptable.
                        filter_updated['tracks'][previously_detected_target_index]['aHis'].append([]) # Add current measurement to association history
                        filter_updated['tracks'][previously_detected_target_index]['weightBLog_k'].append(-np.inf)
        """
        Step 2.4. for update:
        Update Global Hypotheses as described in section V-C3 in [1].
        The objective of global hypotheses is to select k optimal single target hypotheses to propogate towards the next step.
        """

        if number_of_previously_detected_targets>0:

            globWeightLog=[]
            globHyp=[]
            '''
            This step is similar to the joint_association_event_matrix genration in JPDA. The objective is to
            exclude all double association, i.e. a measurement is associated with more than one targets since this
            clearly violates the single target assumption for multi-object tracking. Notice the presentation of this information is different to that of JPDA. 
            in JPDA, every association event is presented by a hot-hit matrix, where one means an established association, 
            0 means no association. Yet here in PMBM, instead of binary registration, the information in registered in a much more 
            succinct manner with its single target hypothesis indexing. Global hypothesis matrix contains the same inforamtion as that of the list of association matrices in JPDA.
            filter_updated['tracks'] is a dictionary of all the possible single target associations for each target
            but in order to assemble the new global hypotheses based on those single target hypotheses, new global constraints are applied. 
            Specifically, the global constraints is the following:
            global hypotheses are a collection of these single-target hypotheses, with the conditions that 
            no measurement is left without being associated and a measurement can only be assigned to 
            one single target hypothesis. 
            The global hypotheses matrix and its corresponding global hypotheses weight matrix is of size:
            (number of global hypotheses, number of potential tracks after update)
            in this matrix, each element is the index of single target hypotheis. 
            For instance, h_0 means this track does not exist, h_1 means this track has one associated measurement which is itself.  
            the format of one specific global hypothesis could be: [2,1,3,1,0,1,2]. The numerical value represent the indexing of single target hypothesis
            Please note that the indexing is not unique. THIS CAN BE A POINT OF CONFUSION. the indexing of single target hypothesis is track specific, 
            therefore, without double association doesn't mean the global hypothesis is all unique indexing. 
            '''
            for global_hypothesis_index in range(number_of_global_hypothesis):
                '''
                Step 2.4.1 Generate Cost Matrix
                '''
                # Initiate a cost matrix for each global hypothesis.
                cost_matrix_log=-np.inf*np.ones((number_of_potential_detected_targets_at_current_frame_after_update, number_of_measurements_at_current_frame))
                cost_misdetection=np.zeros(number_of_previously_detected_targets)

                '''
                Step 2.4.1.1 Fill in cost_matrix with regard to the detected tracks.
                Gasia's original matlab code:
                for i=1:Nprev_tracks
                    index_hyp=filter_pred.globHyp(p,i); %Hypothesis for track i in p global hypothesis
                    Nhyp_i=length(filter_pred.tracks{i}.eB);
                    %We generate the cost matrix for measurements
                    
                    if(index_hyp~=0) % this track exist
                                      
                        index_max=length(filter_upd.tracks{i}.weightBLog_k);
                        indices=index_hyp+Nhyp_i*(1:size(z,2));
                        indices_c=indices<=index_max;
                        indices_c=indices(indices_c);
                        
                        weights_log=filter_upd.tracks{i}.weightBLog_k(indices_c);
                        
                        %We remove the weight of the misdetection hypothesis to use Murty (Later this weight is added).
                        cost_matrix_log(i,1:length(indices_c))=weights_log-filter_upd.tracks{i}.weightBLog_k(index_hyp);
                        cost_misdetection(i)=filter_upd.tracks{i}.weightBLog_k(index_hyp);
                    end
                end
                '''
                for previously_detected_target_index in range(number_of_previously_detected_targets):
                    single_target_hypothesis_index_specified_by_previous_step_global_hypothesis=filter_predicted['globHyp'][global_hypothesis_index][previously_detected_target_index] # Hypothesis for this track in global hypothesis i 
                    if single_target_hypothesis_index_specified_by_previous_step_global_hypothesis!=-1: # if this track exist      
                        # The following codes are specific to Gasia's implementation
                        # because he did not use a matrix for filter_update['tracks'] but only a matrix
                        # therefore, the indexing issue of his code can be a major code of confusion
                        # should he used matrix for the implementation, this part is simply taking the column vector.
                        # all he did here is take the column vector, but exclude the first element which is missed detection
                        """ index_max= len(filter_updated['tracks'][previously_detected_target_index]['weightBLog_k'])-1 """
                        # this step is the indexing issue throughout Gasia's code. He always represent matrix with vector, resulting in confusing indexing
                        # the indexs are the column vector are they are presented in the vector form. unrav.
                        # notice should it be in matrix format, the first element, which is missed detection hypothesis would be skipped
                        """ indices=[int(single_target_hypothesis_index_specified_by_previous_step_global_hypothesis+number_of_single_target_hypothesis_from_previous_step*x) for x in range(number_of_measurements_at_current_frame)] """
                        # get rid of the index(only one) that exceed max
                        """ indices_c=[idx for idx in indices if idx<=index_max] """
                        # take the row vector
                        """ indices_c=np.array(indices)[:indices_c]    """  
                        
                        # THE OBJECTIVE HERE IS TO TAKE THE COLUMN VECOT ASSOCIATED WITH h_k, TAKE OTHER THAN m_0
                        index_max= len(filter_updated['tracks'][previously_detected_target_index]['weightBLog_k'])-1
                        hypothesis_indices =[]
                        
                        for x in range(number_of_measurements_at_current_frame+1)[1:]:
                            index_converted_from_matrix_to_vector = int(single_target_hypothesis_index_specified_by_previous_step_global_hypothesis+number_of_single_target_hypothesis_from_previous_step*x)
                            if index_converted_from_matrix_to_vector<index_max: 
                                hypothesis_indices.append(index_converted_from_matrix_to_vector)
                        weight_log =[]
                        for idx in hypothesis_indices:
                            weight_log.append(filter_updated['tracks'][previously_detected_target_index]['weightBLog_k'][idx])                   
                        # We remove the weight of the misdetection hypothesis to use Murty (Later this weight is added). TODO what does he mean?
                        for idx in range(len(weight_log)):
                            # Currently this would result in NaN

                            # Note that we normalise the previous weights by ρj,i (∅) so that the weight of a hypothesis
                            # that does not assign a measurement to a target is the same for an old and a new target. This is just done so that we can
                            # obtain the k-best global hypotheses efficiently using Murty’s algorithm but we do not alter the real weights, which are
                            # unnormalised. Each new global hypothesis that origin

                            # cost_matrix_log need to remove the missed detection hypothesis
                            cost_matrix_log[previously_detected_target_index][idx] = weight_log[idx] - filter_updated['tracks'][previously_detected_target_index]['weightBLog_k'][int(single_target_hypothesis_index_specified_by_previous_step_global_hypothesis)]
                            """ cost_matrix_log(i,1:length(indices_c))=weights_log-filter_upd.tracks{i}.weightBLog_k(index_hyp); """
                        cost_misdetection[previously_detected_target_index]=filter_updated['tracks'][previously_detected_target_index]['weightBLog_k'][int(single_target_hypothesis_index_specified_by_previous_step_global_hypothesis)]
               
                '''
                Step 2.4.1.2 Fill in the cost matrix with regard to the newly initiated tracks
                Gasia's original code for this part
                %New targets
                for i=Nprev_tracks+1:Nnew_tracks+Nprev_tracks
                    weights_log=filter_upd.tracks{i}.weightBLog;
                    index=filter_upd.tracks{i}.aHis{1};
                    cost_matrix_log(i,index)=weights_log;
                end
                '''
                for potentially_detected_target_index in range(number_of_previously_undetected_targets_and_new_birth_targets_plus_number_of_clutters):
                    weight_log = filter_updated['tracks'][number_of_previously_detected_targets+potentially_detected_target_index]['weightBLog'][0]
                    associated_measurement_index=filter_updated['tracks'][number_of_previously_detected_targets+potentially_detected_target_index]['aHis'][0]
                    cost_matrix_log[number_of_previously_detected_targets+potentially_detected_target_index][associated_measurement_index]=weight_log
                       
                '''
                Step 2.4.2 Genereta the K (which varies each frame) optimal option based on cost matrix
                1. Remove -Inf rows and columns for performing optimal assignment. We take them into account for indexing later. Columns that have only one value different from Inf are not fed into Murty either.
                2. Use murky to get the kth optimal options
                3. Add the cost of the misdetection hypothesis to the cost matrix
                4. Add back the removed infinity options
                '''
                # generate the one hit matrix where only non-infinity elements are registered as 1
                # the dimmention of cost matrix (old_track+potential_new_tracks, measurements)
                cost_matrix_infinity_elements=np.isinf(cost_matrix_log) # if it is infinity, then true   
                cost_matrix_with_valid_elements = 1 - cost_matrix_infinity_elements
                # keep the columns whose sum is greater than 1, which indicate that there are more than one valid associations
                indices_of_measurements_with_more_than_one_associations = [x for x in range(len(cost_matrix_with_valid_elements[0])) if sum(cost_matrix_with_valid_elements[:,x])>1]
                # generate the cost matrix with only the indiced of columns to keep
                cost_matrix_reduced_to_measurements_with_more_than_one_associations = []
                if len(indices_of_measurements_with_more_than_one_associations)>0:
                    for valid_row_vector in cost_matrix_with_valid_elements:
                        new_row_vector = np.array(valid_row_vector)[indices_of_measurements_with_more_than_one_associations]
                        cost_matrix_reduced_to_measurements_with_more_than_one_associations.append(new_row_vector)
                    # keep the rows whos sum is greater than 0
                    indices_of_tracks_with_same_associated_measurements=[x for x in range(len(cost_matrix_reduced_to_measurements_with_more_than_one_associations)) if sum(cost_matrix_reduced_to_measurements_with_more_than_one_associations[x]>0)]
                else:
                    indices_of_tracks_with_same_associated_measurements=[]
                
                # if the column vector sums to one, then this column is to stay for certain
                indices_of_measurements_with_one_association = [x for x in range(len(cost_matrix_with_valid_elements[0])) if sum(cost_matrix_with_valid_elements[:,x])==1]
                if len(indices_of_measurements_with_one_association) > 0:
                    indices_of_tracks_with_one_association = [np.argmax(cost_matrix_with_valid_elements[:,x]) for x in indices_of_measurements_with_one_association]
                else:
                    indices_of_tracks_with_one_association = []
                # cost_for_tracks_with_single_measurement is the overall cost of the hypotheses that always belong to the output 
                cost_matrix_log_for_tracks_with_single_measurement_association = []
                if len(indices_of_tracks_with_one_association)>0:
                    for idx_of_indices in range(len(indices_of_tracks_with_one_association)):
                        row_vector = cost_matrix_log[indices_of_tracks_with_one_association[idx_of_indices]]
                        cost_log = row_vector[indices_of_measurements_with_one_association[idx_of_indices]]
                        cost_matrix_log_for_tracks_with_single_measurement_association.append(cost_log)
    
                    cost_for_tracks_with_single_measurement = sum(cost_matrix_log_for_tracks_with_single_measurement_association)
                else:
                    cost_for_tracks_with_single_measurement = 0 

                globWeightLog_predicted=np.log(filter_predicted['globHypWeight'][global_hypothesis_index])

                cost_matrix_log_for_tracks_with_same_associated_measurements = []
                global_weightLog_for_this_hypothesis =[]

                for track_idx in indices_of_tracks_with_same_associated_measurements:
                    cost_matrix_log_for_tracks_with_same_associated_measurements.append(cost_matrix_reduced_to_measurements_with_more_than_one_associations[track_idx])
                
                if len(cost_matrix_log_for_tracks_with_same_associated_measurements)==0:
                    optimal_associations_between_measurements_and_tracks_with_infinity_tracks_transposed = []
                    association_vector=np.zeros(number_of_measurements_at_current_frame)
                    for idx_of_indices in range(len(indices_of_measurements_with_one_association)):
                        association_vector[indices_of_measurements_with_one_association[idx_of_indices]]=indices_of_tracks_with_one_association[idx_of_indices]
                    
                    optimal_associations_between_measurements_and_tracks_with_infinity_tracks_transposed.append(association_vector)

                    global_weightLog_for_this_hypothesis.append(sum(cost_misdetection)+cost_for_tracks_with_single_measurement+globWeightLog_predicted) # The global weight associated with this hypothesis
              
                else:
                    # Number of new global hypotheses from this global hypothesis
                    # For global hypothesis j, whose weight is wj ∝ ∏n i=1 wj,i, we suggest choosing k = dNh ·wje,
                    # where it is assumed that we want a maximum number Nh of global hypotheses as in [21]. This way, global hypotheses
                    # with higher weights will give rise to more global hypotheses. 
                    
                    kbest=np.ceil(self.model['maximum_number_of_hypothesis']*filter_predicted['globHypWeight'][global_hypothesis_index])
              
                    # We run Murty algorithm to select only k best(beware the len(opt_indices) equals to k) global hypotheses among all global hypotheses. We call the function by using transpose and negative value
                    # murty is compiled cpp file. For usage example, please refer to:
                    # https://github.com/erikbohnsack/murty/blob/master/src/murty.hpp
                    # Murty would provide the optimal choices for each row, under the constraints that each column can only provide at most one element.
                    # cost_matrix has potential number of tracks row and number of measurements column. 
                    # trasnpose(cost_matrix) has number of measurements rows, and number of potential tracks column.
                    # the output would be a row vector, each row(measurements) is associated with one track
                    # ascending order of cost, therefore, the lower the cost the better it is.
                    cost_matrix_object_went_though_murty = Murty(-np.transpose(cost_matrix_log_for_tracks_with_same_associated_measurements))   
                    nlcost = []
                    optimal_associations_between_measurements_and_tracks = []

                    for ith_best_option in range(int(kbest)):
                        still_valid,ith_best_cost,ith_best_solution = cost_matrix_object_went_though_murty.draw()
                        if still_valid == True:
                            # each measurement is either associated with an old track or a new track
                            optimal_associations_between_measurements_and_tracks.append(ith_best_solution) # each solution is a list of row indices
                            nlcost.append(ith_best_cost) # should be of ascending order of cost
                        else:
                            break
                    '''
                    %Optimal indices without removing Inf rows
                    opt_indices_trans=Inf(size(opt_indices,1),size(z,2));           
                    for i=1:size(opt_indices,1)               
                        opt_indices_trans(i,indices_stay_column)=indices_stay_row(opt_indices(i,:));
                        
                      %We add the single trajectory hypotheses that belong to all k
                       %max
                       opt_indices_trans(i,fixed_indices_stay_column)=fixed_indices_stay_row; 
                    end
         
                    globWeightLog=[globWeightLog,-nlcost+sum(cost_misdetection)+cost_for_tracks_with_single_measurement+globWeightLog_pred];
                    '''
                    # Optimal indices without removing Inf rows
                    # NOTICE this part is completely unnecessary if we follow the recipe of their paper
                    # The infinity options was first removed and now here added back up again.
                    # The better option is to have have infinity options in the first place.
                    optimal_associations_between_measurements_and_tracks_with_infinity_tracks_transposed = np.inf*np.ones((len(optimal_associations_between_measurements_and_tracks),number_of_measurements_at_current_frame))           
                    for ith_best_option in range(len(optimal_associations_between_measurements_and_tracks_with_infinity_tracks_transposed)):
                        # First handle the case where there are duplicated associations
                        for idx in range(len(optimal_associations_between_measurements_and_tracks[ith_best_option])):
                            actual_track_index = indices_of_tracks_with_same_associated_measurements[optimal_associations_between_measurements_and_tracks[ith_best_option][idx]]
                            actual_measurement_idx = indices_of_measurements_with_more_than_one_associations[idx]
                            optimal_associations_between_measurements_and_tracks_with_infinity_tracks_transposed[ith_best_option][actual_measurement_idx]=actual_track_index
           
                        # Then handle the case wehre there are single association
                        for idx in range(len(indices_of_measurements_with_one_association)):
                            optimal_associations_between_measurements_and_tracks_with_infinity_tracks_transposed[ith_best_option][indices_of_measurements_with_one_association[idx]]=indices_of_tracks_with_one_association[idx] 
                    for cost in nlcost:
                        global_weightLog_for_this_hypothesis.append(-cost+sum(cost_misdetection)+cost_for_tracks_with_single_measurement+globWeightLog_predicted)
                if len(globWeightLog) == 0:
                    globWeightLog=global_weightLog_for_this_hypothesis
                else:
                    np.hstack((globWeightLog,global_weightLog_for_this_hypothesis))
                     
                '''
                Step 2.4.3 Generate the new global hypothesis based on the cost matrix of this previous global hypothesis
                '''
                
                # Initiate Global Hypoethesis data structure
                globHyp_under_previous_step_global_hypothesis=np.zeros((len(optimal_associations_between_measurements_and_tracks_with_infinity_tracks_transposed), number_of_potential_detected_targets_at_current_frame_after_update))
                '''
                please refer to figure 2 of [1]. The following part discribes track 1 and 2 which is an
                established Bernoulli track. 

                For established Bernoulli tracks, the existance probability is always 1. 
                The hypothesis indexing is the following:
                h_1 missed detection
                h_* associated with measurement with indexing *-1, for instance h_2 means associated with measurement1, notice it is very important to get the indexing of measurement right.
                '''
                single_target_hypothesis_index = []
                for track_index in range(number_of_previously_detected_targets): # first  update the extablished Bernoulli components
                    single_target_hypothesis_index_specified_by_previous_step_global_hypothesis=filter_predicted['globHyp'][global_hypothesis_index][track_index] # Readout the single target hypothesis index as specified by the global hypothesis of previous step
                    number_of_single_target_hypothesis_from_previous_step=len(filter_predicted['tracks'][track_index]['eB']) # total number of single target hypothesis associated with this track        
                    for ith_best_option in range(len(optimal_associations_between_measurements_and_tracks_with_infinity_tracks_transposed)): 
                        measurement_to_track_association_vector = optimal_associations_between_measurements_and_tracks_with_infinity_tracks_transposed[ith_best_option]
                        # if under this global hypothesis, measurement is associated with this track
                        measurement_index = [idx+1 for idx, value in enumerate(measurement_to_track_association_vector) if value == track_index] # if this track is part of optimal single target hypothesis          
                        if len(measurement_index)==0: # if under this global hypothesis, this track is not associated with any measurement
                            single_target_hypothesis_index=single_target_hypothesis_index_specified_by_previous_step_global_hypothesis # it is a missed detection
                            # this is the missed detection hypothesis
                        else:
                            # notice because Murky only make exclusive association, therefore we don't have to worry that there are more than one associaiton here.
                            single_target_hypothesis_index=single_target_hypothesis_index_specified_by_previous_step_global_hypothesis + number_of_single_target_hypothesis_from_previous_step*measurement_index[0]  # the new sth index is an acture association                         
                        globHyp_under_previous_step_global_hypothesis[ith_best_option][track_index]=single_target_hypothesis_index
                '''
                please refer to figure 2 of [1]. The following part discribes track 3 which is a newly initiated track based on measurements of this frame. 
                Notice there is a discrepensy on the definition of single target hypothesis: A single-target hypothesis corresponds to a
                sequence of measurements associated to a potentially detected target. 
                This step is also de facto birth of a new track because now this track would be part of the global hypothesis
                for the next step prediction and update step.             
                
                For newly established potential tracks, there are only two hypothesis
                h_-1 which means this track does not exist
                h_1 which means this track exist and it is associated with itself. 
                '''
                for i in range(number_of_measurements_at_current_frame):
                    potential_new_track_index = number_of_previously_detected_targets + i
                    for ith_best_option in range(len(optimal_associations_between_measurements_and_tracks_with_infinity_tracks_transposed)): # get he number of row vectors of opt_indices_trans
                        association_vector = optimal_associations_between_measurements_and_tracks_with_infinity_tracks_transposed[ith_best_option]
                        hypothesis_indexing= [idx+1 for idx,value in enumerate(association_vector) if value==potential_new_track_index]
                        if len(hypothesis_indexing)==0: # if there is not any measurements associated with this track
                            single_target_hypothesis_index=-1 # the single target hypothesis is h_-1
                        else:
                            single_target_hypothesis_index=0 # the single target hypothesis is h_0, which means this track has one single target hypothesis: associate with itself
                        globHyp_under_previous_step_global_hypothesis[ith_best_option][potential_new_track_index]=single_target_hypothesis_index 
                
                # stack this row to globHyp
                if len(globHyp)==0:
                    globHyp = globHyp_under_previous_step_global_hypothesis
                else:
                    np.vstack((globHyp, globHyp_under_previous_step_global_hypothesis))

            filter_updated['globHyp']=globHyp

            #Normalisation of weights of global hypotheses
            maximum_weight = max(max(x) for x in globWeightLog)
            globWeight=np.exp(globWeightLog-maximum_weight)
            globWeight=globWeight/sum(globWeight)
            filter_updated['globHypWeight']=globWeight  
        return filter_updated

    """
    Step 3: State Estimation. Section VI of [1]
    Firstly, obtain the only global hypothesis with the "maximum weight" from remaining k best global hypotheses(which are pruned from all global hypotheses by using Murty algorithm). 
    Then the state extraction is based on this only global hypothesis. Sepecifically, there are three ways to obtain this only global hypothesis:
    Option 1. The only global hypothesis is obtained via maximum globHypWeight: maxmum_global_hypothesis_index = argmax(globHypWeight).
    Option 2. First, compute for cardinality. Then compute weight_new according to cardinality. Finally, obtain the maximum only global hypothesis via this new weight via argmax(weight_new).
    Option 3. Generate deterministic cardinality via a fixed eB threshold. Then compute weight_new and argmax(weight_new) the same way as does Option 2.  
    """
    def extractStates(self, filter_updated):
        state_extraction_option = self.model['state_extraction_option']
        # Get data
        globHyp=filter_updated['globHyp']
        globHypWeight=filter_updated['globHypWeight']
        number_of_global_hypothesis = len(globHypWeight)
        number_of_tracks=len(filter_updated['tracks'])

        # Initiate datastructure
        state_estimate = {}
        mean = []
        covariance = []
        existence_probability = []
        association_history = []

        if state_extraction_option == 1:
            '''
            The three options all extract states from the highest weight global hypothesis, yet it differs
            as of how they compute for the highest weight global hypothesis
            
            This option is discribed by section VI A.
            The highest weight global hypothesis is simply the highest weight of global weight. 
            Advantage: it is easy to implement.
            Disadvantage: it does not take into account the other global hypotheses.
            '''
            if number_of_global_hypothesis>0: # If there are valid global hypotheses
                highest_weight_global_hypothesis_index = np.argmax(globHypWeight) # get he index of global hypothesis with largest weight
                highest_weight_global_hypothesis=globHyp[highest_weight_global_hypothesis_index] # get the global hypothesis with largest weight
        
                for track_index in range(number_of_tracks): # number of tracks.
                    single_target_hypothesis=int(highest_weight_global_hypothesis[track_index]-1) # Get the single target hypothesis index.
                    if single_target_hypothesis>0: # If the single target hypothesis is not does not exist
                        eB=filter_updated['tracks'][track_index]['eB'][single_target_hypothesis]
                        if eB >self.model['eB_threshold']: # if the existence probability is greater than the threshold
                            # This can be a point of confusion. The existence of a track is actually computed in two places
                            # one is here, with existence probability: if the existence probability is smaller than threshold, then we just assume that this track does not exist.
                            # another place is with hypothesis index: for a newly extablished track, if there is no measurement associated with it, the index would be 0, which means does not exist.
                            # THIS PART NEED TO PROCEED WITH CAUSION AS IT CAN BE CONFUSION                        
                            mean.append(filter_updated['tracks'][track_index]['meanB'][single_target_hypothesis]) # then assume that this track exist.
                            covariance.append(filter_updated['tracks'][track_index]['covB'][single_target_hypothesis])
                            existence_probability.append(filter_updated['tracks'][track_index]['eB'][single_target_hypothesis])
                            association_history.append(filter_updated['tracks'][track_index]['aHis'][single_target_hypothesis])

        elif state_extraction_option ==2:
            '''
            This option is discribed by section VI B.
            cardinality is computed first, and then with this cardinality, the global hypothesis weight is recomputed.
            Advantage: More accurate. with cardinality information, you can easily incorporate pertinent pieces from all global hypotheses.
            Disadvantage: more computation.
            '''
            if number_of_global_hypothesis>0: 
                predicted_cardinality_tot=np.zeros(number_of_tracks+1) # need to plus one to account the condition where cardinality is 0
                eB_tot=[] # initiate the total existence probability as an empty
                for global_hypothesis_index in range(number_of_global_hypothesis):
                    eB_for_this_global_hypothesis=np.zeros(number_of_tracks)
                    for track_index in range(number_of_tracks):
                        single_target_hypothesis_index=globHyp[global_hypothesis_index][track_index]
                        if single_target_hypothesis_index!=-1: # If this track exist
                            eB=filter_updated['tracks'][track_index]['eB'][single_target_hypothesis_index]
                            eB_for_this_global_hypothesis[track_index]=eB
        
                    eB_tot.append(eB_for_this_global_hypothesis)
                    # predicted cardinality is computed in accordance with equation 48 of [1]
                    predicated_cardinality_for_this_global_hypothesis=CardinalityMB(eB_for_this_global_hypothesis) 
                    # Multiply by weights
                    predicted_cardinality_tot+=globHypWeight[global_hypothesis_index]*predicated_cardinality_for_this_global_hypothesis
                
                #We estimate the cardinality
                max_cardinality_index=np.argmax(predicted_cardinality_tot)
                card_estimate=max_cardinality_index
    
                if card_estimate>0: # If there are tracks for this frame.
                    # Now we go through all hypotheses again
                    weight_hyp_card=[int(x) for x in np.ones(number_of_global_hypothesis)]
                    indices_sort_hyp=[]
               
                    for global_hypothesis in range(number_of_global_hypothesis):
                        weight_for_each_single_target_hypothesis_under_this_global_hypothesis=eB_tot[global_hypothesis]
                        sorted_list = [[value,idx] for idx, value in enumerate(weight_for_each_single_target_hypothesis_under_this_global_hypothesis.sort(reverse=True))]
                        sorted_weight_for_each_single_target_hypothesis_under_this_global_hypothesis = [sorted_list[x][0] for x in range(len(weight_for_each_single_target_hypothesis_under_this_global_hypothesis))]
                        sorted_index_for_each_single_target_hypothesis_under_this_global_hypothesis = [sorted_list[x][1] for x in range(len(weight_for_each_single_target_hypothesis_under_this_global_hypothesis))]

                        indices_sort_hyp.append(sorted_index_for_each_single_target_hypothesis_under_this_global_hypothesis)
                        # the following compuation is done in accordance with equation 49 of [1]
                        vector=sorted_weight_for_each_single_target_hypothesis_under_this_global_hypothesis[:card_estimate+1]+(1-sorted_weight_for_each_single_target_hypothesis_under_this_global_hypothesis[card_estimate+1:])
                        # weight_new = sum over cardinality existing tracks's weight*eB + sum over non-existence track's weight*(1-eB)
                        weight_hyp_card[global_hypothesis]*=reduce(operator.mul, vector, 1) # Equation 49 of [1]
                
                    maximum_global_hypothesis_index = np.argmax(weight_hyp_card) # Get the index of the highest weight global hypothesis
                    sorted_index_for_single_target_hypothesis_of_optimal_global_hypothesis=indices_sort_hyp[maximum_global_hypothesis_index]
                    global_hypothesis=globHyp[maximum_global_hypothesis_index]
                    # Gasia's way of initiating the states NOT SURE WHY HE DID IT THIS WAY
                    """ X_estimate=zeros(4*card_estimate,1) """

                    for track_index in card_estimate:
                        target_i=sorted_index_for_single_target_hypothesis_of_optimal_global_hypothesis[track_index] # Target Index
                        hyp_i=global_hypothesis[target_i]
                        # Gasia's way of indexing this 
                        """ X_estimate[4*i-3:4*i]=filter_updated['tracks'][target_i]['meanB'][hyp_i] """ 
                        mean.append(filter_updated['tracks'][target_i]['meanB'][hyp_i])
                        covariance.append(filter_updated['tracks'][target_i]['covB'][hyp_i])
                        existence_probability.append(filter_updated['tracks'][target_i]['eB'][hyp_i])
                        association_history.append(filter_updated['tracks'][target_i]['aHis'][hyp_i])
                        

        elif state_extraction_option==3:
            '''
            This option is discribed by section VI C.
            Instead of compute for an accurate cardinality, the cardinality is estimated by setting eB == 0.5. 
            If the eB is greater than 0.5, we assume it exist with this probability and the multiply it with weight.
            Otherwise, we assume this track does not exist, with probability (1-eB) and multiply it with weight
            and then with this newly computed weight for global hypothesis, we choose the one with maximum weight.
            Advantage: relatively accurate. can incorporate inforamtion from more than one global hypothesis.
            '''
            if number_of_global_hypothesis>0:
                new_weights=copy.deepcopy(globHypWeight)
                for global_hypothesis_index in range(number_of_global_hypothesis):
                    for track_index in range(number_of_tracks):
                        single_target_hypothesis_index=globHyp[global_hypothesis_index][track_index]
                        if single_target_hypothesis_index!=-1: # If this track exist
                            eB=filter_updated['tracks'][track_index]['eB'][single_target_hypothesis_index]
                            # the following is computed in accordance with equation 50 of [1]
                            # this is actually a way to hard code cardinality
                            # instead of compute the cardinality distribution as does in option 2, just set a random value to determine if this track exist
                            if eB >0.5: 
                                new_weights[global_hypothesis_index]*=eB # if this track exist, weight * eB
                            else:
                                new_weights[global_hypothesis_index]*=(1-eB) # if this track does not exist, weigt * (1-eB)
            
                maximum_weight_global_hypothesis_index=np.argmax(new_weights)   # select the hightest weight global hypothesis
                
                # Extract states based on the new weights
                for track_index in range(number_of_tracks):
                    single_target_hypothesis_index=globHyp[maximum_weight_global_hypothesis_index][track_index]
                    if single_target_hypothesis_index!=-1: # If this track exist
                        eB=filter_updated['tracks'][track_index]['eB'][single_target_hypothesis_index]
                        if eB>0.5:
                            mean.append(filter_updated['tracks'][track_index]['meanB'][global_hypothesis_index])
                            covariance.append(filter_updated['tracks'][track_index]['covB'][global_hypothesis_index])
                            existence_probability.append(filter_updated['tracks'][track_index]['eB'][global_hypothesis_index])
                            association_history.append(filter_updated['tracks'][track_index]['aHis'][global_hypothesis_index])

        state_estimate['mean'] = mean
        state_estimate['covariance'] = covariance
        state_estimate['existence_probability'] = existence_probability
        state_estimate['aHis'] = association_history

        return state_estimate
    
    """
    Step 4: Pruning
    4.1. Prune the Poisson part by discarding components whose weight is below a threshold.
    4.2. Prune Multi-Bernoulli RFS:
    4.2.1. Remove Bernoulli components whose existence probability is below a threshold.
    4.2.2. Remove Bernoulli components do not appear in the remaining k best global hypotheses(which are pruned from all global hypotheses by using Murty algorithm).
            By doing this, only the single target hypotheses belong to the k best global hypotheses will be left, propogated to next frame as "root" to generate more
            single target hypotheses at next frame.
            In another word, more than one global hypotheses(i.e. the k best global hypothese) will be propagated into next frame as "base" to generate 
            more global hypotheses for next frame. This is why people claim that PMBM is a MHT like filter(in MHT, the multiple hypotheses are propogated 
            from previous frame to current frame thus generating more hypotheses based the "base" multiple hypotheses from previous frame, and the best 
            hypothesis is selected (like GNN) among all the generated hypotheses at current frame.
    """ 
    
    def prune(self, filter_updated):
        # initiate filter_pruned as a copy of filter_updated
        filter_pruned = copy.deepcopy(filter_updated)

        # extract pertinent data from the dictionary
        number_of_tracks=len(filter_pruned['tracks'])
        weightPois=filter_updated['weightPois']
        global_hypothesis_weights=filter_updated['globHypWeight']
        globHyp=filter_updated['globHyp']
        maximun_number_of_global_hypothesis = self.model['maximum_number_of_hypothesis']
        eB_threshold = self.model['eB_threshold']
        Poisson_threshold = self.model['T_pruning_Pois']
        MBM_threshold = self.model['T_pruning_MBM']
        """
        Step 4.1.
        Prune the Poisson part by discarding components whose weight is below a threshold.
        """
        # if weight is smaller than the threshold, remove the Poisson component
        index_to_remove_poisson=[index for index, value in enumerate(weightPois) if value<Poisson_threshold]
        for index in range(len(index_to_remove_poisson)):
            del filter_pruned['weightPois'][index_to_remove_poisson[index]]
            del filter_pruned['meanPois'][index_to_remove_poisson[index]]
            del filter_pruned['covPois'][index_to_remove_poisson[index]]
            index_to_remove_poisson=[x-1 for x in index_to_remove_poisson]
        """
        Step 4.2.
        Pruning MB
        """
        # only keep global hypothesis whose weight is larger than the threshold
        index_to_keep=[index for index, value in enumerate(global_hypothesis_weights) if value>MBM_threshold]
        weights_after_pruning=[]
        globHyp_after_pruning=[]
        for index in index_to_keep:
            weights_after_pruning.append(global_hypothesis_weights[index])
            globHyp_after_pruning.append(globHyp[index])

        # If after previous step there are still more global hypothesis than desirable:
        # Pruning components so that there is at most a maximum number of components.
        if len(weights_after_pruning)>maximun_number_of_global_hypothesis:
            index_of_reversely_ranked_global_hypothesis_weights=np.argsort(weights_after_pruning) # the index of elements in ascending order
            index_to_keep = [index_of_reversely_ranked_global_hypothesis_weights[maximun_number_of_global_hypothesis-1-x] for x in range(maximun_number_of_global_hypothesis)]
        else:
            index_to_keep=range(len((weights_after_pruning)))
        
        weights_after_pruning=np.array(weights_after_pruning)[index_to_keep]/sum(np.array(weights_after_pruning)[index_to_keep])
        globHyp_after_pruning=np.array(globHyp_after_pruning)[index_to_keep]

        """
        Step 4.2.1.
        Remove Bernoulli components whose existence probability is below a threshold, by setting the global hypotheses indices to zero (They will be deleted by the next algorithms).
        also remove the place holders in the data structure
        """
        """ globHyp_pruned(list_remove,i)=0; """
        for track_index in range(number_of_tracks):        
            eB_for_this_track=filter_pruned['tracks'][track_index]['eB']
            to_be_removed=[index for index,value in enumerate(eB_for_this_track) if value < eB_threshold]
            global_hypothesis_to_be_removed = []
            for idx in to_be_removed:
                if idx in globHyp_after_pruning[:,track_index]: # if the single target hypothesis index is in the prune list
                    global_hypothesis_to_be_removed.append(list(globHyp_after_pruning[:,track_index]).index(idx))
            if len(global_hypothesis_to_be_removed)>0:
                np.array(globHyp_after_pruning[:,track_index])[global_hypothesis_to_be_removed]=0 # it is 0 because h_0 means this track does not exist
        
        """
        Step 4.2.2.
        Remove single-target hypotheses in each track (Bernoulli component) that do not belong to anyone of the remaining k best global hypotheses.
        By doing this, only the single target hypotheses belong to the k best global hypotheses will be left, propogated to next frame as "root" 
        to generate more single target hypotheses at next frame. 
        """

        # Remove tracks(Bernoulli components) that do not take part in any global hypothesis    
        # if the column vector sums to 0 means this track does not participate in any global hypothesis
        tracks_to_be_removed = [x for x in range(number_of_tracks) if sum(globHyp_after_pruning[:,x]) ==-len(globHyp_after_pruning)]
        if len(tracks_to_be_removed)>0:
            for idx in range(len(tracks_to_be_removed)):
                del filter_pruned['tracks'][tracks_to_be_removed[idx]]
                # remove the track column from gloyHyp_after_pruning
                np.delete(globHyp_after_pruning, tracks_to_be_removed[idx], axis=1)
                # after this index's element is purged, the indexing system need to re-calibrate
                tracks_to_be_removed= [x-1 for x in tracks_to_be_removed]
    
        # Remove single-target hypotheses in each track (Bernoulli component) that do not belong to any global hypothesis
        for track_index in range(len(filter_pruned['tracks'])): # notice that the number of tracks has changed
            single_target_hypothesis_indices_to_be_removed = []            
            number_of_single_target_hypothesis =len(filter_pruned['tracks'][track_index]['eB']) # account for h_0
            valid_single_target_hypothesis_for_this_track = globHyp_after_pruning[:,track_index] #Track indices that appear in a global hypothesis
            # We remove single target hypothesis that do not appear in the global hypothesis
            # Notice the indexing issue
            # for Bernoulli components, h_0 means missed detection
            # for Newly established Bernoulli components, h_-1 means does not exist. 
            #for single_target_hypothesis_index in range(number_of_single_target_hypothesis+1):
            stepper=0
            for single_target_hypothesis_index in range(number_of_single_target_hypothesis):
                if single_target_hypothesis_index not in valid_single_target_hypothesis_for_this_track:
                    single_target_hypothesis_indices_to_be_removed.append(single_target_hypothesis_index)
                    del filter_pruned['tracks'][track_index]['covB'][single_target_hypothesis_index-stepper]
                    del filter_pruned['tracks'][track_index]['eB'][single_target_hypothesis_index-stepper]
                    del filter_pruned['tracks'][track_index]['aHis'][single_target_hypothesis_index-stepper]
                    del filter_pruned['tracks'][track_index]['meanB'][single_target_hypothesis_index-stepper]

                    stepper+=1
            # After eliminate single target hypothesis that does not participate in global hypothesis
            # need to change the indices of globHyp_pruned
            # for instance, the global hypothesis indicated single target hypothesis index is 10, but sth 2,4,15 has been removed
            # therefore, the new index is 10-sum(2,4)=4
            if len(single_target_hypothesis_indices_to_be_removed)>0:
                for global_hypothesis_index in range(len(globHyp_after_pruning)):
                    # find out how many single target hypothesis are deleted before this single target hypothesis
                    subtract_before_this_hypothesis_index = globHyp_after_pruning[global_hypothesis_index][track_index]
                    subtraction=sum([single_target_hypothesis_indices_to_be_removed[x] for x in range(len(single_target_hypothesis_indices_to_be_removed)) if single_target_hypothesis_indices_to_be_removed[x]<subtract_before_this_hypothesis_index])
                    globHyp_after_pruning[global_hypothesis_index][track_index]-=subtraction
            '''
            if(sum(index_remove)>0)
                for j=1:size(globHyp_pruned,1)
                    sub=sum(index_remove(1:globHyp_pruned(j,i)));
                    globHyp_pruned(j,i)=globHyp_pruned(j,i)-sub;
                end
            end
            '''
        # When we eliminate a track (Bernoulli component)
        # there can be duplicate global hypotheses 
        # so the solution is to merged those duplicates into one by adding their weights 
        # and removing the duplicated ones. 
        if len(single_target_hypothesis_indices_to_be_removed)>0:
            globHyp_unique = list(set(chain(*globHyp_after_pruning))) # get the unique single target hypothesis
            if len(globHyp_unique)!=len(globHyp_after_pruning): #There are duplicate entries
                weights_unique=np.zeros(len(globHyp_unique))
                for i in range(len(globHyp_unique)):
                    unique_counter = 0
                    for j in range(len(globHyp_after_pruning)):
                        if globHyp_after_pruning[j].all() == globHyp_unique[i].all():
                            unique_counter +=1
                            if unique_counter >=2:
                                weights_unique[i]+=sum(weights_after_pruning[j])
                globHyp_after_pruning=globHyp_unique
                weights_after_pruning=weights_unique
        filter_pruned['globHyp']=globHyp_after_pruning
        filter_pruned['globHypWeight']=weights_after_pruning/sum(weights_after_pruning)
        return filter_pruned
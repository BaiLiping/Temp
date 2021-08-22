"""
%% ----------------------------------- Poisson Multi-Bernoulli Mixture(PMBM) filter ------------------------------ %%
This Python code is reproduction for the "point target PMBM filter" originally proposed in paper [1]. 
The original Matlab code for "point target PMBM filter" could be available from authors page:
https://github.com/Agarciafernandez/MTT
Corresponding video explains MBM, PMBM, TPMBM, TPMB in detail can be seen: https://www.youtube.com/playlist?list=PLadnyz93xCLjl51PzSoFhLLSp2hAYDY0H

%% ----------------------------------- Reference Papers ------------------------------------------ %%
% [1] A. F. García-Fernández, J. L. Williams, K. Granström, and L. Svensson, “Poisson multi-Bernoulli mixture filter: direct 
        derivation and implementation,” IEEE Transactions on Aerospace and Electronic Systems, 2018.
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

from PMBM_Filter_Point_Target import PMBM_Filter
from util import gen_model, gen_data_from_environment
from matplotlib import pyplot as plt
import time
import numpy as np
import gospa as gp
import pickle

# plot function for the demo
def PMBM_Filter_demo_plot(truthStates, observations, estimatedStates, clutter):
    """
    Plot all information(ground truth states, measurements which include both observations and clutters, 
    and estimated states for all targets) per scan(frame)
    """
    # Plot the ground truth state of targets.
    for truth_state_index in range(len(truthStates)):
        truthState = truthStates[truth_state_index]
        plt.plot(truthState[0], truthState[1], '.b', markersize = 10.0, label='ground truth')

    # Plot the measurements.
    for observation_index in range(len(observations)):
        observation = observations[observation_index]
        if len(observation) > 0:
            plt.plot(observation[0], observation[1], '.r', markersize = 10.0,  label='measurement')

    # Plot the clutters.
    for clutter_index in range(len(clutter)):
        clut = clutter[clutter_index]
        plt.plot(clut[0], clut[1], 'xk', markersize = 5.0, label='clutter')

    # Plot the estimated state of targets.
    for state_index in range(len(estimatedStates)):
        estimatedState = np.array(estimatedStates[state_index], dtype=np.float64)
        plt.plot(estimatedState[0], estimatedState[1], '.g', markersize = 10.0, label='estimated state')

    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Ground truth (blue), observations (red), estimated states (green) and clutter (black x)', fontsize=8)
    plt.xlim((model['xrange'][0], model['xrange'][1]))
    plt.ylim((model['yrange'][0], model['yrange'][1]))
    #plt.legend()
    fig.canvas.draw()

    # need the following to make it work on Linux
    fig.canvas.flush_events()
    time.sleep(0.1)

if __name__ == '__main__':
    # Interactive module, the figure will be shown automatically in sequence.
    plt.ion()
    # activate this line for cummulative show(showing information until current frame) or uncomment it for frame-wise show(only showing information within current frame)
    """ plt.clf() """

    fig = plt.figure()

    # Generate model
    model = gen_model()

    # Config the Bayesian filter used inside GM-PHD filter
    Bayesian_filter_config = "Kalman"

    # Config the motion_model_type used inside GM-PHD filter
    motion_model_type = "Constant Velocity"

    # Setup the starting target states (positions , velocities)
    m_start1 = np.array([300, 290, -4, -3]).reshape(4, 1)
    m_start2 = np.array([100,150, 1, 2]).reshape(4, 1)
    m_start3 = np.array([250, 50, -2, 1]).reshape(4, 1)
    m_start4 = np.array([200, 150, -3, 1]).reshape(4, 1)
    m_start5 = np.array([10, 10, 2, 5]).reshape(4, 1)

    # Initialize the initial position of targets
    targetStates = [m_start1, m_start2, m_start3, m_start4, m_start5]

    # Initialize the initial pruned intensity
    prunedIntensity = {}
    prunedIntensity['globHyp'] = []
    prunedIntensity['globHypWeight'] = []
    prunedIntensity['P'] = []

    n_scan = 300 # Duration of frames(number of frames)

    # All the informative data for analysis
    targetStates_list  = []    # It will save "all ground truth states of all targets per scan(frame)" over scans
    prunedIntensity_list = []  # It will save "the pruned intensity information of approximated posterior Poission point process per scan(frame)" over scans
    estimates_list = []        # It will save "all estimated states of all targets per scan(frame)" over scans
    measurements_list = []     # It will save "all measruements(observations + clutters) per scan(frame)" over scans
    
    path_to_save_results = 'D:/Tech_Resource/Paper_Resource/Signal Processing in General/RFS Filter/PMBM Filter/PMBM_PointTarget_Python_Demo/'
    """ path_to_save_results = 'Project_3/PMBM/' """
    gospa_record = []
    target_to_track_assigments_record =[]
    gospa_localization_record =[]
    gospa_missed_record=[]
    gospa_false_record=[]    

    
    for i in range(n_scan): # Here we execute processing for each scan time(frame)           
        # Generate ground truth states of targets, actual observations(measurements), and clutter for current scan time(frame)
        Z_k, targetStates, observations, clutter = gen_data_from_environment(model,targetStates) #Z_k is the only input of the filter, but noisy_targetStates, observations, clutter are required for plotting purposes
        
        # Apply PMBM filter for current scan time(frame). The general process is presented at Section II while the detailed implementation is presented at Section III of [2]
        tic = time.process_time()
        Filter = PMBM_Filter(model, Bayesian_filter_config, motion_model_type)
        '''
        STEP 1: Prediction 
        '''
        if i == 0:  # For the fisrt frame, there are only new birth targets rather than surviving targets thus we call seperate function.
            # the initial step the labmda for weight update is w_birthinit instead of w_birthsum
            filter_predicted = Filter.predict_initial_step()
        else:
            filter_predicted = Filter.predict(filter_pruned)
        '''
        STEP 2: Update 
        '''
        filter_updated = Filter.update(Z_k, filter_predicted, i) #Eq. 20 of [2]
        '''
        STEP 3: Extracting estimated states
        '''
        estimatedStates = Filter.extractStates(filter_updated)  # Extracting estimates from the updated intensity
        estimatedStates_mean = estimatedStates['mean']
        '''
        STEP 4: Pruning
        '''
        filter_pruned = Filter.prune(filter_updated)
        toc = time.process_time()
        print("This is the %dth scan(frame), PMBM processing takes %f seconds" %(i, (toc - tic)))

        # Plot ground truth states, actual observations, estiamted states and clutter for current scan time(frame).
        PMBM_Filter_demo_plot(targetStates, observations, estimatedStates_mean, clutter)

        # Store output information for current scan time(frame).
        # targetStates_list.append(targetStates)
        # prunedIntensity_list.append(prunedIntensity)
        # estimates_list.append(estimatedStates)
        # measurements_list.append(Z_k)
        
        # Store Metrics for Plotting
        gospa,target_to_track_assigments,gospa_localization,gospa_missed,gospa_false = gp.calculate_gospa(targetStates, estimatedStates_mean, c=10.0 , p=2, alpha=2)
        gospa_record.append(gospa)
        target_to_track_assigments_record.append(target_to_track_assigments)
        gospa_localization_record.append(gospa_localization)
        gospa_missed_record.append(gospa_missed)
        gospa_false_record.append(gospa_false)
    
    plt.close(fig)

    # Plot the results of metrices.
    x = range(len(gospa_record)) 
    plt.title("GOSPA") 
    plt.xlabel("frame number") 
    plt.ylabel("gospa") 
    plt.plot(x,gospa_record) 
    plt.savefig(path_to_save_results + 'result/gospa.png')
    plt.close()

    plt.title("gospa_localization") 
    plt.xlabel("frame number") 
    plt.ylabel("gospa_localization") 
    plt.plot(x,gospa_localization_record) 
    plt.savefig(path_to_save_results + 'result/gospa_localization.png')
    plt.close()

    plt.title("gospa_missed") 
    plt.xlabel("frame number") 
    plt.ylabel("missed") 
    plt.plot(x,gospa_missed_record) 
    plt.savefig(path_to_save_results + 'result/missed.png')
    plt.close()

    plt.title("gospa_false") 
    plt.xlabel("frame number") 
    plt.ylabel("gospa_false") 
    plt.plot(x,gospa_false_record) 
    plt.savefig(path_to_save_results + 'result/gospa_false.png')
    plt.close()
    
    # Store Data
    path =  path_to_save_results + 'data/gospa_record.pickle'
    f = open(path, 'wb')
    pickle.dump(gospa_record, f)
    f.close()

    path =  path_to_save_results + 'data/gospa_localization_record.pickle'
    f = open(path, 'wb')
    pickle.dump(gospa_localization_record, f)
    f.close()

    path =  path_to_save_results + 'data/gospa_missed_record.pickle'
    f = open(path, 'wb')
    pickle.dump(gospa_missed_record, f)
    f.close()

    path =  path_to_save_results + 'data/gospa_false_record.pickle'
    f = open(path, 'wb')
    pickle.dump(gospa_false_record, f)
    f.close()
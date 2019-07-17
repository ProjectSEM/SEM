import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from models import SEM, GRUEvent, clear_sem
from scipy.stats import multivariate_normal
from scipy.special import logsumexp


def segment_video(event_sequence, sem_kwargs):
    """
    :param event_sequence: (NxD np.array) the sequence of N event vectors in D dimensions
    :param sem_kwargs: (dict) all of the parameters for SEM
    :return:
    """
    sem_model = SEM(**sem_kwargs)
    sem_model.run(event_sequence, k=event_sequence.shape[0], leave_progress_bar=True)
    log_posterior = sem_model.results.log_like + sem_model.results.log_prior

    # clean up memory
    clear_sem(sem_model)
    sem_model = None

    return log_posterior

def bin_times(array, max_seconds, bin_size=1.0):
    """ Helper function to learn the bin the subject data"""
    cumulative_binned = [np.sum(array <= t0 * 1000) for t0 in np.arange(bin_size, max_seconds + bin_size, bin_size)]
    binned = np.array(cumulative_binned)[1:] - np.array(cumulative_binned)[:-1]
    binned = np.concatenate([[cumulative_binned[0]], binned])
    return binned

def load_comparison_data(data, bin_size=1.0):

    # Movie A is Saxaphone (185s long)
    # Movie B is making a bed (336s long)
    # Movie C is doing dishes (255s long)

    # here, we'll collapse over all of the groups (old, young; warned, unwarned) for now
    n_subjs = len(set(data.SubjNum))

    sax_times = np.sort(list(set(data.loc[data.Movie == 'A', 'MS']))).astype(np.float32)
    binned_sax = bin_times(sax_times, 185, bin_size) / np.float(n_subjs)

    bed_times = np.sort(list(set(data.loc[data.Movie == 'B', 'MS']))).astype(np.float32)
    binned_bed = bin_times(bed_times, 336, bin_size) / np.float(n_subjs)

    dishes_times = np.sort(list(set(data.loc[data.Movie == 'C', 'MS']))).astype(np.float32)
    binned_dishes = bin_times(dishes_times, 255, bin_size) / np.float(n_subjs)

    return binned_sax, binned_bed, binned_dishes

def get_binned_boundary_prop(e_hat, log_post, bin_size=1.0, frequency=30.0):
    """
    :param results: SEM.Results
    :param bin_size: seconds
    :param frequency: in Hz
    :return:
    """

    # normalize
    log_post0 = log_post - np.tile(np.max(log_post, axis=1).reshape(-1, 1), (1, log_post.shape[1]))
    log_post0 -= np.tile(logsumexp(log_post0, axis=1).reshape(-1, 1), (1, log_post.shape[1]))

    boundary_probability = [0]
    for ii in range(1, log_post0.shape[0]):
        idx = range(log_post0.shape[0])
        idx.remove(e_hat[ii - 1])
        boundary_probability.append(logsumexp(log_post0[ii, idx]))
    boundary_probability = np.array(boundary_probability)

    frame_time = np.arange(1, len(boundary_probability) + 1) / float(frequency)

    index = np.arange(0, np.max(frame_time), bin_size)
    boundary_probability_binned = []
    for t in index:
        boundary_probability_binned.append(
            # note: this operation is equivalent to the log of the average boundary probability in the window
            logsumexp(boundary_probability[(frame_time >= t) & (frame_time < (t + bin_size))]) - \
            np.log(bin_size * 30.)
        )
    boundary_probability_binned = pd.Series(boundary_probability_binned, index=index)
    return boundary_probability_binned

def get_binned_boundaries(e_hat, bin_size=1.0, frequency=30.0):
    """ get the binned boundaries from the model""" 
    
    frame_time = np.arange(1, len(e_hat) + 1) / float(frequency)
    index = np.arange(0, np.max(frame_time), bin_size)

    boundaries = np.concatenate([[0], e_hat[1:] !=e_hat[:-1]])

    boundaries_binned = []
    for t in index:
        boundaries_binned.append(np.sum(
            boundaries[(frame_time >= t) & (frame_time < (t + bin_size))]
        ))
    return np.array(boundaries_binned, dtype=bool) 

def get_point_biserial(boundaries_binned, binned_comp):
    
    
    M_1 = np.mean(binned_comp[boundaries_binned == 1])
    M_0 = np.mean(binned_comp[boundaries_binned == 0])

    n_1 = np.sum(boundaries_binned == 1)
    n_0 = np.sum(boundaries_binned == 0)
    n = n_1 + n_0

    s = np.std(binned_comp)
    r_pb = (M_1 - M_0) / s * np.sqrt(n_1 * n_0 / (float(n)**2))
    return r_pb


def get_subjs_rpb(data, bin_size=1.0):
    """get the distribution of subjects' point bi-serial correlation coeffs"""
    grouped_data = np.concatenate(load_comparison_data(data))
    
    r_pbs = []
    
    for sj in set(data.SubjNum):
        _binned_sax =  bin_times(data.loc[(data.SubjNum == sj) & (data.Movie == 'A'), 'MS'], 185, 1.0)
        _binned_bed =  bin_times(data.loc[(data.SubjNum == sj) & (data.Movie == 'B'), 'MS'], 336, 1.0)
        _binned_dishes =  bin_times(data.loc[(data.SubjNum == sj) & (data.Movie == 'C'), 'MS'], 255, 1.0)
        subs = np.concatenate([_binned_sax, _binned_bed, _binned_dishes])
        
        r_pbs.append(get_point_biserial(subs, grouped_data))
    return r_pbs

def plot_boundaries(binned_subj_data, binned_model_bounds, label, batch=0):

    # boundaries = get_binned_boundaries(log_poseterior)
    # boundaries = binned_model_bounds
    
    plt.figure(figsize=(4.5, 2.0))
    plt.plot(binned_subj_data, label='Subject Boundaries')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Boundary Probability')

    b = np.arange(len(binned_model_bounds))[binned_model_bounds][0]
    plt.plot([b, b], [0, 1], 'k:', label='Model Boundary', alpha=0.75)
    for b in np.arange(len(binned_model_bounds))[binned_model_bounds][1:]:
        plt.plot([b, b], [0, 1], 'k:', alpha=0.75)

    plt.legend(loc='upper right', framealpha=1.0)
    plt.ylim([0, 0.6])
    plt.title('"' + label + '"')
    
    sns.despine()
    plt.savefig('video_segmentation_{}_batch_{}.png'.format(label.replace(" ", ""), batch),
                dpi=600, bbox_inches='tight')
    

def convert_type_token(event_types):
    tokens = [0]
    for ii in range(len(event_types)-1):
        if event_types[ii] == event_types[ii+1]:
            tokens.append(tokens[-1])
        else:
            tokens.append(tokens[-1] + 1)
    return tokens

def get_event_duration(event_types, frequency=30):
    tokens = convert_type_token(event_types)
    n_tokens = np.max(tokens)+1
    lens = []
    for ii in range(n_tokens):
        lens.append(np.sum(np.array(tokens) == ii))
    return np.array(lens, dtype=float) / frequency

    
def run_batch(embedded_data_path, human_data_path, lmda, alfa, f_class, f_opts, batch=0, bin_size=1.0):
    Z = np.load(embedded_data_path)

    # the "Sax" movie is from time slices 0 to 5537
    sax = Z[0:5537, :]
    bed = Z[5537:5537 + 10071, :]
    dishes = Z[5537 + 10071: 5537 + 10071 + 7633, :]

    # remove the first three seconds of the sax video for clean up
    sax = sax[3*30:, :]

    # divide each of the videos by the average norm such that they are, in expectation, unit length
    sax /= np.mean(np.linalg.norm(sax, axis=1))
    bed /= np.mean(np.linalg.norm(bed, axis=1))
    dishes /= np.mean(np.linalg.norm(dishes, axis=1))

    # Z[0:5537, :] = sax
    # Z[5537:5537 + 10071, :] = bed
    # Z[5537 + 10071: 5537 + 10071 + 7633, :] = dishes

    # calibrate prior
    mode = f_opts['var_df0'] * f_opts['var_scale0'] / (f_opts['var_df0'] + 2)
    f_opts['prior_log_prob'] = multivariate_normal.logpdf(
        np.mean(Z, axis=0), mean=np.zeros(Z.shape[1]), cov=np.eye(Z.shape[1]) * mode
    ) 
    
    sem_kwargs = {
        'lmda': lmda,  # Stickyness (prior)
        'alfa': alfa, # Concentration parameter (prior)
        'f_class': f_class,
        'f_opts': f_opts
    }

    sax_log_post = segment_video(sax,    sem_kwargs)
    bed_log_post = segment_video(bed,    sem_kwargs)
    dis_log_post = segment_video(dishes, sem_kwargs)
    
    e_hat_sax = np.argmax(sax_log_post, axis=1)
    e_hat_bed = np.argmax(bed_log_post, axis=1)
    e_hat_dis = np.argmax(dis_log_post, axis=1)
    
    binned_sax_bounds = get_binned_boundaries(e_hat_sax, bin_size=bin_size)
    binned_bed_bounds = get_binned_boundaries(e_hat_bed, bin_size=bin_size)
    binned_dis_bounds = get_binned_boundaries(e_hat_dis, bin_size=bin_size)

    binned_sax_log_post = get_binned_boundary_prop(e_hat_sax, sax_log_post, bin_size=bin_size)
    binned_bed_log_post = get_binned_boundary_prop(e_hat_bed, bed_log_post, bin_size=bin_size)
    binned_dis_log_post = get_binned_boundary_prop(e_hat_dis, dis_log_post, bin_size=bin_size)
    
    # pull the subject data for comparions
    data = pd.read_csv(human_data_path, delimiter='\t')
    binned_sax_subj, binned_bed_subj, binned_dis_subj = load_comparison_data(data)

    # remove the first three seconds of the sax video
    binned_sax_subj = binned_sax_subj[3:]
    
    # save the plots 
    plot_boundaries(binned_sax_subj, binned_sax_bounds, "Cleaning Saxophone", batch=batch)
    plot_boundaries(binned_bed_subj, binned_bed_bounds, "Making a Bed",       batch=batch)
    plot_boundaries(binned_dis_subj, binned_dis_bounds, 'Washing Dishes',     batch=batch)
    
    # concatenate all of the data to caluclate the r2 values
    binned_subj_bound_freq  = np.concatenate([binned_sax_subj,     binned_bed_subj,     binned_dis_subj])
    binned_model_prob = np.concatenate([binned_sax_log_post, binned_bed_log_post, binned_dis_log_post])
    r2 = np.corrcoef(binned_subj_bound_freq, binned_model_prob)[0][1] ** 2

    # calculate the point-biserial correlation
    binned_bounds       = np.concatenate([binned_sax_bounds, binned_bed_bounds, binned_dis_bounds])
    r_pb = get_point_biserial(binned_bounds, binned_subj_bound_freq)
    
    # pull the average duration of the events
    sax_duration = np.mean(get_event_duration(binned_sax_log_post))
    bed_duration = np.mean(get_event_duration(binned_bed_log_post))
    dis_duration = np.mean(get_event_duration(binned_dis_log_post))

    # create a data frame with the model's MAP boundaries, boundary log-probabilities and 
    # human boundary frequencies for later permutation testing
    comp_data = {
        'MAP-Boundaries': binned_bounds,
        'Boundary-LogProb': binned_model_prob,
        'Human Boundary Freq': binned_subj_bound_freq,
        'Video': ['Sax'] * len(binned_sax_subj) + ['Bed'] * len(binned_bed_subj) + ['Dishes'] * len(binned_dis_subj),
        't': range(len(binned_sax_subj)) + range(len(binned_bed_subj)) + range(len(binned_dis_subj))
    }

    # and summary data as well
    summary_data = {
        'Bin Size': bin_size,
        'Event Length (Sax)': sax_duration,
        'Event Length (Bed)': bed_duration,
        'Event Length (Dishes)': dis_duration,
        'Model r2': r2,
        'Model rpb': r_pb,
        'Batch': batch
    }

    return summary_data, comp_data

def main(embedded_data_path, human_data_path, lmda, alfa, f_class, f_opts, output_tag='', n_batch=25):
    
    args = [embedded_data_path, human_data_path, lmda, alfa, f_class, f_opts]
    
    summary = []
    comp_data = []
    for batch in range(n_batch):
        summary_stats, _comp_data = run_batch(*args, batch=batch)
        summary.append(summary_stats)
        pd.DataFrame(summary).to_pickle('simulations/saved_simulations/EventR2_GRU_summary' + output_tag + '.pkl')

        _comp_data['Batch'] = [batch] * len(_comp_data['t']) 
        comp_data.append(pd.DataFrame(_comp_data))
        pd.DataFrame(comp_data).to_pickle('simulations/saved_simulations/EventR2_GRU_comp' + output_tag + '.pkl')

    return 


    


if __name__ == "__main__":

    embedded_data_path = 'data/videodata/video_color_Z_embedded_64_5epoch.npy'
    human_data_path = './data/zachs2006_data021011.dat'
    
    f_class = GRUEvent

    f_opts=dict(
        var_df0=10., 
        var_scale0=0.06, 
        l2_regularization=0.0, 
        dropout=0.5,
        n_epochs=10,
        t=4
    )

    lmda = 10**4
    alfa = 10**-1

    output_tag = '_df0_{}_scale0_{}_l2_{}_do_{}'.format(
        f_opts['var_df0'], f_opts['var_scale0'], f_opts['l2_regularization'],
        f_opts['dropout']
        )

    main(embedded_data_path, human_data_path, lmda, alfa, f_class, f_opts, output_tag, n_batch=25)
    

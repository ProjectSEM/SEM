import pandas as pd
import numpy as np
from models import SEM, clear_sem
from models.memory import evaluate_seg
from models.memory import gibbs_memory_sampler
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score
import sys

def generate_experiment(seed=None, scaling_factor=1.0, event_duration=5, n_events=5, d=25):

    n = event_duration * n_events

    if seed:
        np.random.seed(seed)

    x = np.random.randn(n, d)
    e = np.zeros(n, dtype=int)

    # embed a similarity structure within the items of each category
    # by adding the same random vector to all of the items within the
    # category
    categ_one = (np.random.randn(1, d)) * scaling_factor
    categ_two = (np.random.randn(1, d)) * scaling_factor

    for ii in range(n_events):
        if ii % 2 == 0:
            x[ii * event_duration:ii * event_duration + event_duration, :] += categ_one
            e[ii * event_duration:ii * event_duration + event_duration] = 0
        else:
            x[ii * event_duration:ii * event_duration + event_duration, :] += categ_two
            e[ii * event_duration:ii * event_duration + event_duration] = 1

    x /= np.sqrt(d)

    # give the model boundaries....
    e_tokens = np.concatenate([[False], e[1:] != e[:-1]]).cumsum()
    x_list_items = []
    for e0 in set(e_tokens):
        x_list_items.append(x[e0 == e_tokens, :])

    return x_list_items, e_tokens


# diagnostics functions

def hash_y(y):
    if y is not None:
        return np.concatenate([y[0], [y[1]], [y[2]]])
    else:
        return y


def eval_acc(y_samples, y_mem):
    acc = []
    for y_sample in y_samples:
        def item_acc(t0):
            return np.float(any([all(hash_y(yt) == hash_y(y_mem[t0])) for yt in y_sample]))
        # evaluate the accuracy of the boundary items (here, items 10 and 11)
        acc.append(np.mean([item_acc(t) for t in range(20)]))
    return np.mean(acc)


def evaluate_item_position_acc(y_samples, y_mem, t):
    acc = []
    for y_sample in y_samples:
        def item_acc(t0):
            return np.float(any([all(hash_y(yt) == hash_y(y_mem[t0])) for yt in y_sample]))
        acc.append(item_acc(t))
    return np.mean(acc)


def eval_item_acc(y_samples, y_mem, times):
    acc = []
    for y_sample in y_samples:
        def item_acc(t0):
            return np.float(any([all(hash_y(yt) == hash_y(y_mem[t0])) for yt in y_sample]))
        # evaluate the accuracy of the boundary items (here, items 10 and 11)
        acc.append(np.mean([item_acc(t) for t in times]))
    return np.mean(acc)


def score_transitions(y_samples, y_mem, t):
    acc = []
    idx = np.arange(len(y_mem))
    for y_sample in y_samples:
        y_t = [all(hash_y(y0) == hash_y(y_mem[t])) for y0 in y_sample]
        y_t1 = [all(hash_y(y0) == hash_y(y_mem[t - 1])) for y0 in y_sample]
        # position accuracy is conditioned on recall
        if any(y_t):
            if any(y_t1):
                acc.append(idx[y_t][0] == (idx[y_t1][0] + 1))
            else:
                acc.append(False)
    return np.mean(acc)

def run_block(sem_kwargs, gibbs_kwargs, epsilon_e, block_number=0):

    # generate an experiment
    x_list_items, e_tokens = generate_experiment()
    n, d = np.concatenate(x_list_items).shape

    pre_locs = [ii for ii in range(len(e_tokens) - 1) if e_tokens[ii] != e_tokens[ii + 1]]
    pst_locs = [ii for ii in range(1, len(e_tokens)) if e_tokens[ii] != e_tokens[ii - 1]]

    # Train SEM on the stimuli
    sem = SEM(**sem_kwargs)
    sem.run_w_boundaries(list_events=x_list_items, progress_bar=False)

    e_seg = np.reshape([[ii] * np.sum(e_tokens == t, dtype=int) for t, ii in enumerate(sem.results.e_hat)], -1)

    # create the corrupted memory trace
    y_mem = list()  # these are list, not sets, for hashability

    for t in range(n):
        # n.b. python uses stdev, not var
        x_mem = np.concatenate(x_list_items)[t, :] + np.random.normal(scale= gibbs_kwargs['tau'] ** 0.5, size=d)
        e_mem = [None, e_seg[t]][np.random.rand() < epsilon_e]
        t_mem = t + np.random.randint(-gibbs_kwargs['b'], gibbs_kwargs['b'] + 1)
        y_mem.append([x_mem, e_mem, t_mem])

    # add the models to the kwargs
    y_samples, e_samples, x_samples = gibbs_memory_sampler(y_mem, sem, **gibbs_kwargs)

    results = pd.DataFrame({
        'Block': [block_number],
        'Adj-r2': [adjusted_rand_score(sem.results.e_hat, np.array([0, 1, 0, 1, 0]))],
        'Recon Segment': evaluate_seg(e_samples, e_seg),
        'Overall Acc': eval_acc(y_samples, y_mem),
        'Pre-Boundary': np.mean([evaluate_item_position_acc(y_samples, y_mem, t) for t in pre_locs]),
        'Boundary': np.mean([evaluate_item_position_acc(y_samples, y_mem, t) for t in pst_locs]),
        'Transitions Pre-Boundary': np.mean([score_transitions(y_samples, y_mem, t) for t in pre_locs]),
        'Transitions Boundary': np.mean([score_transitions(y_samples, y_mem, t) for t in pst_locs]),
        'Pre-boundary Acc': eval_item_acc(y_samples, y_mem, pre_locs),
        'Boundary Acc': eval_item_acc(y_samples, y_mem, pst_locs),
    })
    clear_sem(sem)
    sem = None
    return results

def run_subject(sem_kwargs, gibbs_kwargs, epsilon_e, n_runs=16, subj_n=0, progress_bar=True):
    subject_results = []

    if progress_bar:
        for ii in tqdm(range(n_runs), desc='Running Subject'):
            subject_results.append(run_block(sem_kwargs, gibbs_kwargs, epsilon_e, block_number=ii))
    else:
        for ii in range(n_runs):
            subject_results.append(run_block(sem_kwargs, gibbs_kwargs, epsilon_e, block_number=ii))
            
    subject_results = pd.concat(subject_results)
    subject_results['Subject'] = [subj_n] * len(subject_results)
    return subject_results

def batch(sem_kwargs, gibbs_kwargs, epsilon_e, n_runs=16, n_batch=8):
    batch_results = []
    for ii in range(n_batch):
        sys.stdout.write("Beginning batch {} of {}\n".format(ii, n_batch))
        batch_results.append(run_subject(sem_kwargs, gibbs_kwargs, epsilon_e, n_runs=n_runs, subj_n=ii))
    return pd.concat(batch_results)
    

import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal as mvnorm
from scipy.special import logsumexp
from models.utils import fast_mvnorm_diagonal_logprob
np.seterr(divide = 'ignore')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import multiprocessing

def sample_pmf(pmf):
    return np.sum(np.cumsum(pmf) < np.random.uniform(0, 1))

def get_scrp_prob(e, lmda, alfa):
    """
    this function isn't used

    :param e: list event labels
    :param lmda: float, sCRP lambda
    :param alpha: float, sCRP alpha
    
    :return: total log likelihood of sequence under sCRP
    """
    c = {e0: 0 for e0 in set(e)}
    log_prob = 0
    e0_prev = None
    
    Z = alfa
    log_alfa = np.log(alfa)
    for e0 in e:
        
        l = lmda * (e0 == e0_prev)
        
        if c[e0] == 0:
            log_prob += log_alfa - np.log(Z + l)
        else:
            log_prob += np.log(c[e0] + l) - np.log(Z + l)
            

        # update the counts
        c[e0] += 1
        Z += 1
        e0_prev = e0
        
    return log_prob

def reconstruction_accuracy(y_samples, y_mem):
    """
    
    :param:     y_samples - list of y_samples
    :param:     y_mem - original corrupted memory trace

    :return:    item_accuracy, list of probabilities each item in original memory 
                is in the final reconstruction

    # checked this function on 5/20/19, this function is correct if unintuitive
    """


    acc = []
    n_orig = len(y_mem)

    for y_sample in y_samples:

        def item_acc(t):
            # loop through all of the items in the reconstruction trace, and compare them to 
            # item t in the corrupted trace.  Return 1.0 if there is a match, zero otherwise
            return np.float(any(
                [np.array_equal(yt_samp[0], y_mem[t][0]) for yt_samp in y_sample if yt_samp != None]
                ))

        # evaluate the accuracy for all of the items in the set
        acc.append([item_acc(t) for t in range(n_orig)])

    # return the vector of accuracy
    return np.mean(acc, axis=0)

def evaluate_seg(e_samples, e_true):
    acc = []
    for e in e_samples:
        acc.append(np.mean(np.array(e) == e_true))
    return np.mean(acc)
    
def create_corrupted_trace(x, e, tau, epsilon_e, b, return_random_draws_of_p_e=False):
    """
    create a corrupted memory trace from feature vectors and event labels

    :param x:           np.array of size nXd, featur vectors
    :param e:           np.array of length n, event labels
    :param tau:         float, feature corruption
    :param epsilon_e:   float, event label precision
    :param b:           int, time index corruption

    :return y_mem: list of corrupted memory tuples:
    """

    n, d = x.shape

    # create the corrupted memory trace
    y_mem = list()  # these are list, not sets, for hashability

    # pre-draw the uniform random numbers to determine the event-label corruption noise so that 
    # we can return them as needed.
    e_noise_draws = [np.random.uniform(0, 1) for _ in range(n)]

    for t in range(n):
        x_mem = x[t, :] + np.random.normal(scale=tau ** 0.5, size=d) # note, built in function uses stdev, not variance 
        e_mem = [None, e[t]][e_noise_draws[t] < epsilon_e]
        t_mem = t + np.random.randint(-b, b + 1)
        y_mem.append([x_mem, e_mem, t_mem])
    
    if return_random_draws_of_p_e:
        return y_mem, e_noise_draws

    return y_mem

def init_y_sample(y_mem, b, epsilon):
    """
    :param y_mem: list of corrupted memory traces
    :param b: time corruption noise
    :param epsilon: "forgetting" parameter 
    :returns: sample of y_mem
    """
    n_t = len(y_mem)
    y_sample = [None] * n_t

    # create a copy of y_mem for sampling without replacement
    y_mem_copy = [[x_i.copy(), e_i, t_mem] for (x_i, e_i, t_mem) in y_mem]

    # loop through timepoints in a random order
    for t in np.random.permutation(range(n_t)):

        # create a probability function over the sample sets
        log_p = np.zeros(len(y_mem_copy) + 1) - np.inf
        for ii, (x_i, e_i, t_mem) in enumerate(y_mem_copy):
            if np.abs(t_mem - t) <= b:
                log_p[ii] = 0
                # draw a sample
        log_p[-1] = np.log(epsilon)
        p = np.exp(log_p - logsumexp(log_p))  # normalize and exponentiate

        ii = sample_pmf(p)

        if ii < len(y_mem_copy):
            # only create a sample for none-None events
            y_sample[t] = y_mem_copy[ii]
            y_mem_copy = y_mem_copy[:ii] + y_mem_copy[ii + 1:]  # remove the item from the list of available
    return y_sample


def init_x_sample_cond_y(y_sample, n, d, tau):
    x_sample = np.random.randn(n, d) * tau

    for ii, y_ii in enumerate(y_sample):
        if y_ii is not None:
            x_sample[ii, :] = y_ii[0]
    return x_sample


def sample_y_given_x_e(y_mem, x, e, b, tau, epsilon):
    # total number of samples
    n, d = np.shape(x)

    #
    y_sample = [None] * n

    # create a copy of y_mem for sampling without replacement
    y_mem_copy = [[x_i.copy(), e_i, t_mem] for (x_i, e_i, t_mem) in y_mem]

    _ones = np.ones(d)

    for t in np.random.permutation(range(n)):

        # create a probability function over the sample sets
        log_p = np.zeros(len(y_mem_copy) + 1) - np.inf
        for ii, (x_i, e_i, t_mem) in enumerate(y_mem_copy):
            if np.abs(t_mem - t) <= b:
                # because we alwasy assume the covariance function is diagonal, we can use the
                # univariate normal to speed up the calculations
                log_p[ii] = fast_mvnorm_diagonal_logprob(x_i.reshape(-1) - x[t, :].reshape(-1), _ones * tau)

            # set probability to zero if event token doesn't match
            if e_i is not None:
                if e_i != e[ii]:
                    log_p[ii] -= np.inf

        # the last token is always the null token
        log_p[-1] = np.log(epsilon)
        p = np.exp(log_p - logsumexp(log_p))  # normalize and exponentiate

        # draw a sample
        ii = sample_pmf(p)

        if ii < len(y_mem_copy):
            # only create a sample for none-None events
            y_sample[t] = y_mem_copy[ii]
            y_mem_copy = y_mem_copy[:ii] + y_mem_copy[ii + 1:]  # remove the item from the list of available

    return y_sample


def sample_e_given_x_y(x, y, event_models, alpha, lmda):
    n, d = np.shape(x)

    # define a special case of the sCRP that caps the number
    # of clusters at k, the number of event models
    k = len(event_models)
    c = np.zeros(k)

    e_prev = None
    e_sample = [None] * n

    # keep a list of all the previous scenes within the sampled event
    x_current = np.zeros((1, d))

    # do this as a filtering operation, just via a forward sweep
    for t in range(n):

        # first see if there is a valid memory token with a event label
        if (y[t] is not None) and (y[t][1] is not None):
            e_sample[t] = y[t][1]
            e_prev = e_sample[t]
            c[e_sample[t]] += 1
        else:

            # calculate the CRP prior
            p_sCRP = c.copy()
            if e_prev is not None:
                p_sCRP[e_prev] += lmda

            # add the alpha value to the unvisited clusters
            if any(p_sCRP == 0):
                p_sCRP[p_sCRP == 0] = alpha / np.sum(p_sCRP == 0)
            # no need to normalize yet

            # calculate the probability of x_t|x_{1:t-1}
            p_model = np.zeros(k) - np.inf
            for idx, e_model in event_models.iteritems():
                if idx != e_prev:
                    x_t_hat = e_model.predict_next_generative(x_current)
                else:
                    x_t_hat = e_model.predict_f0()
                # because we alwasy assume the covariance function is diagonal, we can use the
                # univariate normal to speed up the calculations
                p_model[idx] = fast_mvnorm_diagonal_logprob(x[t, :] - x_t_hat.reshape(-1), e_model.Sigma)

            log_p = p_model + np.log(p_sCRP)
            log_p -= logsumexp(log_p)

            # draw from the model
            e_sample[t] = sample_pmf(np.exp(log_p))

            # update counters
            if e_prev == e_sample[t]:
                x_current = np.concatenate([x_current, x[t, :].reshape(1, -1)])
            else:
                x_current = x[t, :].reshape(1, -1)
        e_prev = e_sample[t]

        # update the counts!
        c[e_sample[t]] += 1

    return e_sample


def sample_x_given_y_e(x_hat, y, e, event_models, tau):
    """
    x_hat: n x d np.array
        the previous sample, to be updated and returned

    y: list
        the sequence of ordered memory traces. Each element is
        either a list of [x_y_mem, t_mem] or None

    e: np.array of length n
        the sequence of event tokens

    event_models: dict {token: model}
        trained event models

    tau:
        memory corruption noise

    """

    # total number of samples
    n, d = np.shape(x_hat)

    x_hat = x_hat.copy()  # don't want to overwrite the thing outside the loop...

    # Note: this a filtering operation as the backwards pass is computationally difficult. 
    # (by this, we mean that sampling from  Pr(x_t| x_{t+1:n}, x_{1:t-1}, theta, e, y_mem) is intractable
    # and we thus only sample from Pr(x_t|, x_{1:t-1}, theta, e, y_mem), which is is Gaussian)
    for t in np.random.permutation(range(n)):
        # pull the active event model
        e_model = event_models[e[t]]

        # pull all preceding scenes within the event
        x_idx = np.arange(len(e))[(e == e[t]) & (np.arange(len(e)) < t)]
        x_prev = np.concatenate([
            np.zeros((1, d)), x_hat[x_idx, :]
        ])

        # pull the prediction of the event model given the previous estimates of x
        f_x = e_model.predict_next_generative(x_prev)

        # is y_t a null tag?
        if y[t] is None:
            x_bar = f_x
            sigmas = e_model.Sigma
        else:
            # calculate noise lambda for each event model
            u_weight = (1. / e_model.Sigma) / (1. / e_model.Sigma + 1. / tau)

            x_bar = u_weight * f_x + (1 - u_weight) * y[t][0]
            sigmas = 1. / (1. / e_model.Sigma + 1. / tau)

        # draw a new sample of x_t 
        # N.B. Handcoding a function to draw random variables introduced error into the algorithm
        # and didn't save _any_ time.
        x_hat[t, :] = mvnorm.rvs(mean=x_bar.reshape(-1), cov=np.diag(sigmas))

    return x_hat


def gibbs_memory_sampler(y_mem, sem_model, memory_alpha, memory_lambda, memory_epsilon, b, tau,
                         n_samples=250, n_burnin=100, progress_bar=True, leave_progress_bar=True):
    """

    :param y_mem: list of 3-tuples (x_mem, e_mem, t_mem), corrupted memory trace
    :param sem_mdoel: trained SEM instance
    :param memory_alpha: SEM alpha parameter to use in reconstruction
    :param memory_labmda: SEM lmbda parameter to use in reconstruction
    :param memory_epsilon: (float) parameter controlling propensity to include null trace in reconstruction
    :param b: (int) time index corruption noise
    :param tau: (float, greater than zero) feature vector corruption noise
    :param n_burnin: (int, default 100) number of Gibbs sampling itterations to burn in
    :param n_samples: (int, default 250) number of Gibbs sampling itterations to collect
    :param progress_bar: (bool) use progress bar for sampling?
    :param leave_progress_bar: (bool, default=True) leave the progress bar at the end? 

    :return: y_samples, e_samples, x_samples - Gibbs samples
    """

    event_models =  {
        k: v for k, v in sem_model.event_models.iteritems() if v.f_is_trained
    }

    d = np.shape(y_mem[0][0])[0]
    n = len(y_mem)

    #
    e_samples = [None] * n_samples
    y_samples = [None] * n_samples
    x_samples = [None] * n_samples

    y_sample = init_y_sample(y_mem, b, memory_epsilon)
    x_sample = init_x_sample_cond_y(y_sample, n, d, tau)
    e_sample = sample_e_given_x_y(x_sample, y_sample, event_models, memory_alpha, memory_lambda)

    # loop through the other events in the list
    if progress_bar:
        def my_it(iterator):
            return tqdm(iterator, desc='Gibbs Sampler', leave=leave_progress_bar)
    else:
        def my_it(iterator):
            return iterator
    
    for ii in my_it(range(n_burnin + n_samples)):

        # sample the memory features
        x_sample = sample_x_given_y_e(x_sample, y_sample, e_sample, event_models, tau)

        # sample the event models
        e_sample = sample_e_given_x_y(x_sample, y_sample, event_models, memory_alpha, memory_lambda)

        # sample the memory traces
        y_sample = sample_y_given_x_e(y_mem, x_sample, e_sample, b, tau, memory_epsilon)

        if ii >= n_burnin:
            e_samples[ii - n_burnin] = e_sample
            y_samples[ii - n_burnin] = y_sample
            x_samples[ii - n_burnin] = x_sample

    return y_samples, e_samples, x_samples

## there appears to be something wrong with this function! do not use for now
# def multichain_gibbs(y_mem, sem_model, memory_alpha, memory_lambda, memory_epsilon, b, tau, n_chains=2,
#                          n_samples=250, n_burnin=50, progress_bar=True, leave_progress_bar=True):

#     """

#     :param y_mem: list of 3-tuples (x_mem, e_mem, t_mem), corrupted memory trace
#     :param sem_mdoel: trained SEM instance
#     :param memory_alpha: SEM alpha parameter to use in reconstruction
#     :param memory_labmda: SEM lmbda parameter to use in reconstruction
#     :param memory_epsilon: (float) parameter controlling propensity to include null trace in reconstruction
#     :param b: (int) time index corruption noise
#     :param tau: (float, greater than zero) feature vector corruption noise
#     :param n_burnin: (int, default 100) number of Gibbs sampling itterations to burn in
#     :param n_samples: (int, default 250) number of Gibbs sampling itterations to collect
#     :param progress_bar: (bool) use progress bar for sampling?
#     :param leave_progress_bar: (bool, default=True) leave the progress bar at the end? 

#     :return: y_samples, e_samples, x_samples - Gibbs samples
#     """

#     y_samples, e_samples, x_samples = [], [], []
#     for _ in range(n_chains):
#         _y0, _e0, _x0 = gibbs_memory_sampler(
#             y_mem, sem_model, memory_alpha, memory_lambda, memory_epsilon, 
#             b, tau, n_samples, progress_bar, False, leave_progress_bar
#             )
#         y_samples += _y0
#         e_samples += _e0
#         x_samples += _x0
#     return y_samples, e_samples, x_samples
import numpy as np
import pandas as pd
from models import *
from tqdm import tnrange
from simulations.exp_dubrow import run_subject, generate_experiment



# SEM parameters
df0 = 1.
scale0 = .2

mode = df0 * scale0 / (df0 + 2)
print("Prior variance (mode): {}".format(mode))

lmda = 10.0  # stickyness parameter
alfa = 1.  # concentration parameter

f_class = GRUEvent
f_opts=dict(var_scale0=scale0, var_df0=df0)

# create the corrupted memory trace
# noise parameters
b = 2
tau = 0.1
print("tau: {}".format(tau))

# set the parameters for the Gibbs sampler
gibbs_kwargs = dict(
    memory_alpha = alfa,
    memory_lambda = lmda,
    memory_epsilon = np.exp(-20),
    b = b,  # re-defined here for completeness
    tau = tau,  # ibid
    n_samples = 250,
    n_burnin = 100,
    progress_bar=False,
)
sem_kwargs = dict(lmda=lmda, alfa=alfa, f_class=f_class, f_opts=f_opts)

epsilon_e = 0.25

x_list_items, e_tokens = generate_experiment()

mode = df0 * scale0 / (df0 + 2)
print("Prior variance (mode): {}".format(mode))
print("Median Feature variance: {}".format(
    np.median(np.var(np.concatenate(x_list_items), axis=0))))

sem_kwargs = dict(
    lmda=lmda, alfa=alfa, f_class=f_class, f_opts=f_opts
)

sem = SEM(**sem_kwargs)
sem.run_w_boundaries(list_events=x_list_items)
print sem.results.e_hat

# fig, axes = plt.subplots(2, 1)
# axes[0].plot(sem.results.log_prior)
# axes[1].plot(sem.results.log_like)
# # plt.show()

from tqdm import tnrange, tqdm

n_batch = 25
n_runs = 16

results = []
for ii in tqdm(range(n_batch), desc='Itteration', leave=True):

    for b in [1, 2, 5, 10]:

          gibbs_kwargs = dict(
               memory_alpha = alfa,
               memory_lambda = lmda,
               memory_epsilon = np.exp(-20),
               b = b,  # re-defined here for completeness
               tau = tau,  # ibid
               n_samples = 250,
               n_burnin = 100,
               progress_bar=False,
          )


          _res = run_subject(
               sem_kwargs, gibbs_kwargs, epsilon_e, n_runs=n_runs, subj_n=ii, progress_bar=False
          )

          # clean up the results and run simple analyses
          _res['b'] = b
          _res.loc[np.isnan(_res['Transitions Pre-Boundary'].values), 'Transitions Pre-Boundary'] = 0.0
          _res.loc[np.isnan(_res['Transitions Boundary'].values), 'Transitions Boundary'] = 0.0
          _res['PreVsPost'] = _res['Transitions Pre-Boundary'].values - _res['Transitions Boundary'].values 


          results.append(_res)
          pd.concat(results).to_pickle('Dubrow_param_sensitivity.pkl')

print "Done!"
    


# EventSegmentation

Accompanying code for the manuscript "Structured event memory: a neuro-symbolic model of event cognition", Franklin, Norman, Ranganath, Zacks, and Gershman 2019, BioRxiv, [https://doi.org/10.1101/541607](https://doi.org/10.1101/541607)

Contains the SEM model,  a few basic demonstrations, and the all of the simulations in the paper.

The main code is listed in the `models` module:
* `models.sem`: contains the code for the SEM model
* `models.event_models`: contains code for the various neural network models used by SEM. They all 
    share a similar structures
    
There are a few prepackaged demonstrations in Jupyter notebooks meant to demonstrate basic functions of the model.
 These have been pre-run and can be opened on github without installation:
* `Demo - Toy Data (Segmentation)`: These simulations demonstrate how SEM can segement simple, 2D dynamical systems with
various different methods of estimating the event dynamics of the system.
* `Demo - HRR.ipynb`: Demonstration of the Holographic reduced representation
* `Demo - Motion Capture Data.ipynb`: Simulations of the SEM model on the 3D motion capture data. 
* `Segmentation - Generalizing Structure`: a demonstratation of how the HRR and Bayesian inference are used
to generalize structure when delineating event boundaries.


There are also multiple simulations that demonstrates how the model can capture a wide range of empirical phenomena
in the event cognition literature:
* `Segmentation - Video (Dishes)`: show human-like segementation of video data, originally used in Zacks & Tversky, 2001.
The dimensionality of the videos has been reduced using a variational auto-encoder, the code for which is available as 
 a seperate library [https://github.com/ProjectSEM/VAE-video](https://github.com/ProjectSEM/VAE-video)
* `Segmentation - Schapiro (n250)`: a simulation of the task found in Schapiro, et al, 2013.
* `Memory Simluation (Bower, 3 setences)`: a simulation of the classic finding in Bower, 1979
* `Memory Simluation (Radvansky & Copeland, 2006)`: a simulation of the findings in Radvansky & Copeland, 2006
* `Memory Simluation (Pettijohn, et al, 2016)`:a simulation of the findings in Pettijohn, et al, 2016
* `Memory Simluation (Dubrow and Davachi, 2013; 2016) `: a simulation of the finding in Dubrow and Davachi, 2013

There are also follow-up analyses:
* `Memory Simluation (Dubrow and Davachi, 2013; 2016) parameter sensitivity`: looks at memory  corruption noise and how it effects order memory
* `Segmentation - Generalizing Structure (Stationary)`: looks at a reduced model that does not simulate event dynamics.

#### Installation Instructions

This library run on Python 2.7 and uses the tensorflow and keras and libraries for neural networks. 

I recommend using Anaconda python and a virtual environment. [You can find instructions to install Anaconda
 here](https://docs.anaconda.com/anaconda/install/).

Once you have anaconda installed, you can install a virtual environment by running

    conda env create --file environment.yml

This will install everything you need to run the demonstration notebooks.

You'll need to activate the virtual environments and open jupyter to access the demonstration notebooks. To do so, run

    source activate sem
    jupyter notebook


To deactivate the virtual environment, run

    source deactivate


Note: if these instructions do not work for some reason, this library uses:
    Tensorflow v1.9
    Keras v2.2.0
    Anaconda
# EventSegmentation 

<a href="https://colab.research.google.com/github/ProjectSEM/SEM/blob/master/Tutorials/Demo - Segmentation and Memory Tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>

Accompanying code for the manuscript "Structured event memory: a neuro-symbolic model of event cognition", Franklin, Norman, Ranganath, Zacks, and Gershman 2019, BioRxiv, [https://doi.org/10.1101/541607](https://doi.org/10.1101/541607)

Contains the SEM model, a few basic demonstrations, and the all of the simulations in the paper.

The main code is listed in the `models` module:
* `models.sem`: contains the code for the SEM model
* `models.event_models`: contains code for the various neural network models used by SEM. They all 
    share a similar structures
    
There is runnable code in Jupyter notebooks:
* `Tutorials`: Contains tutorials, runnable in Google Colab.
* `PaperSimulations`: Contains the simulations presented in the paper.  These have been designed to run locally, with the
 dependencies listed in the enviornments.yml file and have not been tested in colab. These have been pre-run and can be
  opened on github without installation. 

#### Installation Instructions

This library run on Python 2.7 and uses the tensorflow and keras and libraries for neural networks. 

I recommend using Anaconda python and a virtual environment. [You can find instructions to install Anaconda
 here](https://docs.anaconda.com/anaconda/install/).

Once you have anaconda installed, you can install a virtual environment by running

    conda env create --file environment.yml

This will install everything you need to run the Jupyter notebooks.  Note that all of the simulations were run with these
packages versions and may not work with more recent versions (for example, TensorFlow is under active development).

You'll need to activate the virtual environments and open jupyter to access the demonstration notebooks. To do so, run

    conda activate sem
    jupyter notebook


To deactivate the virtual environment, run

    conda deactivate


Note: if these instructions do not work for some reason, the critical libraries the model uses are:
    
* Anaconda Python 2.7
* Tensorflow v1.9
* Keras v2.2.0

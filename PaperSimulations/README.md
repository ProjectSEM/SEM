# Simulations in the Paper



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


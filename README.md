# ENPS GPU
A GPU-based simulator for Enzymatic Numerical P Systems

This software tool is capable of simulating massively parallel ENPS models, i.e., ENPS models composed of a large number of membranes. It is implemented on [CUDA](http://www.nvidia.com/object/cuda_home_new.html), a novel technology which takes advantage of the parallel computational power of graphic cards to execute massively parallel tasks. So far, this simulator is only available for Windows 7 systems, although it will be also adapted for Linux systems. This simulator requires an NVIDIA Graphic Card with [compute capability](http://en.wikipedia.org/wiki/CUDA) 2.0 or greater, as it uses recursive calls not supported by previous capability versions.</p>

To call this simulator, simply unzip the content of this file inside a folder in your local machine. Change directory in the command prompt to the folder in which you have unzipped the previous file. then, execute:

>./ENPSGPUSimulator "model.xml" "iterations" "results.txt" "0"

where:

* **"model.xml"** is the file containing the model to simulate. This model must be in PENPSXML format.
* **"iterations"** is the number of iterations to simulate.
* **"results.txt"** is the file containing the results of the simulation. These results contain a header with a short briefing of the simulation (number of programs, simulation time, etc), along with the ending values of the variables.


This software is available under the GNU General Public License (GPL) version 3.0. More information can be found on the following papers:


* M. García-Quismondo, L.F. Macías-Ramos, M.J. Pérez-Jiménez. Implementing Enzymatic Numerical
P Systems for AI Applications by means of Graphic Processing Units. *Beyond Artificial Intelligence:
Contemplations, Expectations, Applications, Springer Berlin-Heidelberg. Series: Topics in Intelligent
Engineering and Informatics*, vol. 4 (2013), ch. 14, 137-159.
* M. García-Quismondo, A.B. Pavel, M.J. Pérez-Jiménez. Simulating Large-Scale ENPS Models by Means
of GPU. *Proceedings of the Tenth Brainstorming Week on Membrane Computing. Volume I*, 30/01/2012-
03/02/2012, Seville, Spain, p. 137-152.

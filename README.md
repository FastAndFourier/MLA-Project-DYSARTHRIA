# MLA-Project-DYSARTHRIA

Contributors: Bastien DUSSARD, Marc FAVIER, Fabien PICHON, Louis SIMON

Master 2 ISI, Sorbonne Université

Subject: **Learning to detect dysarthria from raw speech**

Rerefence paper: *J. Millet et N. Zeghidour, « Learning to detect dysarthria from raw speech », arXiv:1811.11101 [cs], jan. 2019 Available on: http://arxiv.org/abs/1811.11101*

Dataset is available here: http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html

MODEL ARCHITECTURE : TIME DOMAIN FILTERBANK

You have the possibility to run the network with a specific convolution layer reproducing at initialization a mel filterbank with the keyword "TDfilt" for the "frontEnd" argument.
This make the network a bit bigger with 2 convolutions and some algebra added at its beginning then the speed of the training is slowed down. We need to apply a unique windowing fonction accross channels (2nd convolution).
/!\ THE USE OF THE GROUPED CONVOLUTION MAKE THE USE OF A GPU MANDATORY. THE BACKPROPAGATION CAN NOT BE APPLIED WITH A CPU, THEN WE CAN ONLY DO INFERENCE WITH A CPU.

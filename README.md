# MLA-Project-DYSARTHRIA

Contributors: Bastien DUSSARD, Marc FAVIER, Fabien PICHON, Louis SIMON

Master 2 ISI, Sorbonne Université

Subject: **Learning to detect dysarthria from raw speech**

Rerefence paper: *J. Millet et N. Zeghidour, « Learning to detect dysarthria from raw speech », arXiv:1811.11101 [cs], jan. 2019 Available on: http://arxiv.org/abs/1811.11101*

Dataset is available here: http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html

****

### Python functions and notebooks

The model is build and train using two python files:

    + `blocks.py` which contains elementary blocks of the architecture
    + `model1.py` which gathers model building, training, and a main function

Examples of command:

`python model1.py -frontEnd melfilt -normalization pcen -lr 0.001 -batch_size 2 epochs 10 -decay True`

`python model1.py -frontEnd LLD  -lr 0.003 -batch_size 8 epochs 10 `

`python model1.py -frontEnd TDfilt -lr 0.002 -batch_size 4 epochs 10 -decay True`


These functions utilize a set of four files which allow to extract and pre-process the data:

    + `time_mfb.py` for Time Domain filterbanks
    + `LLD_extract.py` for LLDs
    + `melfilt_preprocess.py` for mel-filterbanks
    + `create_dataset.py` for raw speech extraction

Finally, three notebooks (one for each model) allow to perform training of the network, preferably with a GPU.



### Model architecture

#### Time-domain filterbanks

You have the possibility to run the network with a specific convolution layer reproducing at initialization a mel filterbank with the keyword "TDfilt" for the "frontEnd" argument. This make the network a bit bigger with 2 convolutions and some algebra added at its beginning but the speed of the training is not strongly impact. We need to apply a unique windowing fonction accross channels (2nd convolution).

:exclamation: THE USE OF THE GROUPED CONVOLUTION MAKE THE USE OF A GPU MANDATORY. THE BACKPROPAGATION CAN NOT BE APPLIED WITH A CPU, THEN WE CAN ONLY DO INFERENCE WITH A GPU :exclamation:

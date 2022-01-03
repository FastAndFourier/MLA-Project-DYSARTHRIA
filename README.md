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

Finally, one notebook `notebook_MLA_dysarthria.ipynb` allows to evaluate models.:exclamation: This notebook is design for Google Colab with the GPU on.

### Libaries
`tensorlflow`, `librosa` and `opensmile`


### Note on Time-domain filterbanks model and GPU

:exclamation: 
THE USE OF THE GROUPED CONVOLUTION MAKE THE USE OF A GPU MANDATORY. THE BACKPROPAGATION CAN NOT BE APPLIED WITH A CPU, THEN WE CAN ONLY DO INFERENCE WITH A GPU 
:exclamation:


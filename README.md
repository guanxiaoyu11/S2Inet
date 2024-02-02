S2Inet is a python tool Transformer-based Sequence-to-Image Network for Accurate Nanopore Sequence Recognition

Prerequesites:

python 3.5+

pytorch 1.0

Running S2Inet:

$ python training.py

File Description:

1.The train_data.zip file contains the training dataset and validation dataset of the RNA classification data dataset used in the paper.

2.The test_data.zip file contains the test dataset of the RNA classification data dataset used in the paper. 

3.The Model file is the trained Model model used in this paper.

4.The training.py file is used for model loading, training, testing and loading datasets.

5.The ont-gaf.py file is used to preprocess ONT encoded data. 

6.The gadf_train.npz file is the training dataset of the ONT barcode dataset.

7.The label_train.npz file is the training dataset label of the ONT barcode dataset.

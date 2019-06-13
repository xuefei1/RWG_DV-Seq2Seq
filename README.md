# RWG_DV-Seq2Seq
Code for the Relevant Word Generator + Dual Vocabulary Sequence-to-Sequence generative framework.

As presented in paper: A Deep Generative Approach to Search Extrapolation and Recommendation

### Important files

model_rwg.py: contains implementation for the RWG model.

model_dv_seq2seq.py: contains implementation for the DV_Seq2Seq model.

run_rwg.py: driver file for training/testing the RWG model.

run_dv_seq2seq.py: driver file for training/testing the DV_Seq2Seq model.

### Word vector and data

A pre-trained word vector file is needed. Put the file inside the **libs/** folder and set the name using command line argument **-word_vec_file**

The word vector file must have a format that works with *gensim.models.KeyedVectors.load_word2vec_format(...)*

Due to privacy concerns, the original data cannot be released. Put your own data inside **data/**: 

The expected data format for the RWG model (one line per instance):

*Input words | target relevant words*

The expected data format for the DV_Seq2Seq model (one line per instance):

*Input words | relevant words | target output words*

**Words must be segmented by a single space**

For each model there should be a train data file, a validation data file and a test data file, the filenames can be changed inside the driver files.

\
Requires pytorch 0.4.1+ and Python 3.

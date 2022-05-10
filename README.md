<h1>Named Entity Recognition</h1>

NER is the task of labelling words in a sentence as persons, locations, etc. <br>

The task, in its entirety, involves selecting a sub-label amongst an hierarchy of NER labels. This implementation, though, does not delve into this hierarchy, rather the labels are clubbed until 9 high-level NER tags are left. BIO encoding has been used.<br>

A hybrid biLSTM-CRF model is used, much as outlined in [this](https://www.aclweb.org/anthology/N16-1030) paper. A fast implementation of linear chain CRFs with fully vectorized training is provided.

The dataset: Publicly available GMB dataset, see https://gmb.let.rug.nl/data.php
<br>
More about the NER task can be read [here](https://aclanthology.org/N16-1030/)

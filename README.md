
# CrohnsDisease
Final year Masters project at Imperial College on tackling Crohn's Disease

In this work we establish a baseline for binary prediction of abnormal and healthy MRI volumes using deep learning

To this end we use a small 3D ResNet with added soft attention layers

## Repo Guide
Brief explanation of important files

### Training
<tt>/run_crohns.sh</tt> - Run config specifying training and model parameters (root of execution)

<tt>/run.py</tt> - Parses config options and builds TF Record decode function

<tt>/pipeline.py</tt> - Builds TF Record load pipeline using decode function

<tt>/runner.py</tt> - Constructs and iteratively trains TF network, continually loading TF Record data through pipeline

<tt>/model/resenet.py</tt> - Specification for 3D Resnet

<tt>/model/attention.py</tt> - Specification of [soft attention mechanism](https://arxiv.org/abs/1804.05338)

### Data pre-processing
Files under <tt>/Crohns/</tt> generate the TF Records that are consumed in training

<tt>/Crohns/metadata.py</tt> Loads labels and MRI metadata into memory

<tt>/Crohns/preprocess.py</tt> Crops and rescales MRI volumes

<tt>/Crohns/tfrecords.py</tt> Generates a series of training and test TF Records for cross-fold evaluation (introducing duplication)

<tt>/Crohns/generate_tfrecords.py</tt> Configures and executes the generation process (i.e. how many cross folds)

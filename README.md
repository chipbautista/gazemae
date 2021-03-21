# GazeMAE
This repo contains the code for the paper:
**GazeMAE: General Representations of Eye Movements using a Micro-Macro Autoencoder**, accepted to [ICPR 2020](http://icpr2020.it).
Preprint: https://arxiv.org/abs/2009.02437

## Data
To run the code, you would need the data sets from the ff. websites/papers:
1. **FIFA**: "Predicting human gaze using low-level saliency combined with face
    detection", Cerf M., Harel J., Einhauser W., Koch C., Neural Information Processing Systems (NIPS) 21, 2007. Get the data from [here](https://www.morancerf.com/publications). You need `general.mat`, `faces-jpg` folder, and `subjects` folder. Place them in `data/Cerf2007-FIFA` folder.
2. **ETRA**: Check ETRA 2019 Challenge [here](https://etra.acm.org/2019/challenge.html). Place the files in `data/ETRA2019/` folder.
3. **EMVIC**: Request data set from [here](http://kasprowski.pl/emvic/dataset.php). Place the files in `data/EMVIC2014/official_files/` folder.
4. **MIT-LowRes**: "Fixations on Low-resolution Images", Tilke Judd, Frédo Durand, Antonio Torralba. JoV 2011. Check project page [here](http://people.csail.mit.edu/tjudd/LowRes/index.html). Place the files in `data/MIT-LOWRES/` folder.

Then, create another folder where the code will automatically store the preprocessed datasets: `mkdir generated-data`

## Training
### Requirements
Packages used are in requirements_conda.txt.
Running a simple `conda install --file requirements_conda.txt` might work.
But if it doesn't, here are the main packages used:
- pytorch 1.2.0
- cudatoolkit 10.0.130
- scikit-learn 0.22.1
- tensorboard 2.1.0
- h5py 2.9.0
- pandas 0.25.3
- seaborn 0.10.0
- future 0.18.2
- tables 3.6.1 (install via pip)

### Running the code
With the data files in the correct locations, you should be able to run this Python command: `python train.py --signal-type=vel -bs=128 -vt=2 -hz=500 --hierarchical --slice-time-windows=2s-overlap`

This trains the velocity model described in the paper, with the following settings: batch size is 128, the gaze data is preprocessed into 2-second segments (`vt` or viewing time) and at 500 Hz sampling frequency. `hierarchical` means the AE uses the two-bottleneck architecture, and `2s-overlap` means the gaze data is preprocessed into overlapping 2-second segments. More arguments can be found in `settings.py`

## Files/Codes
### Loading the datasets
The core Python class for loading the datasets is found in `data/corpus.py`, while the code to handle the individual datasets is in `data/corpora.py`. Each data set has a class, containing code to process the raw data.

### Preprocessing
The code for preprocessing the gaze data (normalization, cutting them up into segments) is within the SignalDataset class in data/data.py. All of the data sets are run through this class so that the preprocessing is consistent. It's also a PyTorch Dataset class, which will be later on used with PyTorch DataLoader to iterate during training.

### Autoencoder model
The architecture is defined in the files in `network/` folder. The autoencoder model is in `network/autoencoder.py`. but the actual layer functionalities are in `encoder.py` and `decoder.py`.


## Pre-trained models
I'm uploading the two main models described in the paper.
1. `models/pos-i3738` - Position AE at iteration 3738
2. `models/vel-i8528` - Velocity AE at iteration 8528

To use the pre-trained model, you should be able to load it with `torch.load(model_file)`. For reference, all model functionalities (initializing, loading, saving) are found in `network/__init__.py`

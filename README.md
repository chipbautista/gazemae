# GazeMAE
This repo contains the code for the paper:
**GazeMAE: General Representations of Eye Movements using a Micro-Macro Autoencoder**, accepted to [ICPR 2020](http://icpr2020.it).
Preprint: https://arxiv.org/abs/2009.02437

*If you plan to read through the code, please know that I experimented with regular AEs, VAEs and $\beta$-TCVAEs. I removed the non-regular AE code as much as I can while trying not to alter the code too much that it affects the output. There may still be some code left that I wrote to handle different types of training (e.g. some `mean` and `logvar` variables, and a dictionary to store different loss values). Just keep that in mind so you don't get confused as much.*

To run the code, you would need the data sets from the ff. websites/papers:
1. **FIFA**: "Predicting human gaze using low-level saliency combined with face
    detection", Cerf M., Harel J., Einhauser W., Koch C., Neural Information Processing Systems (NIPS) 21, 2007. Get the data from [here](https://www.morancerf.com/publications)
2. **ETRA**: Check ETRA 2019 Challenge [here](https://etra.acm.org/2019/challenge.html)
3. **EMVIC**: Request data set from [here](http://kasprowski.pl/emvic/dataset.php)
4. **MIT-LowRes**: "Fixations on Low-resolution Images", Tilke Judd, Frédo Durand, Antonio Torralba. JoV 2011. Check project page [here](http://people.csail.mit.edu/tjudd/LowRes/index.html)

To train with using default velocity AE parameters:
`python train.py --signal-type=vel -bs=128 -vt=2 -hz=500 -hierarchical --slice-time-windows=2s-overlap`

You may change the network parameters in `settings.py`.

### Pre-trained models
I'm uploading the two main models described in the paper.
1. `models/pos-i3738` - Position AE at iteration 3738
2. `models/vel-i8528` - Velocity AE at iteration 8528

# Multi Track Music Generation Using Generative Adversarial Networks

This is the final group project of "Neural Networks and Deep Learning" course, held at University of Padua. 
The group that developed the project was composed by Eleonora Bergamin, Stefano Campagnola and me.

In this project, we investigate the complex task of generating music using state-of-the-art models, based on generative adversarial networks. 
The implemented architecture is capable of generating piano-rolls of mono and multi-instrumental polyphonic music both from scratch and conditioned on a previous sample.

The model was trained on two different datasets containing both mono and multi-track MIDI samples. 
- In the first case, the **[unlabelled VGMIDI](https://github.com/lucasnfe/vgmidi/tree/master)** dataset was used, encompassing piano samples. 
- In the second case, the **[LPD 5 cleansed](https://www.kaggle.com/datasets/cloudoak/lpd-5-cleansed)** dataset was used, including samples of bass, drums, guitar, piano, and strings.

Objective evaluation metrics were then implemented to evaluate the quality of generated music.

The main files to run this project are the following:
- `Generating_Music_With_GANs.ipynb` contains data pre-processing and visualization, training procedure for the model, and metrics' evaluation.
- `models.py` contains the models' architectures.
- `metrics.py` contains the functions defining the metrics.

In case someone was interested just in generating some samples, in `GAN weights` folder, the weights of the trained models are contained.
File names summarize in some way the fixed training parameters and used model's architecture.

`Generated_Samples` and `losses` folders respectively contain some generated midi samples and the loss functions associated to the performed training processes.

For more detailed information rely on project's report: `Multi_Track_Music_Generation_Using_Generative_Adversarial_Networks.pdf`.

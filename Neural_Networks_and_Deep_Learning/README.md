# Multi Track Music Generation Using Generative Adversarial Networks

This is the final group project for the "Neural Networks and Deep Learning" course at the University of Padua. The project was developed by a group composed of Stefano Campagnola, Sebastiano Monti, and myself.

In this project, we explore the complex task of generating music using state-of-the-art models based on generative adversarial networks. The architecture we implemented can generate piano rolls of mono and multi-instrumental polyphonic music, both from scratch and conditioned on a previous sample.

The model was trained on two different datasets, containing respectively mono and multi-track MIDI samples.
- In the first case, we used the **[unlabeled VGMIDI](https://github.com/lucasnfe/vgmidi/tree/master)** dataset, which includes piano samples.
- In the second case, we used the **[LPD 5 cleansed](https://www.kaggle.com/datasets/cloudoak/lpd-5-cleansed)** dataset, which includes samples of bass, drums, guitar, piano, and strings.

We then implemented objective evaluation metrics to assess the quality of the music generated.

The main files needed to run this project are the following:
- `Generating_Music_With_GANs.ipynb`,  which contains data pre-processing and visualization, the training procedure for the model, and metrics evaluation.
- `models.py`, which contains the architectures of the models.
- `metrics.py`, which contains the functions defining the metrics.
  
If you are interested in generating samples, the `GAN weights` folder contains the weights of the trained models. The file names summarize the fixed training parameters and the architecture of the used model.

The `Generated_Samples` and `losses` folders contain some generated MIDI samples and the loss functions associated with the performed training processes, respectively.

For more detailed information, please refer to the project’s report:`Multi_Track_Music_Generation_Using_Generative_Adversarial_Networks.pdf`.

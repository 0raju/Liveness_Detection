# Overview

This is the source code for our research titled:  **Iris Print Attack Detection using Eye Movement Signals**.

The paper is published in ETRA'2022. You can find it [here](https://dl.acm.org/doi/abs/10.1145/3517031.3532521).

## Cite

Mehedi Hasan Raju, Dillon J Lohr, and Oleg Komogortsev. 2022. Iris Print Attack Detection using Eye Movement Signals. In 2022 Symposium on Eye Tracking Research and Applications (ETRA '22). Association for Computing Machinery, New York, NY, USA, Article 70, 1–6. [https://doi.org/10.1145/3517031.3532521](https://doi.org/10.1145/3517031.3532521)

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Property of Texas State University.

## Contact

Please reach out to Dr. Komogortsev at ok11@txstate.edu with questions related to the availability of data for this publication.

# Folder/File Description

### Files
1. **run.py**: Main script to train and evaluate the model.
2. **data_preprocessing.py**: Contains helper functions and data-preprocessing steps.
3. **model.py**: Implements C-Resnet18.
4. **train.py**: Contains the training steps.
5. **evaluate.py**: Contains the evaluation steps.
6. **iris.yml**: Specifies the compatible environment to run the code.

### Folders
1. **ETPAD.v2**: You need to download the data from [this link](https://userweb.cs.txstate.edu/~ok11/etpad_v2.html). Contact Dr. Komogortsev for access.
   - Downloaded ETPAD.v2 should contain three folders:
     - **LIVE_EYE_MOVEMENTS**
     - **SAS_I_EYE_MOVEMENTS**
     - **SAS_II_EYE_MOVEMENTS**
2. **Trained_model**: Pre-trained models are stored here.

# Steps to Run the Code and Replicate the Figures

1. Create a virtual environment using the provided `iris.yml` file (e.g., *iris*):
    ```bash
    $ conda env create -f iris.yml
    $ conda activate iris
    ```

2. Run the following command to train and evaluate the model:
    ```bash
    $ python run.py
    ```

3. To train the model with LIVE and SASI/SASII data using `window_size=1500` and `stride=125`, run:
    ```bash
    $ python train.py --sas=1 --window_size=1500 --stride=125
    $ python train.py --sas=2 --window_size=1500 --stride=125
    ```

4. To evaluate the model with LIVE and SASI/SASII data using `window_size=1500` and `stride=125`, run:
    ```bash
    $ python evaluate.py --sas=1 --window_size=1500 --stride=125
    $ python evaluate.py --sas=2 --window_size=1500 --stride=125
    ```

### Note:
This code has been made public two years after the original publication. It might not produce exactly the same results as those reported in the paper, but the differences should be minimal.

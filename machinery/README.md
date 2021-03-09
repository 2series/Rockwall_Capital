# Fault Detection of Machinery by Sound
## Mar 10, 2021
## by Rihad Variawa, Samira Variawa, Salena Ruhi, Delvin Hada
### Artificial Intelligence

![](https://media.giphy.com/media/3o6fJbnN3Wdut4Akda/giphy.gif)

This repository contains a sample on how to perform machinery fault detection by sound (based on the [MIMII Dataset](https://zenodo.org/record/3384388)) leveraging several approaches

## Overview

This repository is accompanied by a [blog post](?????????????), where we compare and contrast two different approaches to identify a malfunctioning machine for which we have audio recordings: we will start by building a neural network based on an autoencoder architecture and we'll then use an image-based approach where we'll feed “images from the audio” (namely spectrograms) to an image based ML classifier

## Installation instructions

[Create an AWS account](https://portal.aws.amazon.com/gp/aws/developer/registration/index.html) if you do not already have one and login

Navigate to the SageMaker console and create a new instance. Using an **ml.c5.2xlarge instance** with a **25 GB attached EBS volume** is recommended to process the dataset comfortably

You need to ensure that this notebook instance has an IAM role which allows it to call the AWS Rekognition Custom Labels API:
1. In our IAM console, look for the SageMaker execution role endorsed by our notebook instance (a role with a name like AmazonSageMaker-ExecutionRole-yyyymmddTHHMMSS)
2. Click on Attach Policies and look for this managed policy: AmazonRekognitionCustomLabelsFullAccess
3. Check the box next to it and click on Attach Policy

Our SageMaker notebook instance can now call the Rekognition Custom Labels APIs

You can know navigate back to the SageMaker console, then to the Notebook Instances menu. Start your instance and launch either Jupyter or JupyterLab session. From there, you can launch a new terminal and clone this repository into your local development machine using `git clone`

## Repository structure

Once you've cloned this repo, browse to the [data exploration](1_data_exploration.ipynb) notebook: this first notebook will download and prepare the data necessary for the others

The dataset used is a subset of the MIMII dataset dedicated to industrial fans sound. This 10 GB archive will be downloaded in the /tmp directory: if you're using a SageMaker instance, you should have enough space on the ephemeral volume to download it. The unzipped data is around 15 GB large and will be located in the EBS volume, make sure it is large enough to prevent any out of space error

```
.
|
+-- README.md                                 - This instruction file

+-- autoencoder/
|   |-- model.py                              - The training script used as an entrypoint of the
|   |                                             TensorFlow container
|   \-- requirements.txt                      - Requirements file to update the training container
                                                  at launch

+-- images/                                   - Assets used in in the introduction and README.md

+-- tools/
|   |-- rekognition_tools.py                  - Utilities to manage Rekognition custom labels models
|   |                                             (start, stop, get inference...)
|   |-- sound_tools.py                        - Utilities to manage sounds dataset
|   \-- utils.py                              - Various tools to build files list, plot curves, and
                                                  confusion matrix...

+-- 1_data_exploration.ipynb                  - START HERE: data exploration notebook, useful to
|                                                 generate the datasets, get familiar with sound datasets
|                                                 and basic frequency analysis

+-- 2_custom_autoencoder.ipynb                - Using the SageMaker TensorFlow container to build a
|                                                 custom autoencoder

\-- 3_rekognition_custom_labels.ipynb         - Performing the same tasks by calling the Rekognition
                                                  Custom Labels API
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information

## License
This collection of notebooks is licensed under the MIT-0 License. See the LICENSE file

## Conflicts of Interest: The practitioner declare no conflict of interest

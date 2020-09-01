# ICPC2020 CodeGNN
This code is part of the reproducibility package for the ICPC 2020 paper "Improved Code Summarization via a Graph Neural Network" - [arxiv](https://arxiv.org/abs/2004.02843)

The reproducibility package has three parts:
1. the code found in this repository
2. the trained models, predictions, and tokenizer (.tok) files can be downloaded [HERE](https://icpc2020.s3.us-east-2.amazonaws.com/ICPC_2020_data.tar.gz)
3. the fully processed data (as a pkl file) can be downloaded [HERE](https://icpc2020.s3.us-east-2.amazonaws.com/dataset.pkl)

The dataset we used is from our NAACL'19 paper "Recommendations for Datasets for Source Code Summarization" where the unprocessed data can be found and downloaded [HERE](http://leclair.tech/data/funcom/)

This code uses Keras v2.3.1 and Tensorflow v1.15.2 

## Running the code and models

To run the trained models from the paper download the three parts of the reproducibility package and run predict.py. Predict.py takes the path to the model file as a positional argument and will output the prediction file to ./modelout/predictions.

`python3 predict.py {path to model} --gpu 0 --modeltype {model type: codegnngru|codegnnbilstm|codegnndense} --data {path to data download}`

`python3 predict.py ./mymodels/codegnngru.h5 --gpu 0 --modeltype codegnngru --data ./mymodels`

To train a new model run train.py with the modeltype and gpu options set.

`python3 train.py --gpu 0 --modeltype codegnnbilstm --data ./mydata`

## Processing Files
We have added our processing files to the processing directory. These files will not work without some tweaking due to hard coded paths/file names/databases but show how we processed our code, comments, and AST. To generate the base AST XML we use SrCML which can be downloaded [HERE](https://www.srcml.org/)

## Cite this work
```
@inproceedings{
leclair2020codegnn,
title={Improved Code Summarization via a Graph Neural Network},
author={Alex LeClair, Sakib Haque, Lingfei Wu, Collin McMillan},
booktitle={2020 IEEE/ACM International Conference on Program Comprehension},
year={2020},
month={Oct.},
doi={10.1145/3387904.3389268}
ISSN={978-1-4503-7958-8/20/05}
}
```

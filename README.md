# DSG: Document Structure Generator

## Paper
Further information and evaluations can be found in the [paper](paper.pdf)


## Requirements

We tested our code on a Linux machine with python 3.10, detectron 0.6 and pytorch 1.11.

Installation instructions for detectron and pytorch can be found [here](https://github.com/facebookresearch/detectron2)

To setup the environment with all the required dependencies, we provide further steps [here](INSTALL.md)




## Datasets and model download
Please use [following link](https://drive.google.com/drive/folders/1Dijhs-zj9KYyusy-e_DvoBDUNktsu78S?usp=share_link) to download model checkpoints and datasets.

Unzip `checkpoints.zip` and `datasets.zip` at the root level of this repository.


## File Naming in our Demo 

`DSG_E2E_arxivdocs`: DSG trained on arXivdocs

`DSG_E2E_eperiodica`: DSG trained on E-Periodica



## Demonstration of our system
### Demo entity prediction with postprocesing:
Note: When running the code for the first time, glove word embeddings are automatically downloaded.

First, create an output directory, e.g. at `./demo/EP_outputs`.

To run DSG for prediction and use grammar-based postprocessing, run:
```
python visualizations/demo.py --config-file ./configs/sgg_end2end_EP.yaml --input ./datasets/eperiodica3/imgs/val/* --output ./demo/EP_outputs --raw_output ./demo/EP_outputs --opts MODEL.ROI_SCENEGRAPH_HEAD.PREDICT_USE_VISION True MODEL.WEIGHTS ./checkpoints/DSG_E2E_eperiodica/dsg_e2e_eperiodica_checkpoint.pth TEST.USE_GRAMMAR_POSTPROCESSING True
```

### Demo of hOCR creation
The hOCR creation demo uses the outputs created by the previous script.

For convenience, we prepared outputs for one sample and a jupyter notebook to demonstrate our hOCR creation and querying [here](sysdemo/system_demonstration.ipynb)

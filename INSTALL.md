## Requirements
Our code is tested on
- Python 3.10
- Pytorch 1.11
- Detectron 0.6


### Installation guide:

1. Install PyTorch 1.11

2. install the cython package (e.g. `pip install cython`)
3. install modified pycocotools:
    - go to subdirectory cocoapi/PythonAPI
    - `python setup.py build_ext install`
2. Copy the official detectron2 repository
3. Install detectron2, e.g. via `python setup.py develop` while inside detectron2 repository
4. Inside DSG directory, run `python setup.py develop`
5. Install additional required packages:

- opencv-python
- scikit-learn
- networkx
- imantics
- easydict
- h5py
- numpy < 1.24
- network == 2.8
- shapely

**Note**: Please make sure the detectron2 version correctly corresponds to the pytorch and cuda versions installed. Please check [this page](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) for installation instructions and common issues.

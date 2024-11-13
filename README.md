# Depth estimation from stereo camera

Given camera parameters, input images (check subfolders in `data`) and an the position of an object of interest the tool finds the disparity and depth of the object.

## Project structure

├── benchmark.py
├── estimate.py
├── helpers.py
├── input.py
├── data
│   ├── A008-01-27
│   │   ├── 1000mm.bag
│   │   ├── 1250mm.bag
│   │   ├── 750mm.bag
│   │   ├── left.yaml
│   │   └── right.yaml
│   ├── A008-02-02
│   │   ├── 1000mm.bag
│   │   ├── 1000mm.bag-roi.yaml
│   │   ├── 1250mm.bag
│   │   ├── 1250mm.bag-roi.yaml
│   │   ├── 750mm.bag
│   │   ├── 750mm.bag-roi.yaml
│   │   ├── left.yaml
│   │   └── right.yaml
│   └── production_data
│       ├── 625e6b3065f651eb60cde755_left.jpg
│       ├── 625e6b3065f651eb60cde755_right.jpg
│       └── 625e6b3065f651eb60cde755_stereo_camera_parameters.json
├── Makefile
├── requirements.txt
├── README.md
└── test_input.py

## Installation

    $ make install

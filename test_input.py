from input import extract_rgb

from input import Meta
import numpy as np


def test_extract_rgb():

    # wanted to extract stereo pair from BAG file and check
    # their size to height and width from *camera_info*
    # parameters, but as these files are big I don't want to
    # commit them.

    try:
        _ = extract_rgb("nothing")
        
    except AssertionError:
        assert True

    except Exception as exception:
        print(f"Add case for the failed test.\n{exception}")
        assert False

def test_yaml():

    meta = Meta('./data/left.yaml')
    assert np.all(meta.camera_matrix == np.array([
        [2530.0544 , 0.     ,  2048.     ],
        [0.     ,  2526.54308,  1500.    ],
        [0.     ,     0.     ,     1.    ]
    ]))

    assert np.all(meta.distortion_coefficients == np.array([
        -0.064243, 0.055123, -0.001135, -0.002677, 0.00000
    ]))

    assert np.all(meta.rectification_matrix == np.array([
        [0.99930353,  0.00168517, -0.03727759],
        [-0.00178025,  0.99999525, -0.00251746],
        [0.03727317,  0.00258207,  0.99930178]
    ]))

    assert np.all(meta.projection_matrix == np.array([
        [2619.26228,     0.     ,  2175.62744,     0.],
        [0.     ,  2619.26228,  1498.41986,     0. ],
        [0.     ,     0.     ,     1.     ,     0.]
    ]))

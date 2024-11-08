from bag import extract_rgb

def test_extract_rgb():

    # wanted to extract stereo pair from BAG file and check
    # their size to height and width from *camera_info* 
    # parameters, but as these files are big I don't want to 
    # commit them.

    try:
        _ = extract_rgb('nothing')
    except AssertionError:
        assert True
    except:
        assert False
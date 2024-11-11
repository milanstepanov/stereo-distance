import numpy as np

def combine_stereo(left_img, right_img, roi_x_left: int=-1, roi_y_right: int=-1):
    """Assume horizontal stereo pair. Take horizontal middle parts of the pair and
    concatenate together. """

    side_by_side = np.zeros_like(left_img)
    
    h,w,c = left_img.shape
    side_by_side[:,:w//2,:] = left_img[:,w//4:3*w//4,:]
    side_by_side[:,w//2:,:] = right_img[:,w//4:3*w//4,:]

    return side_by_side if roi_x_left < 1 or roi_y_right < 1 else side_by_side[roi_x_left:roi_y_right,...]
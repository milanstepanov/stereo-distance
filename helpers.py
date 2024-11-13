import numpy as np

from functools import wraps
import tracemalloc
import time

import os

path_log = '.logs/log'

if os.path.exists(path_log):
    for root, dirs, files in os.walk(path_log, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.removedirs(path_log)

os.makedirs(path_log)



def measure_memory_usage(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()

        # Call the original function
        result = func(*args, **kwargs)

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        with open(os.path.join(path_log, 'memory.txt'), 'a') as fp:

            # Print the top memory-consuming lines
            fp.write(f"\nMemory usage of {func.__name__}:\n")
            for stat in top_stats[:5]:
                fp.write(f"{str(stat)}\n")


        # Return the result
        return result

    return wrapper

def measure_execution_time(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        with open(os.path.join(path_log, 'time.txt'), 'a') as fp:
            fp.write(f'\nFunction {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')

        return result
    
    return timeit_wrapper



def combine_stereo(left_img, right_img, roi_x_left: int=-1, roi_y_right: int=-1):
    """Assume horizontal stereo pair. Take horizontal middle parts of the pair and
    concatenate together. """

    side_by_side = np.zeros_like(left_img)
    
    h,w,c = left_img.shape
    side_by_side[:,:w//2,:] = left_img[:,w//4:3*w//4,:]
    side_by_side[:,w//2:,:] = right_img[:,w//4:3*w//4,:]

    return side_by_side if roi_x_left < 1 or roi_y_right < 1 else side_by_side[roi_x_left:roi_y_right,...]
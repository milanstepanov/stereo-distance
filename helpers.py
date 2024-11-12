import numpy as np

import tracemalloc
import time

def measure_memory_usage(func):
    def wrapper(*args, **kwargs):
        tracemalloc.start()

        # Call the original function
        result = func(*args, **kwargs)

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        # Print the top memory-consuming lines
        print(f"\nMemory usage of {func.__name__}:")
        for stat in top_stats[:5]:
            print(stat)

        # Return the result
        return result

    return wrapper

def measure_execution_time(func):
    def timeit_wrapper(*args, **kwargs):

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        print(f'\nFunction {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
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
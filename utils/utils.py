import numpy as np
from scipy.interpolate import CubicSpline

def time_warp(x, sigma=0.2, num_knots=4):
    """
    Apply Time Warping to the input signal x.
    
    Args:
        x (np.array): Input signal of shape (2, 128) or (128, 2). 
                      We assume (2, 128) as per RML2016 standard, but will handle transpose.
        sigma (float): Standard deviation for the random warp.
        num_knots (int): Number of knots for the spline.
        
    Returns:
        np.array: Warped signal of the same shape.
    """
    # Ensure shape is (Time, Channels) for processing, then transpose back if needed
    transposed = False
    if x.shape[0] == 2 and x.shape[1] != 2:
        x = x.T # (128, 2)
        transposed = True
        
    time_steps = x.shape[0]
    
    # 1. Generate random knots
    # Uniformly spaced knots including start and end
    orig_steps = np.linspace(0, time_steps - 1, num_knots + 2)
    
    # Random perturbations
    # Normal distribution centered at 1.0
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(num_knots + 2, ))
    random_warps[0] = 1.0 # Fix start
    random_warps[-1] = 1.0 # Fix end
    
    # Cumulative sum to get warped time steps
    warped_steps = np.cumsum(random_warps)
    # Normalize to match original length
    warped_steps = warped_steps * (time_steps - 1) / warped_steps[-1]
    
    # 2. Fit Cubic Spline
    # Maps Original Time -> Warped Time
    warper = CubicSpline(orig_steps, warped_steps)
    
    # Generate new time indices
    new_indices = warper(np.arange(time_steps))
    
    # 3. Resample (Linear Interpolation)
    ret = np.zeros_like(x)
    for i in range(x.shape[1]):
        ret[:, i] = np.interp(np.arange(time_steps), new_indices, x[:, i])
            
    if transposed:
        ret = ret.T # Back to (2, 128)
        
    return ret.astype(np.float32)

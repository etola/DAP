import numpy as np

def image_uv(width, height):
    """
    Generate UV coordinates for an image of given width and height.
    Returns:
        np.ndarray: (height, width, 2) array with UV coordinates in range [0, 1].
    """
    # Create coordinates from 0.5 to size-0.5
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    
    # Meshgrid to get (H, W) grids. 
    # Note: numpy meshgrid default is 'xy', so first arg corresponds to columns (width)
    xv, yv = np.meshgrid(u, v)
    
    # Stack to get (H, W, 2)
    return np.stack([xv, yv], axis=-1)

def points_to_normals(points, mask=None):
    """
    Compute normals from a grid of 3D points.
    Args:
        points (np.ndarray): (H, W, 3) array of 3D points.
        mask (np.ndarray): Optional (H, W) boolean mask.
    Returns:
        tuple: (normals, mask)
            normals: (H, W, 3) array of unit normal vectors.
            mask: (H, W) boolean mask indicating valid normals.
    """
    # Calculate gradients using central differences
    # np.gradient returns [gradient_along_axis_0, gradient_along_axis_1, ...]
    grad_y, grad_x = np.gradient(points, axis=(0, 1))
    
    # Cross product of tangent vectors
    # tangent_x is grad_x, tangent_y is grad_y
    # normal = cross(x_tangent, y_tangent)
    normal = np.cross(grad_x, grad_y)
    
    # Normalize
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    
    # Avoid division by zero
    valid = norm > 1e-6
    normal = np.divide(normal, norm, out=np.zeros_like(normal), where=valid)
    
    if mask is None:
        mask = np.ones(points.shape[:2], dtype=bool)
        
    return normal, mask




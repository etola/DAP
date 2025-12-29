import torch
import torch.nn.functional as F

def image_uv(width, height):
    """
    Generate UV coordinates for an image of given width and height.
    Returns:
        torch.Tensor: (height, width, 2) tensor with UV coordinates in range [0, 1].
    """
    # Create coordinates
    u = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    v = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    
    # Meshgrid
    # indexing='ij' means first arg is rows (y), second is cols (x).
    # But we want u (x) and v (y).
    # If we pass (v, u) with 'ij', we get (H, W).
    yv, xv = torch.meshgrid(v, u, indexing='ij')
    
    # Stack to get (H, W, 2)
    return torch.stack([xv, yv], dim=-1)

def points_to_normals(points, mask=None):
    """
    Compute normals from a grid of 3D points.
    Args:
        points (torch.Tensor): (H, W, 3) or (B, H, W, 3) tensor of 3D points.
        mask (torch.Tensor): Optional mask.
    Returns:
        tuple: (normals, mask)
    """
    input_shape = points.shape
    if points.ndim == 3:
        points = points.unsqueeze(0) # (1, H, W, 3)
    
    B, H, W, C = points.shape
    
    # Rearrange to (B, C, H, W) for padding/convolutions if needed,
    # or just use manual shifts.
    p = points.permute(0, 3, 1, 2) # (B, 3, H, W)
    
    # Central differences with padding
    # dy: (p(y+1) - p(y-1))/2
    p_up = F.pad(p, (0, 0, 1, 1), mode='replicate')[:, :, :-2, :]
    p_down = F.pad(p, (0, 0, 1, 1), mode='replicate')[:, :, 2:, :]
    dy = (p_down - p_up) / 2
    
    # dx: (p(x+1) - p(x-1))/2
    p_left = F.pad(p, (1, 1, 0, 0), mode='replicate')[:, :, :, :-2]
    p_right = F.pad(p, (1, 1, 0, 0), mode='replicate')[:, :, :, 2:]
    dx = (p_right - p_left) / 2
    
    # Cross product
    # dim=1 is the channel dimension (3)
    normal = torch.cross(dx, dy, dim=1)
    
    # Normalize
    normal = F.normalize(normal, dim=1, eps=1e-6)
    
    # Rearrange back to (B, H, W, 3)
    normal = normal.permute(0, 2, 3, 1)
    
    if len(input_shape) == 3:
        normal = normal.squeeze(0)
    
    if mask is None:
        mask = torch.ones(input_shape[:-1], dtype=torch.bool, device=points.device)
        
    return normal, mask




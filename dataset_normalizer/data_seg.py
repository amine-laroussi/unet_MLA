import numpy as np
import cv2 as cv 
from scipy.ndimage import rotate, shift



def elastic_deformation_3x3(image, mask, sigma=10, p=0.7):

    """
    Elastic deformation as described in the original UNet paper.
    
    - Random displacement vectors on a coarse 3x3 grid
    - Displacements sampled from N(0, sigma)
    - Bicubic interpolation to full resolution
    
    Args:
        image (H,W) numpy array
        mask  (H,W) numpy array
        sigma (float): std deviation of displacement in pixels
        p (float): probability to apply deformation
    
    Returns:
        deformed image, deformed mask
    """
    # --- 1. Apply function with probability ---
    if np.random.rand() > p:
        return image, mask
    H, W = image.shape

    # --- 2. Generate 3x3 displacement grid ---
    dx_small=np.random.normal(0,sigma,(3,3)).astype(np.float32)
    dy_small=np.random.normal(0,sigma,(3,3)).astype(np.float32)
    # --- 3. Upscale to full resolution using bicubic interpolation ---
    dx = cv.resize(dx_small, (W, H), interpolation=cv.INTER_CUBIC)   #  cubic interpolation (continuous values)
    dy = cv.resize(dy_small, (W, H), interpolation=cv.INTER_CUBIC)

    # --- 4. Create coordinate grid ---
    x, y = np.meshgrid(np.arange(W), np.arange(H))

    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    # --- 5. Apply deformation ---
    img_deformed = cv.remap(
        image,
        map_x,
        map_y,
        interpolation=cv.INTER_CUBIC,
        borderMode=cv.BORDER_REFLECT_101
    )

    mask_deformed = cv.remap(
        mask,
        map_x,
        map_y,
        interpolation=cv.INTER_NEAREST,
        borderMode=cv.BORDER_REFLECT_101
    )

    return img_deformed, mask_deformed

def random_rotate_shift(image,mask,max_angle=15,max_shift=3,p=0.8):
    if np.random.rand()>p:
        return image ,mask
    angle = np.random.uniform(-max_angle, max_angle)
    tx = np.random.uniform(-max_shift, max_shift)
    ty = np.random.uniform(-max_shift, max_shift)

    image = rotate(image, angle, reshape=False, order=3, mode="reflect")
    mask  = rotate(mask, angle, reshape=False, order=0, mode="reflect")

    image = shift(image,  shift=(ty, tx), order=3, mode="reflect")
    mask  = shift(mask,   shift=(ty, tx), order=0, mode="reflect")

    return image, mask

def intensity_variation(image):
    """
    Apply photometric augmentation to the image:
    - Changes brightness and contrast
    - Does NOT modify spatial geometry
    """


    # Convert PIL image to NumPy array (float32 in range [0,1])
    image_np = np.array(image).astype(np.float32) / 255.0

    # Apply intensity variation
    gain = np.random.uniform(0.9, 1.1)       # contrast
    bias = np.random.uniform(-0.05, 0.05)    # brightness
    image_np = image_np * gain + bias

    # Clip values to keep them in valid range
    image_np = np.clip(image_np, 0.0, 1.0)

    return image_np * 255.0














































































































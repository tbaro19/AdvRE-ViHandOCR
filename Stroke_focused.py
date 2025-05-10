import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vietocr.tool.translate import process_input, translate
from tqdm import tqdm
import editdistance
import seaborn as sns
import pandas as pd

# Set matplotlib to non-interactive backend
plt.switch_backend('agg')

# Existing functions
def load_model(checkpoint_path):
    """Load the OCR model"""
    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cuda:0'
    config['weights'] = checkpoint_path
    
    # Create predictor for easy inference
    predictor = Predictor(config)
    
    return predictor, config

def process_image_for_attack(image_path, config):
    """Process image to prepare for attack"""
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # For normal prediction
    img_tensor = process_input(img, config['dataset']['image_height'], 
                              config['dataset']['image_min_width'], 
                              config['dataset']['image_max_width'])
    
    # For PGD attack - normalize to [0,1]
    img_np = np.array(img) / 255.0
    
    return img, img_tensor, img_np

def predict_text(predictor, img):
    """Predict text using the predictor"""
    return predictor.predict(img)

def structure_aware_attack(img_np, predictor, epsilon=0.05, iters=100, 
                          focus_edges=True, smoothness=0.8):
    """
    Perform adversarial attack focusing on character structure/edges
    
    Args:
        img_np: Original image (numpy array, values in [0,1])
        predictor: OCR predictor function
        epsilon: Maximum perturbation strength (kept smaller to avoid visual detection)
        iters: Number of iterations
        focus_edges: Whether to focus perturbations on edge regions of text
        smoothness: Controls smoothness of perturbations (higher = smoother)
        
    Returns:
        Adversarial image (numpy array, values in [0,1])
    """
    from scipy.ndimage import gaussian_filter, sobel
    import cv2
    
    # Get original prediction
    orig_img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    orig_pred = predictor.predict(orig_img_pil)
    
    # Convert to float32
    img = img_np.astype(np.float32)
    
    # Detect edges - potential character regions
    img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, 100, 200) / 255.0
    
    # Dilate edges to include full character strokes
    kernel = np.ones((3, 3), np.uint8)
    edge_mask = cv2.dilate(edges, kernel, iterations=2)
    
    # Apply Gaussian blur to create a softer mask that extends beyond hard edges
    edge_mask = gaussian_filter(edge_mask, sigma=1.5)
    
    # Normalize the mask
    if edge_mask.max() > 0:
        edge_mask = edge_mask / edge_mask.max()
    
    # Create weight mask for balancing perturbation
    weight_mask = edge_mask if focus_edges else np.ones_like(edge_mask)
    
    # Expand mask to match image dimensions if needed
    if len(weight_mask.shape) == 2 and len(img.shape) == 3:
        weight_mask = np.expand_dims(weight_mask, axis=2)
        weight_mask = np.repeat(weight_mask, 3, axis=2)
    
    # Best results tracking
    best_adv_img = img.copy()
    best_score = 0
    best_direction = None
    
    # Iteratively build up perturbation
    accumulated_perturbation = np.zeros_like(img)
    
    for iteration in range(iters):
        # Create a smooth perturbation direction (important for less visible noise)
        if iteration == 0 or iteration % 10 == 0:
            # Generate new direction occasionally
            direction = np.random.uniform(-1, 1, img.shape).astype(np.float32)
            # Make it smooth by applying Gaussian filter
            for c in range(direction.shape[2]):
                direction[:,:,c] = gaussian_filter(direction[:,:,c], sigma=smoothness*2)
            # Normalize direction
            direction_norm = np.sqrt(np.mean(direction**2)) + 1e-8
            direction = direction / direction_norm
        
        # Scale the smooth perturbation
        step_size = epsilon * (1 - iteration/iters) * 2  # Gradually reduce step size
        perturbation = step_size * direction
        
        # Apply weight mask to focus perturbation on character regions
        perturbation = perturbation * weight_mask
        
        # Update accumulated perturbation
        new_perturbation = accumulated_perturbation + perturbation
        new_perturbation = np.clip(new_perturbation, -epsilon, epsilon)
        
        # Apply the updated perturbation
        perturbed_img = np.clip(img + new_perturbation, 0, 1)
        
        # Check prediction
        perturbed_img_pil = Image.fromarray((perturbed_img * 255).astype(np.uint8))
        adv_pred = predictor.predict(perturbed_img_pil)
        
        # Calculate score - we want different prediction but minimal change
        import editdistance
        score = editdistance.eval(adv_pred, orig_pred)
        
        # Update best result if this perturbation works better
        if score > best_score:
            best_score = score
            best_adv_img = perturbed_img.copy()
            best_direction = direction
            accumulated_perturbation = new_perturbation.copy()
            
            # Early stopping if we've significantly changed the prediction
            if score > len(orig_pred) * 0.5:
                break
        
        # If we're not making progress, try a new direction
        if iteration > 0 and iteration % 20 == 0 and score == 0:
            direction = np.random.uniform(-1, 1, img.shape).astype(np.float32)
            for c in range(direction.shape[2]):
                direction[:,:,c] = gaussian_filter(direction[:,:,c], sigma=smoothness*3)
            direction_norm = np.sqrt(np.mean(direction**2)) + 1e-8
            direction = direction / direction_norm
    
    return best_adv_img

def stroke_modification_attack(img_np, predictor, epsilon=0.05, iters=100):
    """
    Perform adversarial attack that specifically targets character strokes
    by thinning, thickening or breaking connections
    
    Args:
        img_np: Original image (numpy array, values in [0,1])
        predictor: OCR predictor function
        epsilon: Maximum perturbation strength
        iters: Number of iterations
        
    Returns:
        Adversarial image (numpy array, values in [0,1])
    """
    from scipy.ndimage import gaussian_filter
    import cv2
    
    # Get original prediction
    orig_img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    orig_pred = predictor.predict(orig_img_pil)
    
    # Convert to float32
    img = img_np.astype(np.float32)
    
    # Convert to grayscale for stroke analysis
    img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Find text regions (assume darker pixels are text on lighter background)
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Get skeleton (thin lines representing strokes)
    skeleton = cv2.ximgproc.thinning(binary)
    
    # Distance transform to find stroke width
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # Normalize to 0-1
    if dist_transform.max() > 0:
        dist_transform = dist_transform / dist_transform.max()
    
    # Create stroke mask - areas where we want to focus perturbations
    stroke_mask = dist_transform.copy()
    
    # Expand skeleton slightly to get junction points and end points
    kernel = np.ones((3,3), np.uint8)
    skeleton_dilated = cv2.dilate(skeleton, kernel, iterations=1)
    
    # Find junctions and endpoints (critical points in character structure)
    # These are often where small changes can affect recognition
    critical_points = cv2.bitwise_and(skeleton_dilated, skeleton)
    critical_points_mask = gaussian_filter(critical_points / 255.0, sigma=2.0)
    
    # Combine masks - higher weight on critical points
    combined_mask = 0.7 * stroke_mask + 0.3 * critical_points_mask
    combined_mask = gaussian_filter(combined_mask, sigma=1.0)
    
    # Normalize
    if combined_mask.max() > 0:
        combined_mask = combined_mask / combined_mask.max()
    
    # Expand mask to match image dimensions
    if len(combined_mask.shape) == 2 and len(img.shape) == 3:
        combined_mask = np.expand_dims(combined_mask, axis=2)
        combined_mask = np.repeat(combined_mask, 3, axis=2)
    
    # Best results tracking
    best_adv_img = img.copy()
    best_score = 0
    
    # Iteratively build up perturbation
    accumulated_perturbation = np.zeros_like(img)
    
    # Different stroke modification strategies
    strategies = [
        # Thin strokes: darken background pixels near edges
        lambda p: -np.abs(p) * 0.8,
        # Thicken strokes: lighten stroke pixels
        lambda p: np.abs(p) * 0.8,
        # Break connections: create small gaps
        lambda p: -np.abs(p) * np.sign(gaussian_filter(p, sigma=0.5))
    ]
    
    for iteration in range(iters):
        # Select a strategy
        strategy_idx = iteration % len(strategies)
        strategy_fn = strategies[strategy_idx]
        
        # Create a smooth perturbation
        perturbation = np.random.uniform(-epsilon/2, epsilon/2, img.shape).astype(np.float32)
        
        # Apply smoothing for less visual noise
        for c in range(perturbation.shape[2]):
            perturbation[:,:,c] = gaussian_filter(perturbation[:,:,c], sigma=0.5)
        
        # Apply the selected stroke modification strategy
        perturbation = strategy_fn(perturbation)
        
        # Apply mask to focus on stroke regions
        perturbation = perturbation * combined_mask
        
        # Update accumulated perturbation
        new_perturbation = accumulated_perturbation + perturbation
        new_perturbation = np.clip(new_perturbation, -epsilon, epsilon)
        
        # Apply the updated perturbation
        perturbed_img = np.clip(img + new_perturbation, 0, 1)
        
        # Evaluate the perturbed image
        perturbed_img_pil = Image.fromarray((perturbed_img * 255).astype(np.uint8))
        adv_pred = predictor.predict(perturbed_img_pil)
        
        # Calculate score based on edit distance
        import editdistance
        score = editdistance.eval(adv_pred, orig_pred)
        
        # Update best result if this perturbation works better
        if score > best_score:
            best_score = score
            best_adv_img = perturbed_img.copy()
            accumulated_perturbation = new_perturbation.copy()
            
            # Early stopping if we've significantly changed the prediction
            if score > len(orig_pred) * 0.4:
                break
    
    return best_adv_img

def character_transition_attack(img_np, predictor, epsilon=0.05, target_chars=None, iters=100):
    """
    Attack focusing on modifying characters to look like similar characters
    (e.g., 'o' to 'a', '0' to '8', etc.)
    
    Args:
        img_np: Original image (numpy array, values in [0,1])
        predictor: OCR predictor function
        epsilon: Maximum perturbation strength
        target_chars: Dict of character transitions to target (or None for automatic targeting)
        iters: Number of iterations
        
    Returns:
        Adversarial image (numpy array, values in [0,1])
    """
    from scipy.ndimage import gaussian_filter
    import cv2
    
    # Get original prediction
    orig_img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    orig_pred = predictor.predict(orig_img_pil)
    
    # Default character transition targets if none provided
    if target_chars is None:
        target_chars = {
            # Lowercase-letter confusions
            'o': 'a', 'l': 'i', 'i': 'j', 'c': 'e', 's': 'z', 'n': 'm', 'h': 'b', 'u': 'v',
            'g': 'q', 'r': 'n', 'v': 'y', 'a': 'o', 'z': 's',

        # Uppercase-letter confusions
            'O': 'Q', 'I': 'L', 'C': 'G', 'S': 'Z', 'B': 'E', 'D': 'O', 'Z': 'S',
            'M': 'N', 'P': 'R', 'U': 'V', 'V': 'Y', 'G': 'C', 'Q': 'O',

        # Digit confusions
            '0': 'O', '1': 'I', '2': 'Z', '3': '8', '4': 'A', '5': 'S', '6': 'G',
            '7': 'T', '8': 'B', '9': 'g',

        # Reversed mappings for symmetry
            'a': 'o', 'e': 'c', 'j': 'i', 'm': 'n', 'b': 'h', 'q': 'g',
            'y': 'v', 't': 'l', 'z': 's',

            'Q': 'O', 'L': 'I', 'G': 'C', 'Z': 'S', 'E': 'B', 'N': 'M',
            'R': 'P', 'Y': 'V',

            'O': '0', 'I': '1', 'Z': '2', '8': '3', 'A': '4', 'S': '5',
            'G': '6', 'T': '7', 'B': '8', 'g': '9',
        }

    
    # Convert to float32
    img = img_np.astype(np.float32)
    
    # Find text regions through binarization
    img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Try to segment characters to target specific ones
    # This is a simple approach; more advanced methods would use better segmentation
    kernel = np.ones((1, 5), np.uint8)  # Horizontal kernel to separate characters
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # Connected component analysis to find potential characters
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded)
    
    # Create character location mask 
    char_mask = np.zeros_like(binary, dtype=np.float32)
    
    # Find probable locations of characters we want to target
    for i in range(1, num_labels):  # Skip background (0)
        # Only consider components that could be characters (filter out noise)
        if stats[i, cv2.CC_STAT_AREA] > 20:  # Minimum area threshold
            # Create a mask for this component
            component_mask = (labels == i).astype(np.float32)
            
            # Apply Gaussian blur to soften the mask edges
            component_mask = gaussian_filter(component_mask, sigma=1.0)
            
            # Add to character mask
            char_mask += component_mask
    
    # Normalize the mask
    if char_mask.max() > 0:
        char_mask = char_mask / char_mask.max()
    
    # Expand mask to match image dimensions
    if len(char_mask.shape) == 2 and len(img.shape) == 3:
        char_mask = np.expand_dims(char_mask, axis=2)
        char_mask = np.repeat(char_mask, 3, axis=2)
    
    # Best results tracking
    best_adv_img = img.copy()
    best_score = 0
    
    # Iteratively build up perturbation
    accumulated_perturbation = np.zeros_like(img)
    
    # Structuring elements for morphological operations in perturbation
    structural_elements = [
        # Horizontal line (to turn 'c' into 'e', etc.)
        np.ones((1, 3), np.float32),
        # Vertical line (to turn 'o' into 'p', etc.)
        np.ones((3, 1), np.float32),
        # Small dot (to turn 'i' into 'j', etc.)
        np.ones((2, 2), np.float32),
        # Diagonal (to turn '/' into 'X', etc.)
        np.eye(3, dtype=np.float32),
        # Inverse diagonal
        np.flip(np.eye(3), axis=0).astype(np.float32)
    ]
    
    for iteration in range(iters):
        # Create perturbation based on structural elements
        perturbation = np.zeros_like(img)
        
        # Select a structural element
        struct_idx = iteration % len(structural_elements)
        struct_element = structural_elements[struct_idx]
        
        # Random magnitude and sign
        magnitude = epsilon * (0.5 + 0.5 * np.random.random())
        sign = 1 if np.random.random() > 0.5 else -1
        
        # Apply the structural element as a convolution kernel
        for c in range(perturbation.shape[2]):
            # Random offset for placement
            h_offset = np.random.randint(0, img.shape[0] - struct_element.shape[0])
            w_offset = np.random.randint(0, img.shape[1] - struct_element.shape[1])
            
            # Place the structured perturbation
            perturbation[
                h_offset:h_offset+struct_element.shape[0], 
                w_offset:w_offset+struct_element.shape[1], 
                c
            ] = sign * magnitude * struct_element
        
        # Apply Gaussian filter to make the perturbation smoother
        for c in range(perturbation.shape[2]):
            perturbation[:,:,c] = gaussian_filter(perturbation[:,:,c], sigma=0.5)
        
        # Apply character mask to focus perturbation on text regions
        perturbation = perturbation * char_mask
        
        # Update accumulated perturbation
        new_perturbation = accumulated_perturbation + perturbation
        new_perturbation = np.clip(new_perturbation, -epsilon, epsilon)
        
        # Apply the updated perturbation
        perturbed_img = np.clip(img + new_perturbation, 0, 1)
        
        # Evaluate the perturbed image
        perturbed_img_pil = Image.fromarray((perturbed_img * 255).astype(np.uint8))
        adv_pred = predictor.predict(perturbed_img_pil)
        
        # Calculate score based on edit distance
        import editdistance
        score = editdistance.eval(adv_pred, orig_pred)
        
        # Update best result if this perturbation works better
        if score > best_score:
            best_score = score
            best_adv_img = perturbed_img.copy()
            accumulated_perturbation = new_perturbation.copy()
    
    return best_adv_img

def position_aware_stroke_attack(img_np, predictor, epsilon=0.05, iters=100):
    """
    Advanced adversarial attack that uses OCR model's attention to determine precise
    text positions and applies targeted stroke modifications
    
    Args:
        img_np: Original image (numpy array, values in [0,1])
        predictor: OCR predictor function
        epsilon: Maximum perturbation strength
        iters: Number of iterations
        
    Returns:
        Adversarial image (numpy array, values in [0,1])
    """
    from scipy.ndimage import gaussian_filter
    import cv2
    
    # Get original prediction and attention map from the model
    orig_img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    orig_pred = predictor.predict(orig_img_pil)
    
    # Create a tensor for getting model attention (if supported)
    img_tensor = process_input(orig_img_pil, predictor.config['dataset']['image_height'], 
                             predictor.config['dataset']['image_min_width'], 
                             predictor.config['dataset']['image_max_width'])
    
    # Extract character positions using model attention or traditional CV methods
    try:
        # Try to get attention maps from the model (if implemented)
        attention_maps = predictor.get_attention_maps(img_tensor) if hasattr(predictor, 'get_attention_maps') else None
        
        if attention_maps is not None:
            # Process attention maps to create position mask
            position_mask = attention_maps.mean(0).cpu().numpy()
            position_mask = cv2.resize(position_mask, (img_np.shape[1], img_np.shape[0]))
        else:
            # Fall back to CV-based character detection
            position_mask = extract_character_positions_cv(img_np)
    except:
        # Fall back to CV-based character detection
        position_mask = extract_character_positions_cv(img_np)
    
    # Convert to float32
    img = img_np.astype(np.float32)
    
    # Best results tracking
    best_adv_img = img.copy()
    best_score = 0
    
    # Apply multiple stroke modification strategies
    stroke_strategies = [
        # Strategy 1: Disrupt character contours
        lambda region: apply_contour_disruption(region, epsilon),
        # Strategy 2: Introduce structural confusions (e.g., connect/disconnect parts)
        lambda region: apply_structural_confusion(region, epsilon),
        # Strategy 3: Target critical points (junctions/endpoints)
        lambda region: target_critical_points(region, epsilon),
        # Strategy 4: Add fine stroke-level noise aligned with character orientation
        lambda region: add_stroke_aligned_noise(region, epsilon)
    ]
    
    # Expand mask to match image dimensions
    if len(position_mask.shape) == 2 and len(img.shape) == 3:
        position_mask = np.expand_dims(position_mask, axis=2)
        position_mask = np.repeat(position_mask, 3, axis=2)
    
    # Normalize position mask
    position_mask = position_mask / (position_mask.max() + 1e-8)
    
    # Apply position-aware perturbations
    accumulated_perturbation = np.zeros_like(img)
    
    for iteration in range(iters):
        # Select strategy
        strategy_idx = iteration % len(stroke_strategies)
        strategy = stroke_strategies[strategy_idx]
        
        # Create perturbation
        perturbation = np.zeros_like(img)
        
        # Apply strategy to the entire image (the strategy will focus on stroke regions)
        perturbation = strategy(img)
        
        # Mask the perturbation to focus on character positions
        perturbation = perturbation * position_mask * epsilon
        
        # Apply smoothing for less visual detectability
        for c in range(perturbation.shape[2]):
            perturbation[:,:,c] = gaussian_filter(perturbation[:,:,c], sigma=0.5)
        
        # Update accumulated perturbation
        new_perturbation = accumulated_perturbation + perturbation
        new_perturbation = np.clip(new_perturbation, -epsilon, epsilon)
        
        # Apply the updated perturbation
        perturbed_img = np.clip(img + new_perturbation, 0, 1)
        
        # Check prediction
        perturbed_img_pil = Image.fromarray((perturbed_img * 255).astype(np.uint8))
        adv_pred = predictor.predict(perturbed_img_pil)
        
        # Calculate score - we want different prediction but minimal change
        import editdistance
        score = editdistance.eval(adv_pred, orig_pred)
        
        # Update best result if this perturbation works better
        if score > best_score:
            best_score = score
            best_adv_img = perturbed_img.copy()
            accumulated_perturbation = new_perturbation.copy()
            
            # Early stopping if we've significantly changed the prediction
            if score > len(orig_pred) * 0.5:
                break
    
    return best_adv_img

def extract_character_positions_cv(img_np):
    """Extract character positions using computer vision techniques"""
    import cv2
    
    # Convert to grayscale
    img_gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding to handle varying illumination
    binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Create character mask
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # Find connected components (potential characters)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated)
    
    # Create position mask
    position_mask = np.zeros_like(img_gray, dtype=np.float32)
    
    # Consider components that could be characters
    for i in range(1, num_labels):  # Skip background (0)
        if stats[i, cv2.CC_STAT_AREA] > 10:  # Minimum area threshold
            # Create component mask
            component_mask = (labels == i).astype(np.float32)
            
            # Add to position mask
            position_mask += component_mask
    
    # Apply Gaussian blur to create a smooth mask
    position_mask = cv2.GaussianBlur(position_mask, (5, 5), 0)
    
    return position_mask

def apply_contour_disruption(img, epsilon):
    """Disrupt character contours"""
    from scipy.ndimage import gaussian_filter
    
    # Create noise pattern
    noise = np.random.uniform(-1, 1, img.shape).astype(np.float32)
    
    # Apply Gaussian smoothing to create coherent patterns
    for c in range(noise.shape[2]):
        noise[:,:,c] = gaussian_filter(noise[:,:,c], sigma=0.7)
    
    # Normalize
    noise = noise / (np.sqrt(np.mean(noise**2)) + 1e-8)
    
    return noise * epsilon

def apply_structural_confusion(img, epsilon):
    """Introduce structural confusions like connecting/disconnecting strokes"""
    from scipy.ndimage import gaussian_filter
    import cv2
    
    # Convert to grayscale
    img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Get edges - potential character boundaries
    edges = cv2.Canny(img_gray, 100, 200) / 255.0
    
    # Create directional perturbations
    h_kernel = np.array([[-1, 2, -1]], dtype=np.float32)
    v_kernel = np.array([[-1], [2], [-1]], dtype=np.float32)
    
    # Apply directional kernels
    h_perturbation = cv2.filter2D(edges, -1, h_kernel)
    v_perturbation = cv2.filter2D(edges, -1, v_kernel)
    
    # Combine directional perturbations
    combined = (h_perturbation + v_perturbation) / 2.0
    
    # Smooth the perturbation
    combined = gaussian_filter(combined, sigma=0.5)
    
    # Normalize
    combined = combined / (np.max(np.abs(combined)) + 1e-8)
    
    # Expand to 3 channels
    perturbation = np.expand_dims(combined, axis=2)
    perturbation = np.repeat(perturbation, 3, axis=2)
    
    return perturbation * epsilon

def target_critical_points(img, epsilon):
    """Target junction points and endpoints of character strokes"""
    import cv2
    
    # Convert to grayscale
    img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Threshold to get binary image
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Get skeleton
    skeleton = cv2.ximgproc.thinning(binary)
    
    # Find potential junction points using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(skeleton, kernel, iterations=1)
    potential_junctions = cv2.bitwise_and(dilated, skeleton)
    
    # Apply focused perturbations at these points
    critical_points_mask = potential_junctions / 255.0
    
    # Create perturbation focused on these points
    perturbation = np.random.uniform(-1, 1, img.shape).astype(np.float32)
    
    # Expand critical points mask if needed
    if len(critical_points_mask.shape) == 2 and len(perturbation.shape) == 3:
        critical_points_mask = np.expand_dims(critical_points_mask, axis=2)
        critical_points_mask = np.repeat(critical_points_mask, 3, axis=2)
    
    # Apply mask to focus perturbation on critical points
    perturbation = perturbation * critical_points_mask
    
    # Normalize
    perturbation = perturbation / (np.sqrt(np.mean(perturbation**2)) + 1e-8)
    
    return perturbation * epsilon

def add_stroke_aligned_noise(img, epsilon):
    """Add noise aligned with stroke orientation"""
    import cv2
    
    # Convert to grayscale
    img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Calculate gradient
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and direction
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize magnitude
    magnitude = magnitude / (magnitude.max() + 1e-8)
    
    # Create aligned noise
    perturbation = np.random.uniform(-1, 1, img.shape).astype(np.float32)
    
    # Weight by gradient magnitude to focus on stroke edges
    if len(magnitude.shape) == 2 and len(perturbation.shape) == 3:
        magnitude = np.expand_dims(magnitude, axis=2)
        magnitude = np.repeat(magnitude, 3, axis=2)
    
    perturbation = perturbation * magnitude
    
    # Normalize
    perturbation = perturbation / (np.sqrt(np.mean(perturbation**2)) + 1e-8)
    
    return perturbation * epsilon
def combined_stroke_attack(img_np, predictor, epsilon=0.05, iters=150):
    """
    Enhanced version that combines multiple strategies with position awareness
    
    Args:
        img_np: Original image (numpy array, values in [0,1])
        predictor: OCR predictor function
        epsilon: Maximum perturbation strength
        iters: Number of iterations
        
    Returns:
        Adversarial image (numpy array, values in [0,1])
    """
    # Use our new position-aware attack
    adv_img = position_aware_stroke_attack(img_np, predictor, epsilon=epsilon, iters=iters)
    pred = predictor.predict(Image.fromarray((adv_img * 255).astype(np.uint8)))
    
    # Get original prediction
    orig_pred = predictor.predict(Image.fromarray((img_np * 255).astype(np.uint8)))
    
    # Calculate score
    import editdistance
    score = editdistance.eval(pred, orig_pred)
    
    # If the position-aware attack didn't work well, try other methods
    if score < len(orig_pred) * 0.2:
        # Try other strategies
        adv_img1 = structure_aware_attack(img_np, predictor, epsilon=epsilon*0.8, iters=iters//3)
        pred1 = predictor.predict(Image.fromarray((adv_img1 * 255).astype(np.uint8)))
        
        adv_img2 = stroke_modification_attack(img_np, predictor, epsilon=epsilon*0.8, iters=iters//3)
        pred2 = predictor.predict(Image.fromarray((adv_img2 * 255).astype(np.uint8)))
        
        # Choose the best result
        score1 = editdistance.eval(pred1, orig_pred)
        score2 = editdistance.eval(pred2, orig_pred)
        
        if score1 > score and score1 > score2:
            return adv_img1
        elif score2 > score and score2 > score1:
            return adv_img2
    
    return adv_img

# Integrate with the main function
def apply_pgd_attack(img_np, predictor, epsilon=0.05, alpha=0.01, iters=100):
    """
    Modified to use the position-aware stroke attack
    """
    return combined_stroke_attack(img_np, predictor, epsilon=epsilon, iters=iters)

def load_validation_data(val_annotation_file):
    """
    Load validation data from the annotation file
    
    Args:
        val_annotation_file: Path to the validation annotation file
        
    Returns:
        Dictionary of image paths and labels
    """
    images = {}
    labels = {}
    
    with open(val_annotation_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"Warning: Malformed line: {line}")
                continue
                
            img_path, text = parts
            img_name = os.path.basename(img_path)
            images[img_name] = img_path
            labels[img_name] = text
    
    print(f"Loaded {len(images)} images and labels from validation file")
    return images, labels

# New evaluation metrics functions
def compute_cer(pred_text, gt_text):
    """Calculate Character Error Rate"""
    dist = editdistance.eval(pred_text, gt_text)
    return dist / max(len(gt_text), 1)

def compute_wer(pred_text, gt_text):
    """Calculate Word Error Rate"""
    pred_words = pred_text.split()
    gt_words = gt_text.split()
    dist = editdistance.eval(pred_words, gt_words)
    return dist / max(len(gt_words), 1)

def has_error(pred_text, gt_text):
    """Check if there's any error in the prediction"""
    return pred_text != gt_text

def compute_char_accuracy(pred_text, gt_text):
    """Calculate character-level accuracy"""
    if len(gt_text) == 0:
        return 1.0 if len(pred_text) == 0 else 0.0
    
    dist = editdistance.eval(pred_text, gt_text)
    correct_chars = max(len(gt_text) - dist, 0)
    return correct_chars / len(gt_text)

def attack_and_evaluate(predictor, image_base_dir, test_images, gt_labels, config, save_dir="attack_evaluation", num_samples=270, 
                        epsilon=0.05, alpha=0.01, iters=200, apply_attack_func=None):
    """Attack images, evaluate metrics and save results"""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)
    
    # Results collection
    results = {
        'image_name': [],
        'ground_truth': [],
        'orig_prediction': [],
        'adv_prediction': [],
        'orig_cer': [],
        'adv_cer': [],
        'orig_wer': [],
        'adv_wer': [],
        'orig_char_acc': [],
        'adv_char_acc': [],
        'orig_has_error': [],
        'adv_has_error': []
    }
    
    # Open result file
    with open(os.path.join(save_dir, 'attack_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write("Image\tGround Truth\tOriginal Prediction\tAttacked Prediction\t" +
                "Orig CER\tAttack CER\tOrig WER\tAttack WER\tOrig Char Acc\tAttack Char Acc\n")
        
        saved_samples = 0
        
        for idx, img_name in enumerate(tqdm(test_images)):
            try:
                # Get the relative path and construct full path
                rel_path = test_images[img_name]
                img_path = os.path.join(image_base_dir, rel_path)
                
                # Check if file exists
                if not os.path.exists(img_path):
                    print(f"Warning: Image file not found: {img_path}")
                    continue
                
                gt_text = gt_labels[img_name]
                
                # Process image
                img_pil, img_tensor, img_np = process_image_for_attack(img_path, config)
                
                # Get original prediction
                orig_pred = predict_text(predictor, img_pil)
                
                # Apply PGD attack
                if apply_attack_func:
                        adv_img_np = apply_attack_func(img_np, predictor)
                else:
                        adv_img_np = apply_pgd_attack(img_np, predictor, epsilon=epsilon, alpha=alpha, iters=iters)
                
                # Convert back to PIL image for prediction
                adv_img_pil = Image.fromarray((adv_img_np * 255).astype(np.uint8))
                
                # Get prediction on adversarial image
                adv_pred = predict_text(predictor, adv_img_pil)
                
                # Calculate metrics
                orig_cer = compute_cer(orig_pred, gt_text)
                adv_cer = compute_cer(adv_pred, gt_text)
                orig_wer = compute_wer(orig_pred, gt_text)
                adv_wer = compute_wer(adv_pred, gt_text)
                orig_char_acc = compute_char_accuracy(orig_pred, gt_text)
                adv_char_acc = compute_char_accuracy(adv_pred, gt_text)
                orig_has_error = has_error(orig_pred, gt_text)
                adv_has_error = has_error(adv_pred, gt_text)
                
                # Store results
                results['image_name'].append(img_name)
                results['ground_truth'].append(gt_text)
                results['orig_prediction'].append(orig_pred)
                results['adv_prediction'].append(adv_pred)
                results['orig_cer'].append(orig_cer)
                results['adv_cer'].append(adv_cer)
                results['orig_wer'].append(orig_wer)
                results['adv_wer'].append(adv_wer)
                results['orig_char_acc'].append(orig_char_acc)
                results['adv_char_acc'].append(adv_char_acc)
                results['orig_has_error'].append(int(orig_has_error))
                results['adv_has_error'].append(int(adv_has_error))
                
                # Write to results file
                f.write(f"{img_name}\t{gt_text}\t{orig_pred}\t{adv_pred}\t"
                        f"{orig_cer:.4f}\t{adv_cer:.4f}\t{orig_wer:.4f}\t{adv_wer:.4f}\t"
                        f"{orig_char_acc:.4f}\t{adv_char_acc:.4f}\n")
                
                # Save sample visualizations
                if saved_samples < num_samples:
                    # Calculate perturbation
                    perturbation = np.abs(img_np - adv_img_np)
                    perturbation = np.clip(perturbation * 10, 0, 1)  # Amplify for visibility
                    
                    # Create visualization
                    plt.figure(figsize=(15, 5))
                    
                    plt.subplot(1, 3, 1)
                    plt.imshow(img_pil)
                    plt.title(f"Original\nPred: {orig_pred}\nCER: {orig_cer:.4f}, WER: {orig_wer:.4f}")
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 2)
                    plt.imshow(adv_img_pil)
                    plt.title(f"Adversarial\nPred: {adv_pred}\nCER: {adv_cer:.4f}, WER: {adv_wer:.4f}")
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 3)
                    plt.imshow(perturbation)
                    plt.title(f"Perturbation (x10)\nGT: {gt_text}")
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'visualizations', f"sample_{saved_samples}.png"), dpi=150)
                    plt.close('all')
                    
                    # Also save adversarial image
                    adv_img_pil.save(os.path.join(save_dir, 'visualizations', f"adv_{saved_samples}.png"))
                    
                    saved_samples += 1
                    
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue
    
    # Calculate overall metrics
    df = pd.DataFrame(results)
    
    # Calculate and save summary metrics
    summary = {
        'total_samples': len(df),
        'orig_avg_cer': df['orig_cer'].mean(),
        'adv_avg_cer': df['adv_cer'].mean(),
        'orig_avg_wer': df['orig_wer'].mean(),
        'adv_avg_wer': df['adv_wer'].mean(),
        'orig_avg_char_acc': df['orig_char_acc'].mean(),
        'adv_avg_char_acc': df['adv_char_acc'].mean(),
        'orig_ser': df['orig_has_error'].mean(),  # SER = proportion of sequences with errors
        'adv_ser': df['adv_has_error'].mean(),
        'attack_success_rate': (df['adv_prediction'] != df['orig_prediction']).mean()
    }
    
    # Save summary to file
    with open(os.path.join(save_dir, 'summary_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Total samples: {summary['total_samples']}\n")
        f.write(f"Original CER: {summary['orig_avg_cer']:.4f} | Adversarial CER: {summary['adv_avg_cer']:.4f}\n")
        f.write(f"Original WER: {summary['orig_avg_wer']:.4f} | Adversarial WER: {summary['adv_avg_wer']:.4f}\n")
        f.write(f"Original Char Acc: {summary['orig_avg_char_acc']:.4f} | Adversarial Char Acc: {summary['adv_avg_char_acc']:.4f}\n")
        f.write(f"Original SER: {summary['orig_ser']:.4f} | Adversarial SER: {summary['adv_ser']:.4f}\n")
        f.write(f"Attack Success Rate: {summary['attack_success_rate']:.4f}\n")
    
    # Save detailed results as CSV
    df.to_csv(os.path.join(save_dir, 'detailed_metrics.csv'), index=False)
    
    # Create visualization of metrics comparison
    plot_metrics_comparison(summary, save_dir)
    
    # Create distribution plots
    plot_metric_distributions(df, save_dir)
    
    return summary, df

def plot_metrics_comparison(summary, save_dir):
    """Create bar charts comparing before/after metrics"""
    # Prepare data for plotting
    metrics = ['CER', 'WER', 'SER', 'Char Accuracy']
    before_vals = [
        summary['orig_avg_cer'],
        summary['orig_avg_wer'],
        summary['orig_ser'],
        summary['orig_avg_char_acc']
    ]
    after_vals = [
        summary['adv_avg_cer'],
        summary['adv_avg_wer'],
        summary['adv_ser'],
        summary['adv_avg_char_acc']
    ]
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Create the bar groups
    plt.bar(x - width/2, before_vals, width, label='Before Attack', color='royalblue')
    plt.bar(x + width/2, after_vals, width, label='After Attack', color='firebrick')
    
    # Add labels and formatting
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Values', fontsize=14)
    plt.title('OCR Performance Before and After Adversarial Attack', fontsize=16)
    plt.xticks(x, metrics, fontsize=12)
    plt.ylim(0, 1.0)
    
    # Add value labels on the bars
    for i, v in enumerate(before_vals):
        plt.text(i - width/2, v + 0.03, f'{v:.3f}', ha='center', fontsize=10)
    
    for i, v in enumerate(after_vals):
        plt.text(i + width/2, v + 0.03, f'{v:.3f}', ha='center', fontsize=10)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300)
    plt.close('all')
    
    # Create a second plot focusing on attack success
    plt.figure(figsize=(8, 6))
    
    success_rate = summary['attack_success_rate']
    plt.bar(['Attack Success Rate'], [success_rate], color='darkred', width=0.5)
    plt.ylim(0, 1.0)
    plt.title('Adversarial Attack Success Rate', fontsize=16)
    plt.text(0, success_rate + 0.03, f'{success_rate:.3f}', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'attack_success_rate.png'), dpi=300)
    plt.close('all')

def plot_metric_distributions(df, save_dir):
    """Create distribution plots showing how metrics changed"""
    # Set up the figure for the distributions
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # CER distribution
    sns.histplot(df['orig_cer'], color='blue', alpha=0.5, label='Original', ax=axs[0, 0])
    sns.histplot(df['adv_cer'], color='red', alpha=0.5, label='Adversarial', ax=axs[0, 0])
    axs[0, 0].set_title('Character Error Rate (CER) Distribution')
    axs[0, 0].set_xlabel('CER')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].legend()
    
    # WER distribution
    sns.histplot(df['orig_wer'], color='blue', alpha=0.5, label='Original', ax=axs[0, 1])
    sns.histplot(df['adv_wer'], color='red', alpha=0.5, label='Adversarial', ax=axs[0, 1])
    axs[0, 1].set_title('Word Error Rate (WER) Distribution')
    axs[0, 1].set_xlabel('WER')
    axs[0, 1].set_ylabel('Count')
    axs[0, 1].legend()
    
    # Character Accuracy distribution
    sns.histplot(df['orig_char_acc'], color='blue', alpha=0.5, label='Original', ax=axs[1, 0])
    sns.histplot(df['adv_char_acc'], color='red', alpha=0.5, label='Adversarial', ax=axs[1, 0])
    axs[1, 0].set_title('Character Accuracy Distribution')
    axs[1, 0].set_xlabel('Character Accuracy')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].legend()
    
    # CER change scatter plot
    axs[1, 1].scatter(df['orig_cer'], df['adv_cer'], alpha=0.5)
    axs[1, 1].set_title('CER Before vs. After Attack')
    axs[1, 1].set_xlabel('Original CER')
    axs[1, 1].set_ylabel('Adversarial CER')
    # Add diagonal line (y=x)
    min_val = min(df['orig_cer'].min(), df['adv_cer'].min())
    max_val = max(df['orig_cer'].max(), df['adv_cer'].max())
    axs[1, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metric_distributions.png'), dpi=300)
    plt.close('all')

def main():
    # Configuration
    checkpoint_path = "/root/OCR/weights/transformerocr.pth"
    validation_file = "/root/OCR/converted_data/val_annotation.txt"
    data_base_dir = "/root/OCR/converted_data"  # Base directory for images
    save_dir = "thua"
    
    # Attack parameters - optimized for less visible perturbations
    epsilon = 0.05      # Reduced maximum perturbation (from 0.2)
    alpha = 0.01      # Smaller step size for finer control
    iters = 100        # Keep high number of iterations for optimization
    
    # Load model
    predictor, config = load_model(checkpoint_path)
    
    # Load validation data
    image_paths, labels = load_validation_data(validation_file)
    
    print(f"Found {len(image_paths)} images with labels from validation file")
    
    # Choose attack method
    attack_method = "stroke"  # Options: "targeted_perceptual", "frequency", "color_shift", "improved_pgd", "ensemble", "stroke"
    
    # Create a wrapper function for the selected attack method
    def attack_wrapper(img_np):
        if attack_method == "targeted_perceptual":
            return targeted_perceptual_attack(img_np, predictor, 
                                             epsilon=epsilon, 
                                             alpha=alpha, 
                                             iters=iters, 
                                             perceptual_weight=0.8)
        elif attack_method == "frequency":
            return frequency_domain_attack(img_np, predictor, 
                                          epsilon=epsilon*2,  # Frequency domain can handle larger epsilon
                                          iters=iters, 
                                          focus_factor=0.7)
        elif attack_method == "color_shift":
            return color_shift_attack(img_np, predictor, 
                                     epsilon=epsilon, 
                                     iters=iters//2, 
                                     channels_weight=(0.2, 0.5, 0.3))
        elif attack_method == "improved_pgd":
            return apply_pgd_attack(img_np, predictor,
                                   epsilon=epsilon, 
                                   alpha=alpha, 
                                   iters=iters)
        elif attack_method == "ensemble":
            return ensemble_advanced_attack(img_np, predictor, 
                                           epsilon=epsilon, 
                                           iters=iters)
        elif attack_method == "stroke":
            return combined_stroke_attack(img_np, predictor, 
                                         epsilon=epsilon, 
                                         iters=iters)
        else:
            # Default to improved PGD
            return apply_pgd_attack(img_np, epsilon=epsilon, alpha=alpha, iters=iters)
    
    # Run attack and evaluation with the chosen method
    print(f"Running less visible adversarial attack using method: {attack_method}")
    summary, results_df = attack_and_evaluate(
        predictor=predictor,
        image_base_dir=data_base_dir,
        test_images=image_paths,
        gt_labels=labels,
        config=config,
        save_dir=f"{save_dir}_{attack_method}",
        num_samples=270,
        apply_attack_func=lambda img_np, predictor_unused=None: attack_wrapper(img_np)
    )
    
    print(f"\nAttack and Evaluation Complete. Results saved to {save_dir}_{attack_method}")
    print(f"Attack Success Rate: {summary['attack_success_rate']:.2%}")
    print("\nMetrics Summary:")
    print(f"CER: {summary['orig_avg_cer']:.4f} → {summary['adv_avg_cer']:.4f} (change: {summary['adv_avg_cer'] - summary['orig_avg_cer']:.4f})")
    print(f"WER: {summary['orig_avg_wer']:.4f} → {summary['adv_avg_wer']:.4f} (change: {summary['adv_avg_wer'] - summary['orig_avg_wer']:.4f})")
    print(f"Char Accuracy: {summary['orig_avg_char_acc']:.4f} → {summary['adv_avg_char_acc']:.4f} (change: {summary['adv_avg_char_acc'] - summary['orig_avg_char_acc']:.4f})")
    print(f"SER: {summary['orig_ser']:.4f} → {summary['adv_ser']:.4f} (change: {summary['adv_ser'] - summary['orig_ser']:.4f})")

if __name__ == "__main__":
    main()
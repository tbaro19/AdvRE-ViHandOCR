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

def targeted_perceptual_attack(img_np, predictor, epsilon=0.2, alpha=0.005, iters=200, perceptual_weight=0.8):
    """
    Apply adversarial attack focusing on less perceptible changes
    
    Args:
        img_np: Original image (numpy array, values in [0,1])
        predictor: OCR predictor
        epsilon: Maximum perturbation
        alpha: Step size
        iters: Number of iterations
        perceptual_weight: Weight for perceptual loss (higher = less visible changes)
        
    Returns:
        Adversarial image (numpy array, values in [0,1])
    """
    # Convert to float32
    img = img_np.astype(np.float32)
    
    # Get original prediction
    orig_img_pil = Image.fromarray((img * 255).astype(np.uint8))
    orig_pred = predictor.predict(orig_img_pil)
    
    # Initialize perturbation
    delta = np.zeros_like(img, dtype=np.float32)
    
    # Create an importance mask (focus on text regions and edges)
    # Use simple edge detection
    grayscale = np.mean(img, axis=2)
    from scipy.ndimage import gaussian_filter
    
    # Detect edges using gradient magnitude
    smoothed = gaussian_filter(grayscale, sigma=1.0)
    gx = np.gradient(smoothed, axis=1)
    gy = np.gradient(smoothed, axis=0)
    importance_mask = np.sqrt(gx**2 + gy**2)
    
    # Normalize to [0,1]
    importance_mask = importance_mask / (np.max(importance_mask) + 1e-8)
    
    # Weight towards text regions (higher values)
    importance_mask = importance_mask ** 0.5  # Apply power to increase contrast
    
    # For storing best perturbation
    best_delta = delta.copy()
    best_score = 0
    
    # PGD iterations with perceptual considerations
    for i in range(iters):
        # Try a small random perturbation
        noise = np.random.normal(0, alpha, img.shape).astype(np.float32)
        
        # Weight noise by importance mask to focus on text regions
        weighted_noise = noise * importance_mask[:, :, np.newaxis]
        
        # Try this perturbation
        candidate_delta = delta + weighted_noise
        
        # Project to epsilon-ball (constrain magnitude)
        candidate_delta = np.clip(candidate_delta, -epsilon, epsilon)
        
        # Apply spatial smoothing to make perturbation less visible
        from scipy.ndimage import gaussian_filter
        for c in range(3):
            candidate_delta[:,:,c] = gaussian_filter(candidate_delta[:,:,c], sigma=0.5)
        
        # Ensure image remains valid after perturbation
        perturbed = np.clip(img + candidate_delta, 0, 1)
        candidate_delta = perturbed - img
        
        # Convert to PIL for prediction
        perturbed_img = Image.fromarray((perturbed * 255).astype(np.uint8))
        adv_pred = predictor.predict(perturbed_img)
        
        # Calculate score based on prediction difference
        import editdistance
        score = editdistance.eval(adv_pred, orig_pred)
        
        # Also consider perceptual penalty (L2 norm of perturbation weighted by mask)
        # Lower values of delta where importance is low
        perceptual_penalty = np.mean((1 - importance_mask[:, :, np.newaxis]) * candidate_delta**2)
        
        # Combined score: higher edit distance is better, lower perceptual penalty is better
        combined_score = score - perceptual_weight * perceptual_penalty * 1000  # Scale perceptual penalty
        
        # Keep the best perturbation
        if combined_score > best_score and adv_pred != orig_pred:
            best_score = combined_score
            best_delta = candidate_delta.copy()
            
            # Print progress occasionally
            if i % 50 == 0:
                print(f"Iter {i}, Current edit distance: {score}, " 
                      f"Perceptual penalty: {perceptual_penalty:.6f}, "
                      f"Combined score: {combined_score:.2f}")
        
        # Update delta with momentum towards good directions
        if adv_pred != orig_pred:
            delta = delta + 0.8 * weighted_noise
            delta = np.clip(delta, -epsilon, epsilon)
    
    # Final perturbed image using best delta found
    adv_img = np.clip(img + best_delta, 0, 1)
    
    return adv_img

def frequency_domain_attack(img_np, predictor, epsilon=0.2, iters=200, focus_factor=0.7):
    """
    Perform adversarial attack in the frequency domain to reduce perceptibility
    
    Args:
        img_np: Original image (numpy array, values in [0,1])
        predictor: OCR predictor
        epsilon: Maximum perturbation strength in frequency domain
        iters: Number of iterations
        focus_factor: Balance between high and low frequencies (0-1)
                     Higher values focus more on mid frequencies
        
    Returns:
        Adversarial image (numpy array, values in [0,1])
    """
    # Import necessary functions
    from scipy.fftpack import dct, idct
    
    # Get original prediction
    orig_img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    orig_pred = predictor.predict(orig_img_pil)
    
    # Convert to float32
    img = img_np.astype(np.float32)
    
    # Split image into channels
    channels = []
    for c in range(img.shape[2]):
        channels.append(img[:,:,c])
    
    # Apply DCT to each channel
    dct_channels = []
    for c in channels:
        dct_channels.append(dct(dct(c.T, norm='ortho').T, norm='ortho'))
    
    # Create a frequency mask that focuses on mid frequencies
    # (humans are less sensitive to changes in very high and very low frequencies)
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    
    # Distance from center (normalized to [0,1])
    center_h, center_w = h // 2, w // 2
    dist_from_center = np.sqrt((Y - center_h)**2 + (X - center_w)**2)
    dist_from_center = dist_from_center / np.max(dist_from_center)
    
    # Create band-pass like mask that emphasizes mid-frequencies
    # This creates a mask that peaks at mid-frequencies
    mid_freq_mask = 1.0 - np.abs(2.0 * dist_from_center - focus_factor) ** 2
    mid_freq_mask = mid_freq_mask ** 0.5  # Sharpen the mask
    
    # Iteratively search for good perturbations
    best_adv_img = img.copy()
    best_score = 0
    
    for i in range(iters):
        # Create perturbation in frequency domain
        dct_perturbations = []
        for _ in range(len(channels)):
            # Random perturbation with mid-frequency focus
            perturbation = np.random.uniform(-epsilon, epsilon, (h, w)) * mid_freq_mask
            dct_perturbations.append(perturbation)
        
        # Apply perturbation to DCT coefficients
        perturbed_dct = []
        for j in range(len(dct_channels)):
            perturbed_dct.append(dct_channels[j] + dct_perturbations[j])
        
        # Convert back to spatial domain
        perturbed_channels = []
        for c in perturbed_dct:
            channel = idct(idct(c, norm='ortho').T, norm='ortho').T
            perturbed_channels.append(channel)
        
        # Combine channels and clip to valid range
        perturbed_img = np.stack(perturbed_channels, axis=2)
        perturbed_img = np.clip(perturbed_img, 0, 1)
        
        # Check if attack is successful
        perturbed_img_pil = Image.fromarray((perturbed_img * 255).astype(np.uint8))
        adv_pred = predictor.predict(perturbed_img_pil)
        
        # Calculate score based on edit distance
        import editdistance
        score = editdistance.eval(adv_pred, orig_pred)
        
        # Calculate visual distortion metric (L2 norm)
        distortion = np.mean((perturbed_img - img) ** 2)
        
        # Combined score: balance between effectiveness and visual quality
        combined_score = score - distortion * 100  # Penalize visual distortion
        
        if combined_score > best_score and adv_pred != orig_pred:
            best_score = combined_score
            best_adv_img = perturbed_img.copy()
            
            # Print progress occasionally
            if i % 50 == 0:
                print(f"Iter {i}, Edit distance: {score}, Distortion: {distortion:.6f}")
    
    return best_adv_img

def color_shift_attack(img_np, predictor, epsilon=0.2, iters=100, channels_weight=(0.2, 0.5, 0.3)):
    """
    Apply adversarial attack using subtle color shifts
    
    Args:
        img_np: Original image (numpy array, values in [0,1])
        predictor: OCR predictor
        epsilon: Maximum perturbation
        iters: Number of iterations
        channels_weight: Weights for RGB channels (R,G,B)
        
    Returns:
        Adversarial image (numpy array, values in [0,1])
    """
    # Convert to float32
    img = img_np.astype(np.float32)
    
    # Get original prediction
    orig_img_pil = Image.fromarray((img * 255).astype(np.uint8))
    orig_pred = predictor.predict(orig_img_pil)
    
    # Initialize perturbation
    best_adv_img = img.copy()
    best_score = 0
    
    # Attack iterations
    for i in range(iters):
        # Create color shift perturbation
        # Focus more on color shifts than brightness changes
        perturbation = np.random.uniform(-epsilon, epsilon, (1, 1, 3)) * np.array(channels_weight)
        
        # Apply color shift (broadcast to entire image)
        perturbed = img + perturbation
        
        # Ensure image remains valid
        perturbed = np.clip(perturbed, 0, 1)
        
        # Check prediction
        perturbed_img_pil = Image.fromarray((perturbed * 255).astype(np.uint8))
        adv_pred = predictor.predict(perturbed_img_pil)
        
        # Calculate score based on edit distance
        import editdistance
        score = editdistance.eval(adv_pred, orig_pred)
        
        if score > best_score and adv_pred != orig_pred:
            best_score = score
            best_adv_img = perturbed.copy()
    
    return best_adv_img

def ensemble_advanced_attack(img_np, predictor, epsilon=0.2, iters=100):
    """
    Apply an ensemble of different advanced attack methods
    
    Args:
        img_np: Original image (numpy array, values in [0,1])
        predictor: OCR predictor function
        epsilon: Maximum perturbation strength
        iters: Number of iterations
        
    Returns:
        Adversarial image (numpy array, values in [0,1])
    """
    # Get original prediction
    orig_img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    orig_pred = predictor.predict(orig_img_pil)
    
    # Try different attack methods
    attack_methods = [
        # Method 1: Targeted perceptual attack
        lambda: targeted_perceptual_attack(img_np, predictor, 
                                           epsilon=epsilon, 
                                           alpha=epsilon/20, 
                                           iters=iters//2,
                                           perceptual_weight=0.8),
        # Method 2: Frequency domain attack
        lambda: frequency_domain_attack(img_np, predictor, 
                                       epsilon=epsilon*2, 
                                       iters=iters//2, 
                                       focus_factor=0.7),
        # Method 3: Color shift attack
        lambda: color_shift_attack(img_np, predictor, 
                                  epsilon=epsilon, 
                                  iters=iters//3, 
                                  channels_weight=(0.2, 0.5, 0.3))
    ]
    
    # Try each method and pick the best result
    best_adv_img = img_np.copy()
    best_score = 0
    
    for attack_func in attack_methods:
        try:
            # Apply this attack method
            adv_img = attack_func()
            
            # Evaluate result
            adv_img_pil = Image.fromarray((adv_img * 255).astype(np.uint8))
            adv_pred = predictor.predict(adv_img_pil)
            
            # Calculate visual difference
            diff = np.mean(np.abs(adv_img - img_np))
            
            # Calculate attack effectiveness
            import editdistance
            edit_dist = editdistance.eval(adv_pred, orig_pred)
            
            # Combined score: balance between effectiveness and visual quality
            score = edit_dist / (diff * 20 + 0.001)  # Higher is better
            
            if score > best_score and adv_pred != orig_pred:
                best_score = score
                best_adv_img = adv_img.copy()
                print(f"Found better adversarial example, edit distance: {edit_dist}, " 
                      f"visual diff: {diff:.6f}, score: {score:.2f}")
                
        except Exception as e:
            print(f"Attack method failed: {e}")
            continue
    
    return best_adv_img

def apply_pgd_attack(img_np, epsilon=0.2, alpha=0.005, iters=200):
    """
    Improved PGD attack with focus on imperceptible changes
    
    Args:
        img_np: Original image (numpy array, values in [0,1])
        epsilon: Maximum perturbation (reduced for less visible noise)
        alpha: Step size (reduced for finer control)
        iters: Number of iterations (increased for better optimization)
    
    Returns:
        Adversarial image (numpy array, values in [0,1])
    """
    # Convert to float32
    img = img_np.astype(np.float32)
    
    # Initialize with small random noise
    delta = np.random.uniform(-epsilon/2, epsilon/2, img.shape).astype(np.float32)
    delta = np.clip(delta, -epsilon, epsilon)
    
    # Better initialization: focus on edges where text likely exists
    from scipy.ndimage import gaussian_filter
    
    # Simple edge detection
    grayscale = np.mean(img, axis=2)
    blurred = gaussian_filter(grayscale, sigma=1.0)
    edges = np.abs(grayscale - blurred)
    
    # Normalize edge mask to [0,1]
    edge_mask = edges / (np.max(edges) + 1e-8)
    
    # Apply edge mask to initial perturbation
    delta = delta * edge_mask[:, :, np.newaxis]
    
    # PGD iterations with momentum for stability
    momentum = np.zeros_like(delta)
    
    # Decaying alpha for fine-tuning
    curr_alpha = alpha
    
    for i in range(iters):
        # Decay step size over time
        if i > iters // 2:
            curr_alpha = alpha * 0.5
        
        # Add random noise in the direction of gradient
        noise = np.random.normal(0, curr_alpha, img.shape).astype(np.float32)
        
        # Weight noise by edge mask to focus on text regions
        noise = noise * (0.2 + 0.8 * edge_mask[:, :, np.newaxis])
        
        # Apply momentum update
        momentum = 0.9 * momentum + noise
        
        # Update delta
        delta = delta + 0.8 * momentum
        
        # Project back to epsilon ball
        delta = np.clip(delta, -epsilon, epsilon)
        
        # Apply spatial smoothing for less visible noise
        from scipy.ndimage import gaussian_filter
        for c in range(3):
            delta[:,:,c] = gaussian_filter(delta[:,:,c], sigma=0.5)
        
        # Ensure the image remains valid
        perturbed = np.clip(img + delta, 0, 1)
        delta = perturbed - img
    
    # Final perturbed image
    adv_img = np.clip(img + delta, 0, 1)
    
    return adv_img

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
                        epsilon=0.2, alpha=0.01, iters=200, apply_attack_func=None):
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
                adv_img_np = apply_pgd_attack(img_np, epsilon=epsilon, alpha=alpha, iters=iters)
                
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
    save_dir = "attack_PGD_new_weight"
    
    # Attack parameters - optimized for less visible perturbations
    epsilon = 0.2      # Reduced maximum perturbation (from 0.2)
    alpha = 0.005      # Smaller step size for finer control
    iters = 200        # Keep high number of iterations for optimization
    
    # Load model
    predictor, config = load_model(checkpoint_path)
    
    # Load validation data
    image_paths, labels = load_validation_data(validation_file)
    
    print(f"Found {len(image_paths)} images with labels from validation file")
    
    # Choose attack method
    attack_method = "frequency"  # Options: "targeted_perceptual", "frequency", "color_shift", "improved_pgd", "ensemble"
    
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
            return apply_pgd_attack(img_np, 
                                   epsilon=epsilon, 
                                   alpha=alpha, 
                                   iters=iters)
        elif attack_method == "ensemble":
            return ensemble_advanced_attack(img_np, predictor, 
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
        apply_attack_func=attack_wrapper
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
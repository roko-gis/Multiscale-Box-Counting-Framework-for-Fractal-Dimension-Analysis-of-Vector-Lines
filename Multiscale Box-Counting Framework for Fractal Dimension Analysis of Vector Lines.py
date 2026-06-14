"""
Multiscale Box-Counting Framework for Fractal Dimension Analysis of Vector Lines
Box-Counting Method - Version 1.0 
Publication-Grade | Scale-Block Bootstrap | Median Ensemble | Arithmetic Mean
"""

from qgis.core import *
from qgis.utils import iface
import numpy as np
from scipy.stats import linregress, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os, tempfile, traceback, webbrowser, logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# =============================================
# Configuration
# =============================================
class BoxCountingConfig:
    def __init__(self):
        self.num_scales = 15
        self.min_divisions = 5
        self.max_div = 200
        self.buffer_factor = 1.08
        self.min_scale_points = 6
        self.optimal_scale_points = 10
        self.min_r2_threshold = 0.85
        self.slope_range = (0.7, 2.3)
        self.max_slope_variation = 0.15
        self.bootstrap_samples = 200
        self.grid_shifts = 3
        self.grid_rotations = 8
        self.fdr_alpha = 0.05
        self.top_windows = 8
        self.curvature_threshold = 0.1
        self.spearman_threshold = 0.93
        self.max_window_size_multiplier = 3
        self.window_step = 2
        self.block_bootstrap_scale_fraction = 0.25
        self.use_by_fdr = True
        self.seed = 42

class ConfidenceLevel:
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNRELIABLE = "UNRELIABLE"

# =============================================
# 1. Extent
# =============================================
def get_data_extent(layer):
    extent = layer.extent()
    if extent.isNull() or extent.width() <= 0 or extent.height() <= 0:
        extent = QgsRectangle()
        for f in layer.getFeatures():
            if f.hasGeometry():
                bbox = f.geometry().boundingBox()
                if not bbox.isNull() and bbox.width() > 0 and bbox.height() > 0:
                    extent.combineExtentWith(bbox)
    if extent.isNull() or extent.width() <= 0 or extent.height() <= 0:
        raise ValueError("Invalid extent")
    return extent.xMinimum(), extent.yMinimum(), extent.xMaximum(), extent.yMaximum()

# =============================================
# 2. Spatial Index
# =============================================
def build_spatial_index(layer):
    spatial_index = QgsSpatialIndex()
    feature_geoms = {}
    
    for f in layer.getFeatures():
        if f.hasGeometry():
            g = f.geometry()
            if not g.isEmpty():
                spatial_index.addFeature(f)
                feature_geoms[f.id()] = g
    
    if not feature_geoms:
        raise ValueError("No valid geometries")
    
    logger.info("Indexed " + str(len(feature_geoms)) + " features")
    return spatial_index, feature_geoms

# =============================================
# 3. Generate Scales
# =============================================
def generate_scales(config):
    try:
        candidates = np.unique(np.geomspace(config.min_divisions, config.max_div, 
                                           num=config.num_scales+5, dtype=int))
    except:
        candidates = np.unique(np.logspace(np.log10(config.min_divisions), 
                                          np.log10(config.max_div), 
                                          num=config.num_scales+5, dtype=int))
    if len(candidates) == 0:
        raise ValueError("No valid scale candidates")
    filtered = [candidates[0]]
    for div in candidates[1:]:
        if div / filtered[-1] >= 1.2:
            filtered.append(div)
    if len(filtered) < config.optimal_scale_points:
        try:
            filtered = np.unique(np.geomspace(config.min_divisions, config.max_div, 
                                             num=config.optimal_scale_points+3, dtype=int)).tolist()
        except:
            filtered = np.unique(np.logspace(np.log10(config.min_divisions), 
                                            np.log10(config.max_div), 
                                            num=config.optimal_scale_points+3, dtype=int)).tolist()
    result = np.array(filtered[:config.num_scales])
    result = result[result > 0]
    if len(result) < config.min_scale_points:
        raise ValueError("Only " + str(len(result)) + " scales")
    return result

# =============================================
# 4. DETERMINISTIC SO(2) Rotation Sampling
# =============================================
def get_deterministic_rotations(n_rotations):
    return np.linspace(0, np.pi/2, n_rotations, endpoint=False)

def rotate_point(x, y, cx, cy, angle):
    dx, dy = x - cx, y - cy
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return cx + dx*cos_a - dy*sin_a, cy + dx*sin_a + dy*cos_a

def get_rotated_extent(x_min, y_min, x_max, y_max, angle):
    cx, cy = (x_min+x_max)/2, (y_min+y_max)/2
    corners = [(x_min,y_min), (x_max,y_min), (x_min,y_max), (x_max,y_max)]
    rotated = [rotate_point(x, y, cx, cy, angle) for x, y in corners]
    xs, ys = [p[0] for p in rotated], [p[1] for p in rotated]
    return min(xs), min(ys), max(xs), max(ys)

# =============================================
# 5. Box Counting
# =============================================
def count_boxes_rotated(spatial_index, feature_geoms, n, box_size, 
                        offset_x, offset_y, angle, center_x, center_y, save_boxes=False):
    count = 0
    occupied_boxes = []
    
    for i in range(n):
        bx_min = offset_x + i * box_size
        for j in range(n):
            by_min = offset_y + j * box_size
            
            corners = [(bx_min, by_min), (bx_min+box_size, by_min),
                      (bx_min+box_size, by_min+box_size), (bx_min, by_min+box_size)]
            original_corners = [rotate_point(x, y, center_x, center_y, -angle) 
                              for x, y in corners]
            xs, ys = [p[0] for p in original_corners], [p[1] for p in original_corners]
            
            cell_orig = QgsRectangle(min(xs), min(ys), max(xs), max(ys))
            candidates = spatial_index.intersects(cell_orig)
            
            if candidates:
                cell_geom = QgsGeometry.fromRect(cell_orig)
                for fid in candidates:
                    try:
                        if feature_geoms[fid].intersects(cell_geom):
                            count += 1
                            if save_boxes and n <= 50 and len(occupied_boxes) < 1000:
                                occupied_boxes.append((bx_min, bx_min+box_size, 
                                                       by_min, by_min+box_size))
                            break
                    except:
                        continue
    return count, occupied_boxes

def vector_box_counting(layer, config):
    x_min, y_min, x_max, y_max = get_data_extent(layer)
    width, height = x_max-x_min, y_max-y_min
    side = max(width, height) * config.buffer_factor
    center_x, center_y = (x_min+x_max)/2, (y_min+y_max)/2

    divisions = generate_scales(config)
    spatial_index, feature_geoms = build_spatial_index(layer)
    
    epsilon_dict = {}
    grid_examples = {}
    
    shifts = np.linspace(0, 1, config.grid_shifts, endpoint=False)
    rotations = get_deterministic_rotations(config.grid_rotations)
    
    config_idx = 0
    
    for rot_angle in rotations:
        rx_min, ry_min, rx_max, ry_max = get_rotated_extent(x_min, y_min, x_max, y_max, rot_angle)
        rot_side = max(rx_max-rx_min, ry_max-ry_min) * config.buffer_factor
        
        for shift_x in shifts:
            for shift_y in shifts:
                config_idx += 1
                
                for n in divisions:
                    box_size = rot_side / n
                    offset_x = rx_min - (rot_side-(rx_max-rx_min))/2.0 + shift_x*box_size
                    offset_y = ry_min - (rot_side-(ry_max-ry_min))/2.0 + shift_y*box_size
                    
                    save_boxes = (config_idx == 1)
                    count, boxes = count_boxes_rotated(
                        spatial_index, feature_geoms, n, box_size,
                        offset_x, offset_y, rot_angle, center_x, center_y, save_boxes
                    )
                    
                    if count > 0:
                        eps_key = round(np.log(box_size), 6)
                        if eps_key not in epsilon_dict:
                            epsilon_dict[eps_key] = []
                        epsilon_dict[eps_key].append(count)
                        if save_boxes and n <= 50:
                            grid_examples[n] = boxes
    
    if not epsilon_dict:
        raise ValueError("No valid box-counting results")
    
    eps_sorted = sorted(epsilon_dict.keys())
    avg_eps = np.exp(np.array(eps_sorted))
    avg_N = np.array([np.mean(epsilon_dict[e]) for e in eps_sorted])
    
    return avg_eps, avg_N, grid_examples

# =============================================
# 6. Benjamini-Yekutieli FDR
# =============================================
def apply_by_fdr(p_values, alpha=0.05):
    if len(p_values) == 0:
        return []
    p_values = np.array(p_values)
    n = len(p_values)
    idx = np.argsort(p_values)
    sorted_p = p_values[idx]
    
    harmonic_sum = np.sum(1.0 / np.arange(1, n+1))
    thresholds = alpha * np.arange(1, n+1) / (n * harmonic_sum)
    sig = sorted_p <= thresholds
    
    if not np.any(sig):
        return []
    sig_indices = np.where(sig)[0]
    if len(sig_indices) == 0:
        return []
    return idx[:np.max(sig_indices)+1].tolist()

# =============================================
# 7. Find Top Windows
# =============================================
def find_top_windows(X, Y, config):
    n = len(X)
    windows = []
    max_w = min(n, config.optimal_scale_points * config.max_window_size_multiplier)
    
    for w in range(config.min_scale_points, max_w + 1):
        for start in range(0, n - w + 1, config.window_step):
            end = start + w
            if w < 3:
                continue
            
            try:
                slope, intercept, r_value, p_value, std_err = linregress(X[start:end], Y[start:end])
                r2 = r_value**2
                
                if r2 < config.min_r2_threshold:
                    continue
                if not (config.slope_range[0] < slope < config.slope_range[1]):
                    continue
                
                rho, _ = spearmanr(X[start:end], Y[start:end])
                
                local_std = np.std(X[start:end]) + 1e-9
                quad = np.polyfit(X[start:end], Y[start:end], 2)
                curvature = abs(quad[0]) / (local_std**2)
                
                is_linear = abs(rho) > config.spearman_threshold and curvature < config.curvature_threshold
                
                score = 0.5 * r2 + 0.3 * (w / max_w) + 0.2 * min(1.0, 1.0/(std_err + 0.01))
                if is_linear:
                    score += 0.1
                
                windows.append({
                    'start': start, 'end': end, 'slope': slope,
                    'r2': r2, 'p_value': p_value, 'std_err': std_err,
                    'intercept': intercept, 'size': w, 'score': score,
                    'is_linear': is_linear, 'rho': rho
                })
            except:
                continue
    
    if not windows:
        return []
    
    p_values = [w['p_value'] for w in windows]
    
    if config.use_by_fdr:
        sig_idx = apply_by_fdr(p_values, config.fdr_alpha)
    else:
        sig_idx = apply_fdr_standard(p_values, config.fdr_alpha)
    
    if not sig_idx:
        return []
    
    sig_windows = [windows[i] for i in sig_idx]
    sig_windows.sort(key=lambda x: x['score'], reverse=True)
    return sig_windows[:config.top_windows]

def apply_fdr_standard(p_values, alpha=0.05):
    if len(p_values) == 0:
        return []
    p_values = np.array(p_values)
    idx = np.argsort(p_values)
    sorted_p = p_values[idx]
    n = len(p_values)
    thresholds = alpha * np.arange(1, n+1) / n
    sig = sorted_p <= thresholds
    if not np.any(sig):
        return []
    return idx[:np.max(np.where(sig)[0])+1].tolist()

# =============================================
# 8. Ensemble D (MEDIAN for final estimate)
# =============================================
def ensemble_fractal_dimension(top_windows):
    if not top_windows:
        return 0, 0, 0, 0
    
    slopes = np.array([w['slope'] for w in top_windows])
    positive_fraction = np.mean(slopes > 0)
    d_median = np.median(slopes)
    
    jackknife_d = []
    for i in range(len(top_windows)):
        idx = [j for j in range(len(top_windows)) if j != i]
        if len(idx) == 0:
            continue
        jackknife_d.append(np.median(slopes[idx]))
    
    d_std = np.std(jackknife_d) if len(jackknife_d) > 1 else 0.05
    cv = d_std / abs(d_median) if abs(d_median) > 0.001 else 1.0
    stability_score = 1.0 / (1.0 + d_std)
    
    return d_median, d_std, stability_score, positive_fraction

# =============================================
# 9. SCALE-BLOCK Bootstrap (RESTORED from v18)
# =============================================
def scale_block_bootstrap_ci(X, Y, config):
    """
    Scale-block bootstrap: resamples contiguous scale intervals.
    Preserves local correlation structure between adjacent scales.
    """
    n = len(X)
    block_size = max(2, int(n * config.block_bootstrap_scale_fraction))
    d_values = []
    
    np.random.seed(config.seed)
    
    for _ in range(config.bootstrap_samples):
        sampled_X = []
        sampled_Y = []
        
        while len(sampled_X) < n:
            start = np.random.randint(0, n - block_size + 1)
            end = min(start + block_size, n)
            sampled_X.extend(X[start:end].tolist())
            sampled_Y.extend(Y[start:end].tolist())
        
        sampled_X = np.array(sampled_X[:n])
        sampled_Y = np.array(sampled_Y[:n])
        
        sort_idx = np.argsort(sampled_X)
        sampled_X = sampled_X[sort_idx]
        sampled_Y = sampled_Y[sort_idx]
        
        if len(sampled_X) >= 4:
            try:
                slope, _, _, _, _ = linregress(sampled_X, sampled_Y)
                d_values.append(slope)  # NO filtering by slope > 0
            except:
                continue
    
    if len(d_values) < 30:
        if len(d_values) > 0:
            d_median = np.median(d_values)
            d_mad = np.median(np.abs(d_values - d_median)) * 1.4826
            return d_median - 2*d_mad, d_median + 2*d_mad
        return 0, 0
    
    # CI from all bootstrap values (no filtering)
    return np.percentile(d_values, 2.5), np.percentile(d_values, 97.5)

# =============================================
# 10. Scale Stability Curve
# =============================================
def compute_scale_stability_curve(X, Y, config):
    n = len(X)
    if n < 8:
        return []
    
    curves = []
    for fraction in [0.4, 0.6, 0.8]:
        window_size = max(int(n * fraction), config.min_scale_points)
        step = max(1, (n - window_size) // 15)
        
        for start in range(0, n - window_size + 1, step):
            end = start + window_size
            if end - start >= 4:
                try:
                    slope, _, _, _, _ = linregress(X[start:end], Y[start:end])
                    mid_scale = (X[start] + X[end-1]) / 2
                    curves.append({
                        'mid_scale': mid_scale,
                        'D': slope,
                        'fraction': fraction
                    })
                except:
                    continue
    
    return curves

# =============================================
# 11. Confidence
# =============================================
def determine_confidence(r2, cv, ci_width, positive_fraction):
    if (r2 > 0.98 and cv < 0.05 and ci_width < 0.1 and positive_fraction > 0.9):
        return ConfidenceLevel.HIGH
    elif (r2 > 0.95 and cv < 0.10 and positive_fraction > 0.8):
        return ConfidenceLevel.MEDIUM
    elif (r2 > 0.90 and positive_fraction > 0.7):
        return ConfidenceLevel.LOW
    return ConfidenceLevel.UNRELIABLE

# =============================================
# 12. Visualization (FIXED regression line)
# =============================================
def create_plot(result, output_path=None):
    plt.switch_backend('Agg')
    
    colors = {
        'HIGH': '#2E86AB', 'MEDIUM': '#A23B72',
        'LOW': '#F18F01', 'UNRELIABLE': '#C73E1D'
    }
    mc = colors.get(result.get('confidence_level', 'MEDIUM'), '#2E86AB')
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3,
                 height_ratios=[1.2, 1], width_ratios=[1.5, 1])
    
    # Main log-log plot
    ax_main = fig.add_subplot(gs[0, 0])
    X = np.array(result['log_eps'])
    Y = np.array(result['log_n'])
    
    ax_main.scatter(X, Y, s=70, c='#2C3E50', alpha=0.7,
                   edgecolors='white', linewidth=0.5, zorder=3, label='Data')
    
    # FIXED: Use primary window slope and intercept for regression line
    D_display = result.get('primary_slope', result['fractal_dimension'])
    intercept_display = result.get('primary_intercept', result.get('regression_intercept', 0))
    
    xf = np.linspace(X.min(), X.max(), 200)
    yf = D_display * xf + intercept_display
    ax_main.plot(xf, yf, '-', color=mc, linewidth=2.5, alpha=0.9,
                label='D = ' + result['fractal_dimension_rounded'], zorder=5)
    
    ci_l, ci_u = result.get('nested_ci_lower', 0), result.get('nested_ci_upper', 0)
    if ci_l != ci_u and abs(ci_u-ci_l) < 1.0:
        ax_main.fill_between(xf, yf-(D_display-ci_l)*xf, yf+(ci_u-D_display)*xf,
                            alpha=0.1, color=mc, zorder=1)
    
    ax_main.set_xlabel('X = log(1/ε)', fontsize=12, labelpad=10)
    ax_main.set_ylabel('Y = log(N(ε))', fontsize=12, labelpad=10)
    ax_main.set_title('Fractal Dimension: N(ε) ∝ ε^(-D)', fontsize=14, fontweight='bold')
    ax_main.legend(fontsize=10, loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax_main.grid(True, alpha=0.2, linestyle='--')
    
    d_text = "D = " + result['fractal_dimension_rounded']
    ax_main.text(0.03, 0.97, d_text, transform=ax_main.transAxes, fontsize=20,
                fontweight='bold', va='top', fontfamily='serif',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor=mc, linewidth=2, alpha=0.95))
    
    # Scale stability
    ax_stab = fig.add_subplot(gs[0, 1])
    scale_curve = result.get('scale_stability_curve', [])
    
    if scale_curve:
        scales = [s['mid_scale'] for s in scale_curve]
        d_vals = [s['D'] for s in scale_curve]
        fractions = [s['fraction'] for s in scale_curve]
        
        unique_fractions = sorted(set(fractions))
        for f in unique_fractions:
            idx = [i for i, fr in enumerate(fractions) if fr == f]
            ax_stab.scatter([scales[i] for i in idx], [d_vals[i] for i in idx],
                          s=40, alpha=0.7, label=str(int(f*100)) + '% window')
        
        ax_stab.axhline(y=D_display, color=mc, linestyle='--', linewidth=2,
                       alpha=0.7, label='D = ' + result['fractal_dimension_rounded'])
        
        ax_stab.set_xlabel('Mid-scale X', fontsize=10)
        ax_stab.set_ylabel('Local D', fontsize=10)
        ax_stab.set_title('Multi-Scale Stability', fontsize=12, fontweight='bold')
        ax_stab.legend(fontsize=8, loc='best')
        ax_stab.grid(True, alpha=0.2, linestyle='--')
    
    # Grid example
    ax_grid = fig.add_subplot(gs[1, 0])
    levels = sorted(result.get('grid_examples', {}).keys())
    if len(levels) >= 2:
        n = levels[len(levels)//2]
        boxes = result['grid_examples'].get(n, [])
        
        if boxes:
            for b in boxes[:500]:
                rect = mpatches.Rectangle((b[0], b[2]), b[1]-b[0], b[3]-b[2],
                                         linewidth=0.1, edgecolor='#2C3E50',
                                         facecolor=mc, alpha=0.3)
                ax_grid.add_patch(rect)
            
            if len(boxes) > 0:
                xs = [b[0] for b in boxes[:200]] + [b[1] for b in boxes[:200]]
                ys = [b[2] for b in boxes[:200]] + [b[3] for b in boxes[:200]]
                if xs and ys:
                    m = (max(xs)-min(xs))*0.05
                    ax_grid.set_xlim(min(xs)-m, max(xs)+m)
                    ax_grid.set_ylim(min(ys)-m, max(ys)+m)
        
        ax_grid.set_aspect('equal')
        ax_grid.set_title('Grid ' + str(n) + 'x' + str(n) + ' (' + str(len(boxes)) + ' cells)',
                         fontsize=11, fontweight='bold')
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])
    
    # Results panel
    ax_info = fig.add_subplot(gs[1, 1])
    ax_info.axis('off')
    
    conf = result.get('confidence_level', 'N/A')
    cv = result.get('cv', 0)
    ci_width = abs(result.get('nested_ci_upper', 0) - result.get('nested_ci_lower', 0))
    
    info_text = (
        "┌───────────────────────────────────┐\n"
        "│  FRACTAL DIMENSION D = " + result['fractal_dimension_rounded'] + "        │\n"
        "├───────────────────────────────────┤\n"
        "│                                   │\n"
        "│  95% CI: [" + f"{result.get('nested_ci_lower',0):.3f}" + ",                │\n"
        "│           " + f"{result.get('nested_ci_upper',0):.3f}" + "]                │\n"
        "│  CI Width: " + f"{ci_width:.4f}" + "                    │\n"
        "│                                   │\n"
        "│  R² = " + f"{result.get('r_squared',0):.4f}" + "                         │\n"
        "│  CV(D) = " + f"{cv*100:.1f}" + "%                        │\n"
        "│  Stability = " + f"{result.get('stability_score',0):.3f}" + "                    │\n"
        "│                                   │\n"
        "│  Confidence: " + f"{conf:<12}" + "            │\n"
        "│  Positive windows: " + f"{result.get('positive_fraction',0)*100:.0f}" + "%             │\n"
        "│  Windows: " + f"{len(result.get('top_windows',[])):<3}" + " BY-FDR significant        │\n"
        "│  Points: " + f"{result.get('scaling_points_count',0):<3}" + "                        │\n"
        "│                                   │\n"
        "│  Count averaging: Arithmetic mean  │\n"
        "│  D estimator: Median ensemble      │\n"
        "│  Rotations: Deterministic SO(2)    │\n"
        "│  Bootstrap: Scale-Block            │\n"
        "│  Configs: " + f"{result.get('total_configs',0):<4}" + "                     │\n"
        "│  Time: " + f"{result.get('analysis_duration',0):.1f}" + "s                      │\n"
        "│                                   │\n"
        "└───────────────────────────────────┘"
    )
    
    ax_info.text(0.5, 0.5, info_text, transform=ax_info.transAxes, fontsize=9,
                va='center', ha='center', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='#F8F9FA',
                         edgecolor='#BDC3C7', linewidth=1.5, alpha=0.95))
    
    plt.suptitle(result['layer_name'] + ' | Box-Counting Fractal Dimension Analysis',
                fontsize=13, fontweight='bold', y=0.98)
    
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(),
            "fractal_D" + result['fractal_dimension_rounded'] + "_" +
            datetime.now().strftime("%Y%m%d_%H%M%S") + ".png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    return output_path

# =============================================
# 13. Main Pipeline
# =============================================
def analyze_fractal_dimension(layer=None, config=None):
    if config is None:
        config = BoxCountingConfig()
    if layer is None:
        layer = iface.activeLayer()
    
    logger.info("="*60)
    logger.info("FRACTAL DIMENSION v20.0 FINAL")
    logger.info("Layer: " + layer.name() + " | Features: " + str(layer.featureCount()))
    
    start_time = datetime.now()
    
    # Box counting
    logger.info("Box counting (deterministic SO(2), arithmetic mean)...")
    eps, N, grids = vector_box_counting(layer, config)
    if len(eps) < config.min_scale_points:
        raise ValueError("Only " + str(len(eps)) + " scales")
    
    X = np.log(1.0/eps)
    Y = np.log(N)
    
    # BY-FDR windows
    logger.info("Finding BY-FDR significant windows...")
    top_windows = find_top_windows(X, Y, config)
    if not top_windows:
        raise ValueError("No BY-FDR significant windows found")
    
    logger.info("Found " + str(len(top_windows)) + " significant windows")
    
    # Ensemble D (median for final estimate)
    D_median, D_std, stability, positive_fraction = ensemble_fractal_dimension(top_windows)
    cv = D_std / abs(D_median) if abs(D_median) > 0.001 else 1.0
    
    # Primary window for regression line display
    primary = top_windows[0]
    
    # Scale stability
    scale_curve = compute_scale_stability_curve(X, Y, config)
    
    # Scale-block bootstrap
    logger.info("Scale-block bootstrap...")
    ci_lower, ci_upper = scale_block_bootstrap_ci(X, Y, config)
    ci_width = abs(ci_upper - ci_lower)
    
    # Confidence
    conf = determine_confidence(primary['r2'], cv, ci_width, positive_fraction)
    is_fractal = conf in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]
    
    effective_range = X[primary['end']-1] - X[primary['start']]
    duration = (datetime.now() - start_time).total_seconds()
    total_configs = config.grid_shifts**2 * config.grid_rotations
    
    result = {
        'fractal_dimension': D_median,
        'fractal_dimension_rounded': f"{D_median:.2f}",
        'primary_slope': primary['slope'],
        'primary_intercept': primary['intercept'],
        'r_squared': primary['r2'],
        'stability_score': stability,
        'cv': cv,
        'positive_fraction': positive_fraction,
        'confidence_level': conf,
        'nested_ci_lower': ci_lower,
        'nested_ci_upper': ci_upper,
        'is_fractal': is_fractal,
        'scaling_points_count': primary['size'],
        'total_points': len(X),
        'regression_intercept': primary['intercept'],
        'eps_values': eps,
        'n_values': N,
        'log_eps': X,
        'log_n': Y,
        'grid_examples': grids,
        'layer_name': layer.name(),
        'analysis_duration': duration,
        'top_windows': top_windows,
        'scale_stability_curve': scale_curve,
        'effective_range': effective_range,
        'total_configs': total_configs,
        'is_reliable': conf in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]
    }
    
    logger.info("D = " + result['fractal_dimension_rounded'] +
                " | CV=" + f"{cv*100:.1f}%" +
                " | " + conf + " | " + f"{duration:.1f}s")
    
    return result

# =============================================
# 14. Main Execution
# =============================================
try:
    print("\n" + "="*65)
    print("FRACTAL DIMENSION v20.0 FINAL")
    print("Publication-Grade Box-Counting Tool")
    print("="*65)
    
    layer = iface.activeLayer()
    if not layer or layer.type() != QgsMapLayer.VectorLayer:
        raise RuntimeError("Select a vector layer!")
    
    print("Layer: " + layer.name() + " (" + str(layer.featureCount()) + " features)")
    print("CRS: " + layer.crs().authid())
    print("-"*65 + "\n")
    
    config = BoxCountingConfig()
    config.num_scales = 15
    config.grid_rotations = 8
    config.grid_shifts = 3
    config.top_windows = 8
    config.bootstrap_samples = 200
    config.window_step = 2
    config.spearman_threshold = 0.93
    config.block_bootstrap_scale_fraction = 0.25
    config.use_by_fdr = True
    config.seed = 42
    
    result = analyze_fractal_dimension(layer, config)
    plot_path = create_plot(result)
    
    print("\n" + "="*65)
    print("FINAL RESULTS")
    print("="*65)
    print("D = " + result['fractal_dimension_rounded'])
    print("95% CI: [" + f"{result['nested_ci_lower']:.3f}" + ", " + f"{result['nested_ci_upper']:.3f}" + "]")
    print("CV(D): " + f"{result['cv']*100:.1f}%")
    print("R² = " + f"{result['r_squared']:.4f}")
    print("Positive windows: " + f"{result['positive_fraction']*100:.0f}%")
    print("Confidence: " + result['confidence_level'])
    print("Time: " + f"{result['analysis_duration']:.1f}s")
    print("\nPlot: " + plot_path)
    
    try:
        layer.setCustomProperty("fractal_D", result['fractal_dimension_rounded'])
        layer.setCustomProperty("fractal_CI", f"[{result['nested_ci_lower']:.3f}, {result['nested_ci_upper']:.3f}]")
        layer.setCustomProperty("fractal_CV", f"{result['cv']*100:.1f}%")
        layer.setCustomProperty("fractal_R2", f"{result['r_squared']:.4f}")
        layer.setCustomProperty("fractal_confidence", result['confidence_level'])
        print("Saved to layer metadata")
    except:
        pass
    
    webbrowser.open(plot_path)
    print("\nDone! Ready for publication.")
    
except Exception as e:
    print("\nError: " + str(e))
    traceback.print_exc()
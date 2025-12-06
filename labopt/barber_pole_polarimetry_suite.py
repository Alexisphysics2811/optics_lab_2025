"""
barber_pole_polarimetry_suite.py

Updated: added command-line options to specify tube length and concentration and to compute implied specific rotation and enantiomeric excess (ee%). Also added --out-dir to write all outputs into a chosen directory.

Contents:
1) Python analysis script: fits I(theta) to extract optical rotation alpha (deg) and its uncertainty
   - supports CSV (theta_deg, intensity) and image-series (filenames containing angle in degrees)
   - uses the cos(2θ) linearization and analytic propagation of covariance for σ_alpha
   - NEW: options --length-cm and --concentration to compute implied specific rotation(s)
   - NEW: option --alpha-pure to compute ee% (if you know literature specific rotation at that λ)
   - NEW: option --out-dir to write all result files into a directory

2) Parts list + alignment checklist (embedded below as commented markdown)

3) Sample spreadsheet (CSV block) and formulas to convert measured alpha(λ) -> ee% with propagated uncertainty

Requirements:
  numpy, scipy, pandas, matplotlib, opencv-python (cv2) [cv2 optional if using images]
  Install e.g.: pip install numpy scipy pandas matplotlib opencv-python

Usage examples (from command line):
  python barber_pole_polarimetry_suite.py --csv laser_input.csv --length-cm 50 --concentration 0.6917 --alpha-pure -92.4 --out laser_trial --out-dir results/laser_run1
  python barber_pole_polarimetry_suite.py --images ./img_folder --pattern "img_(?P<angle>\d+)_ch{channel}.jpg" --channel R --out-dir ./results

"""

import re
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# ---------------------------
# Core fitting utilities
# ---------------------------

def design_matrix(theta_deg):
    """Return design matrix X for model I = C + D*cos(2θ) + E*sin(2θ).
    theta_deg: array-like in degrees
    returns: X (N x 3), columns [1, cos2, sin2]
    """
    th = np.deg2rad(theta_deg)
    cos2 = np.cos(2.0 * th)
    sin2 = np.sin(2.0 * th)
    X = np.vstack([np.ones_like(th), cos2, sin2]).T
    return X


def fit_cos2(theta_deg, I, sigma=None):
    """Linear least-squares fit to I(theta) = C + D cos(2θ) + E sin(2θ)
    If sigma provided, performs weighted LS. Returns dict with keys:
      C, D, E, cov (3x3 cov matrix), residuals, sigma2, R2, alpha_deg, sigma_alpha_deg
    """
    theta_deg = np.asarray(theta_deg)
    I = np.asarray(I)
    if theta_deg.shape != I.shape:
        raise ValueError('theta and I must have same shape')

    X = design_matrix(theta_deg)

    if sigma is None:
        # ordinary LS
        XtX = X.T.dot(X)
        try:
            XtX_inv = linalg.inv(XtX)
        except linalg.LinAlgError:
            XtX_inv = linalg.pinv(XtX)
        p = XtX_inv.dot(X.T).dot(I)
        resid = I - X.dot(p)
        dof = max(len(I) - X.shape[1], 1)
        sigma2 = np.sum(resid**2) / dof
        cov = sigma2 * XtX_inv
    else:
        # weighted LS
        w = 1.0 / (np.asarray(sigma)**2)
        W = np.diag(w)
        XtWX = X.T.dot(W).dot(X)
        try:
            XtWX_inv = linalg.inv(XtWX)
        except linalg.LinAlgError:
            XtWX_inv = linalg.pinv(XtWX)
        p = XtWX_inv.dot(X.T).dot(W).dot(I)
        resid = I - X.dot(p)
        dof = max(len(I) - X.shape[1], 1)
        # estimate sigma2 from weighted residuals
        sigma2 = np.sum((resid**2) * w) / dof
        cov = sigma2 * XtWX_inv

    C, D, E = p

    # compute alpha (deg) from D,E via alpha = 0.5 * atan2(-E, D)
    alpha_rad = 0.5 * np.arctan2(-E, D)
    alpha_deg = np.rad2deg(alpha_rad)

    # propagate uncertainty for alpha using cov(D,E)
    varD = cov[1,1]
    varE = cov[2,2]
    covDE = cov[1,2]
    denom = (D**2 + E**2)
    if denom == 0:
        sigma_alpha_deg = np.nan
    else:
        d_alpha_dD = 0.5 * (E / denom)  # ∂alpha/∂D
        d_alpha_dE = -0.5 * (D / denom)  # ∂alpha/∂E
        var_alpha_rad = (d_alpha_dD**2) * varD + (d_alpha_dE**2) * varE + 2*d_alpha_dD*d_alpha_dE*covDE
        sigma_alpha_deg = np.rad2deg(np.sqrt(max(var_alpha_rad, 0.0)))

    # R^2
    ss_tot = np.sum((I - np.mean(I))**2)
    ss_res = np.sum(resid**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        'C': float(C), 'D': float(D), 'E': float(E),
        'cov': cov, 'residuals': resid, 'sigma2': float(sigma2), 'R2': float(R2),
        'alpha_deg': float(alpha_deg), 'sigma_alpha_deg': float(sigma_alpha_deg)
    }

# ---------------------------
# CSV-based analysis
# ---------------------------

def analyze_csv(path, theta_col='theta_deg', I_col='intensity', sigma_col=None, plot=True, out_prefix=None, length_cm=None, concentration=None, alpha_pure=None, out_dir='.'):
    df = pd.read_csv(path)
    if theta_col not in df.columns or I_col not in df.columns:
        raise ValueError(f'CSV must have columns {theta_col} and {I_col}')
    theta = df[theta_col].to_numpy()
    I = df[I_col].to_numpy()
    sigma = None
    if sigma_col and sigma_col in df.columns:
        sigma = df[sigma_col].to_numpy()
    res = fit_cos2(theta, I, sigma=sigma)
    if plot:
        plot_fit(theta, I, res, title=os.path.basename(path))
    if out_prefix:
        save_results(res, out_prefix, theta, I, sigma, length_cm, concentration, alpha_pure, out_dir=out_dir)
    return res

# ---------------------------
# Image-series analysis
# ---------------------------

def parse_angle_from_filename(filename, pattern):
    m = re.search(pattern, os.path.basename(filename))
    if not m:
        return None
    return float(m.group('angle'))


def read_roi_mean(image_path, roi=None, channel='gray', linearize_gamma=True):
    if not _HAS_CV2:
        raise RuntimeError('OpenCV (cv2) is required for image processing')
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f'Could not read image {image_path}')
    imgf = img.astype(np.float64)
    if len(imgf.shape) == 2:
        channel_data = imgf
    else:
        b, g, r = imgf[:,:,0], imgf[:,:,1], imgf[:,:,2]
        if channel.lower() == 'r':
            channel_data = r
        elif channel.lower() == 'g':
            channel_data = g
        elif channel.lower() == 'b':
            channel_data = b
        else:
            channel_data = 0.2126*r + 0.7152*g + 0.0722*b
    h, w = channel_data.shape
    if roi is None:
        x0 = int(w*0.25); y0 = int(h*0.25); w0 = int(w*0.5); h0 = int(h*0.5)
    else:
        x0, y0, w0, h0 = roi
    patch = channel_data[y0:y0+h0, x0:x0+w0]
    mean_val = np.mean(patch)
    maxval = np.max(imgf)
    if linearize_gamma and maxval > 0:
        norm = mean_val / maxval
        lin = np.power(norm, 2.2)
        mean_val_lin = lin * maxval
        return float(mean_val_lin)
    else:
        return float(mean_val)


def analyze_image_series(folder, pattern, channel='gray', roi=None, plot=True, out_prefix=None, length_cm=None, concentration=None, alpha_pure=None, out_dir='.'):
    files = sorted([os.path.join(folder,f) for f in os.listdir(folder)])
    angles = []
    intensities = []
    for f in files:
        ang = parse_angle_from_filename(f, pattern)
        if ang is None:
            continue
        try:
            I = read_roi_mean(f, roi=roi, channel=channel)
        except Exception as e:
            print('skipping', f, 'error', e)
            continue
        angles.append(ang)
        intensities.append(I)
    if len(angles) < 6:
        raise RuntimeError('Found too few valid images to fit')
    idx = np.argsort(angles)
    angles = np.array(angles)[idx]
    intensities = np.array(intensities)[idx]
    res = fit_cos2(angles, intensities)
    if plot:
        plot_fit(angles, intensities, res, title=f'Image series {folder}')
    if out_prefix:
        save_results(res, out_prefix, angles, intensities, None, length_cm, concentration, alpha_pure, out_dir=out_dir)
    return res

# ---------------------------
# plotting & I/O
# ---------------------------

def plot_fit(theta_deg, I, res, title='fit'):
    th_grid = np.linspace(min(theta_deg), max(theta_deg), 400)
    Xg = design_matrix(th_grid)
    p = np.array([res['C'], res['D'], res['E']])
    Ig = Xg.dot(p)

    plt.figure(figsize=(6,4))
    plt.scatter(theta_deg, I, label='data')
    plt.plot(th_grid, Ig, '-', label=f'fit: alpha={res["alpha_deg"]:.3f}° ± {res["sigma_alpha_deg"]:.3f}°')
    plt.xlabel('Analyzer angle (deg)')
    plt.ylabel('Intensity (arb)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_results(res, prefix, theta=None, I=None, sigma=None, length_cm=None, concentration=None, alpha_pure=None, out_dir='.'):
    # ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # helper to build full path
    def outpath(suffix):
        return os.path.join(out_dir, f"{prefix}{suffix}")

    # save a small JSON/csv summary
    out = {
        'alpha_deg': res['alpha_deg'], 'sigma_alpha_deg': res['sigma_alpha_deg'],
        'C': res['C'], 'D': res['D'], 'E': res['E'], 'R2': res['R2']
    }
    pd.DataFrame([out]).to_csv(outpath('_summary.csv'), index=False)
    np.savetxt(outpath('_cov.csv'), res['cov'], delimiter=',')

    # save fit table if data provided
    if theta is not None and I is not None:
        df = pd.DataFrame({'theta_deg': np.asarray(theta), 'I': np.asarray(I)})
        if sigma is not None:
            df['sigma'] = sigma
        # add fitted values
        X = design_matrix(df['theta_deg'].to_numpy())
        p = np.array([res['C'], res['D'], res['E']])
        df['I_fit'] = X.dot(p)
        df['residual'] = df['I'] - df['I_fit']
        df.to_csv(outpath('_data_and_fit.csv'), index=False)

    # If length and concentration provided, compute possible implied specific rotations for multiple unwraps
    if length_cm is not None and concentration is not None:
        l_dm = float(length_cm) / 10.0
        c = float(concentration)
        alpha_obs = float(res['alpha_deg'])
        sigma_alpha = float(res['sigma_alpha_deg'])
        rows = []
        for m in range(-10, 11):
            alpha_true = alpha_obs + 180.0 * m
            implied_specific = alpha_true / (l_dm * c)
            ee_percent = None
            if alpha_pure is not None and alpha_pure != 0:
                ee_percent = 100.0 * alpha_true / (alpha_pure * l_dm * c)
            rows.append({'m': m, 'alpha_true_deg': alpha_true, 'implied_specific_deg_per_dm_per_gpermL': implied_specific, 'ee_percent': ee_percent})
        dfm = pd.DataFrame(rows)
        dfm.to_csv(outpath('_unwrap_table.csv'), index=False)
        # also write a short human-readable summary
        summary = {
            'length_cm': length_cm, 'length_dm': l_dm, 'concentration_g_per_mL': c,
            'alpha_obs_deg': alpha_obs, 'sigma_alpha_deg': sigma_alpha,
        }
        pd.DataFrame([summary]).to_csv(outpath('_measurement_context.csv'), index=False)

# ---------------------------
# Command-line interface
# ---------------------------

def main_cli():
    p = argparse.ArgumentParser(description='Barber-pole polarimetry analysis tool (with specific rotation computation and out-dir)')
    p.add_argument('--csv', help='CSV with columns theta_deg,intensity')
    p.add_argument('--images', help='Folder with images. Filenames must encode angle.')
    p.add_argument('--pattern', help='Regex pattern with named group "angle" to parse filenames', default=r'(?P<angle>\d+\.?\d*)')
    p.add_argument('--channel', help='Image channel: gray,R,G,B', default='gray')
    p.add_argument('--roi', help='ROI as x,y,w,h (pixels)', default=None)
    p.add_argument('--out', help='Output filename prefix for results', default='barberpole_out')
    p.add_argument('--out-dir', help='Output directory where result files will be written', default='.')
    p.add_argument('--no-plot', dest='plot', action='store_false')
    p.add_argument('--sigma-col', help='If CSV has a sigma column name, provide it here', default=None)
    p.add_argument('--length-cm', type=float, help='Tube length in cm (for specific rotation calculation)', default=None)
    p.add_argument('--concentration', type=float, help='Concentration in g per mL (for specific rotation calculation)', default=None)
    p.add_argument('--alpha-pure', type=float, help='Specific rotation of pure enantiomer at same lambda (deg per dm per g/mL) to compute ee%%', default=None)
    args = p.parse_args()

    if args.csv:
        res = analyze_csv(args.csv, sigma_col=args.sigma_col, plot=args.plot, out_prefix=args.out, length_cm=args.length_cm, concentration=args.concentration, alpha_pure=args.alpha_pure, out_dir=args.out_dir)
    elif args.images:
        res = analyze_image_series(args.images, args.pattern, channel=args.channel, roi=None if not args.roi else tuple(map(int,args.roi.split(','))), plot=args.plot, out_prefix=args.out, length_cm=args.length_cm, concentration=args.concentration, alpha_pure=args.alpha_pure, out_dir=args.out_dir)
    else:
        p.print_help()
        sys.exit(1)

    print('Fit result:')
    print('  alpha (observed, wrapped) =', f"{res['alpha_deg']:.4f}", '+/-', f"{res['sigma_alpha_deg']:.4f}", 'deg')
    if args.length_cm is not None and args.concentration is not None:
        l_dm = args.length_cm / 10.0
        c = args.concentration
        print('Computed possible unwrapped specific rotations (see *_unwrap_table.csv in out-dir):')
        # load table and print a few entries
        out_tbl = pd.read_csv(os.path.join(args.out_dir, args.out + '_unwrap_table.csv'))
        # print rows where implied specific is within reasonable range (abs < 1000) for readability
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(out_tbl[['m','alpha_true_deg','implied_specific_deg_per_dm_per_gpermL','ee_percent']].to_string(index=False, float_format='{:0.4f}'.format))
        # suggest best match if alpha_pure given
        if args.alpha_pure is not None:
            diffs = np.abs(out_tbl['implied_specific_deg_per_dm_per_gpermL'] - args.alpha_pure)
            best = out_tbl.iloc[diffs.idxmin()]
            print('Best match to provided alpha_pure =', args.alpha_pure, 'is m =', int(best['m']), ', implied specific =', float(best['implied_specific_deg_per_dm_per_gpermL']))

    print('Saved outputs with prefix:', args.out, 'in directory:', args.out_dir)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main_cli()
    else:
        print('Module loaded. Use --help for CLI usage.')


# ==========================
# Parts list & alignment checklist
# (human-readable markdown below — keep inside this file for versioning)
# ==========================

"""

# Parts list (given your lab inventory)

You already listed equipment available. Below is an ordered shopping/borrow checklist and recommended minimum specs.

## Already available (from your message)
- Spectrometers (lab) — can be used for best ORD (recommended).
- LEDs (various colors) — good narrow-ish illumination when driven stably.
- Lasers (HeNe, diode etc.) — very narrowband; use diffuser to avoid speckle for camera.
- Collimated white light lamps — good for spectrometer + camera measurements.
- Element discharge lamps (e.g., Hg, Na, etc.) — good discrete spectral lines (e.g., 589 nm sodium D-line classic).
- Photometers / photodiodes — for point intensity measurements with high SNR.
- "Donated Barber Pole experiment tube" ~ probable length > 1 m — long path length, big rotation expected.

## Recommended / useful extras
- High-extinction linear polarizers (Glan–Thompson, or good sheet polarizers) for input and analyzer.
- Precision rotation mount (vernier or motorized) for analyzer with at least 0.1° resolution (0.01° if you want highly precise results).
- Temperature probe / bath or at least thermometer to log T (±0.1°C ideal).
- Stable sample mount for the long tube (avoid stress on glass that induces birefringence).
- Neutral-density filters or variable attenuator for lasers to avoid saturating detectors.
- Diffusers / integrating sphere to remove speckle when using lasers with camera.
- Beam stops, apertures, irises to define beam and reduce stray light.
- Tripod stands, optical posts and rails for alignment.
- Zoned (stress-free) cuvettes if you later use shorter path-length cells.

# Alignment & measurement checklist (step-by-step)

0) IDENTIFY PATH LENGTH
- Measure the physical internal path length of the donated barber-pole tube (mm/cm). Convert to **decimeters** for polarimetry formulas: l(dm) = length_cm / 10.
- If path > 1 m, it's >10 dm — expect huge optical rotation; consider dilute solution or shorter cell if rotation >~100° (ambiguity/wrapping).

1) OPTICAL LAYOUT (input → sample → analyzer → detector/camera/screen)
- Mount input linear polarizer on a fixed, stable mount. Mark its transmission axis (0° reference).
- Collimate and center the light source through input polarizer. For spectrometer-based approach, feed spectrometer input with the beam after analyzer (use fiber if needed).
- Mount sample tube so beam passes centrally, orthogonal to tube windows.
- Place analyzer (rotatable polarizer) after sample on a precision rotation stage.
- Place detector (photodiode, photometer) or camera after analyzer. If using a screen for visual demo, place screen far enough to avoid edge effects.

2) ZEROING & CALIBRATION
- With empty tube (air) or solvent-only, rotate analyzer to find the reference angle (min or max, depending on whether you use crossed or parallel arrangement).
- Record that reference as θ0(λ). Do this for each wavelength you will use (since small wavelength-dependent misalignments can occur).
- If using spectrometer, measure background spectrum with no sample and with solvent-only.
- Measure a calibration standard (known sucrose solution or pure enantiomer) at known concentration to validate pipeline.

3) MEASUREMENT
- For each wavelength (or for spectrometer: for each spectral bin): rotate analyzer through 0°→180° in fine steps (e.g., 1–2° or smaller) and record intensity.
- If using a camera: ensure fixed exposure, RAW, linear response (disable auto gain, white balance). Acquire a dark frame.
- If using LEDs or lamps: stabilize with current source (low noise) and wait thermal equilibrium.
- Repeat runs for averaging. Collect solvent-only runs to subtract any leftover offsets.

4) SPECIAL NOTES for long path-length tube
- Expect very large rotations. Keep concentrations low or use shorter optical path to keep alpha within −90°…+90° to avoid ambiguity in arctan.
- If you intentionally use long path for sensitivity, measure at multiple concentrations so you can determine the linearity and avoid wrapping issues.
- If rotation > 90°, the cos^2 pattern still fits but alpha will be modulo 180°/2? In practice, unwrap across wavelengths or use a reference standard.

5) CHECKS
- Rotate the sample tube 90° about beam axis: if the apparent rotation changes, you may have stress birefringence in the tube.
- Use two orthogonal sample orientations to detect and correct for linear birefringence.
- For lasers: use a diffuser to remove speckle before imaging.

6) DATA PROCESSING
- Fit I(θ) to C + D cos(2θ) + E sin(2θ). Recover alpha = 0.5 * atan2(-E, D) (in radians).
- Estimate σ_alpha by full covariance propagation (script included does this).
- Convert to specific rotation and ee using: [α] = α / (l dm * c g·mL^-1) ; ee% = α / ([α]_pure * l * c_tot) * 100

# Notes on wavelength choices with your inventory
- Use spectrometer + white lamp for full ORD across the visible: best and recommended. Acquire S(λ,θ) and fit for each λ bin.
- Use element discharge lamps (e.g., Na D 589 nm) for classic reference points.
- Use LEDs at a few colors for quick checks (G, R, B). Drive LEDs with stable current.
- Use lasers (e.g., HeNe 632.8 nm) for high SNR single-wavelength measurements; mitigate speckle.

# Practical calibration standards
- Sucrose (table sugar) is common; literature [α]_D^20 for sucrose ~ +66.47° (20 °C, 589 nm) but verify for your temperature and wavelength.
- Prepare a calibration solution (e.g., 10 g/100 mL) and measure to extract any system offset or effective path-length correction.

"""

# ==========================
# Sample spreadsheet (CSV) block + formulas
# Save the block below as sample_spreadsheet.csv or open in Excel.
# Columns and formula explanation follow the CSV.
# ==========================

sample_csv = """
wavelength_nm,alpha_deg,sigma_alpha_deg,path_length_cm,conc_g_per_mL,alpha_pure_deg_per_dm_per_1g_per_mL,sigma_alpha_pure_deg,path_length_dm,conc_g_per_mL_used,ee_percent,sigma_ee_percent
589,6.647,0.010,10.0,0.10,66.47,0.05,1.0,0.10,100.0,1.50
632.8,5.80,0.012,10.0,0.10,60.0,0.10,1.0,0.10,96.6667,2.00

"""

# Column meanings and Excel formulas (use these in spreadsheet cells):
# Assume columns: A wavelength_nm, B alpha_deg, C sigma_alpha_deg, D path_length_cm, E conc_g_per_mL,
# F alpha_pure_deg_per_dm_per_1g_per_mL (specific rotation units matching our definition), G sigma_alpha_pure_deg,
# H path_length_dm (helper) = D/10
# I conc_g_per_mL_used = E
# J ee_percent = 100 * B / (F * H * I)
# K sigma_ee_percent = J * SQRT( (C/B)^2 + (G/F)^2 + (0.01*???)^2 + (sigma_conc/conc)^2 ) -- better use exact propagation

# Exact propagation formula (for spreadsheet use):
# ee = 100 * A / (B * L * C)
# var(ee) = (100/(B*L*C))^2 * var(A) + (100*A/(B^2*L*C))^2 * var(B) + (100*A/(B*L^2*C))^2 * var(L) + (100*A/(B*L*C^2))^2 * var(C)
# In spreadsheet terms (example cell refs):
# J2 = 100 * B2 / (F2 * H2 * I2)
# K2 = SQRT( (100/(F2*H2*I2))^2 * (C2^2) + (100*B2/(F2^2*H2*I2))^2 * (G2^2) + (100*B2/(F2*H2^2*I2))^2 * (sigma_L^2) + (100*B2/(F2*H2*I2^2))^2 * (sigma_conc^2) )
# where sigma_L = uncertainty in H (dm) and sigma_conc = uncertainty in concentration (g/mL)

# Minimal example values above (two rows) — real experimental values go in B,C,D,E,F,G

# Save sample csv locally for convenience
open('sample_spreadsheet.csv','w').write(sample_csv)

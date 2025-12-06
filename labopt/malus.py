#!/usr/bin/env python3
"""
Interactive fitter for I(theta) = A * cos^2(theta) + B

Usage:
  python fit_Acos2_interactive.py

Then type or paste lines of "theta intensity" (degrees), one pair per line.
Finish by hitting ENTER on an empty line or typing 'done' / 'quit'.

Examples of accepted input lines:
  0  1.234
  2,1.210
  4    1.187
"""

import sys
import math
import numpy as np

def fit_A_B(theta_deg, I):
    # design matrix X = [cos^2(theta), 1]
    th = np.deg2rad(np.asarray(theta_deg))
    X = np.vstack([np.cos(th)**2, np.ones_like(th)]).T
    # ordinary least squares
    XtX = X.T @ X
    try:
        cov_p = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        cov_p = np.linalg.pinv(XtX)
    p = cov_p @ (X.T @ I)
    residuals = I - X @ p
    dof = max(len(I) - 2, 1)
    sigma2 = (residuals**2).sum() / dof
    cov = sigma2 * cov_p
    A, B = float(p[0]), float(p[1])
    sigma_A = float(np.sqrt(max(cov[0,0], 0.0)))
    sigma_B = float(np.sqrt(max(cov[1,1], 0.0)))

    ss_tot = np.sum((I - I.mean())**2)
    ss_res = np.sum(residuals**2)
    R2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else float('nan')

    return {
        'A': A, 'B': B,
        'sigma_A': sigma_A, 'sigma_B': sigma_B,
        'cov': cov, 'residuals': residuals, 'R2': R2,
        'sigma2': sigma2
    }

def parse_line_to_pair(line):
    # Accept either whitespace separated or comma separated pairs.
    s = line.strip()
    if not s:
        return None
    if s.lower() in ('done','quit','exit','q'):
        return 'END'
    # replace commas with spaces then split
    s2 = s.replace(',', ' ')
    parts = s2.split()
    if len(parts) < 2:
        raise ValueError("Line must contain two numbers: theta and intensity")
    th = float(parts[0])
    I = float(parts[1])
    return (th, I)

def pretty_print_results(res):
    print("\nFit results (model: I(theta) = A*cos^2(theta) + B)\n")
    print(f"  A = {res['A']:.6g}  ± {res['sigma_A']:.6g}")
    print(f"  B = {res['B']:.6g}  ± {res['sigma_B']:.6g}")
    print(f"  R^2 = {res['R2']:.6g}")
    print(f"  residuals: mean = {np.mean(res['residuals']):.4g}, std = {np.std(res['residuals'], ddof=1):.4g}")
    print(f"  sigma^2 (residual variance) = {res['sigma2']:.6g}")
    print("\nCovariance matrix for [A, B]:")
    print(res['cov'])

def maybe_plot(theta, I, res):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available — skipping plot.")
        return
    thf = np.linspace(min(theta), max(theta), 500)
    Ifit = res['A'] * np.cos(np.deg2rad(thf))**2 + res['B']
    plt.figure(figsize=(6,4))
    plt.scatter(theta, I, s=20, label='data', zorder=5)
    plt.plot(thf, Ifit, '-', label=f'fit: A={res["A"]:.4g}, B={res["B"]:.4g}')
    plt.xlabel('theta (deg)')
    plt.ylabel('Intensity (arb)')
    plt.title('Fit of I(theta) = A cos^2(theta) + B')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    print(__doc__)
    print("Paste or type your data pairs now. Empty line (or 'done') finishes input.\n")
    theta_list = []
    I_list = []
    line_num = 0
    while True:
        try:
            line = input(f"[{line_num:03d}] theta intensity > ")
        except EOFError:
            print("\nEOF received — finishing input.")
            break
        line_num += 1
        if not line.strip():
            # empty line ends input
            break
        try:
            out = parse_line_to_pair(line)
        except Exception as e:
            print("  parse error:", e)
            continue
        if out == 'END':
            break
        th, I = out
        theta_list.append(th)
        I_list.append(I)

    if len(theta_list) < 3:
        print("Need at least 3 data points to fit. Exiting.")
        sys.exit(1)

    theta = np.array(theta_list)
    I = np.array(I_list)

    print(f"\nReceived {len(theta)} points. Performing fit...")
    res = fit_A_B(theta, I)
    pretty_print_results(res)

    # optionally save to CSV
    saveq = input("\nSave input data to CSV file? (enter filename or press ENTER to skip) > ").strip()
    if saveq:
        import csv
        try:
            with open(saveq, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['theta_deg', 'intensity'])
                for t,ii in zip(theta, I):
                    writer.writerow([t, ii])
            print(f"Saved to {saveq}")
        except Exception as e:
            print("Failed to save:", e)

    # ask whether to plot
    plotq = input("Show plot? (y/N) > ").strip().lower()
    if plotq in ('y','yes'):
        maybe_plot(theta, I, res)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
labtool_with_equation_v2.py

A cleaned, modular, and robust rewrite of the original labtool_with_equation.py.

Key improvements / style:
 - No forced Matplotlib backend (don't call matplotlib.use()).
 - Clear separation: DataManager, Fitter, Plotter.
 - Safer CSV import with flexible column mapping.
 - Automatic polynomial-degree selection via adjusted R² (configurable max degree).
 - Manual override for polynomial degree.
 - Plot saving uses dpi/facecolor and pumps the GUI event loop (plt.pause)
   so IDEs like PyCharm can capture the figure reliably.
 - Helpful, small interactive CLI with the same commands as before.

Commands (interactive):
  X,Y[,Err]      -> add a data point
  bulk           -> paste many X,Y[,Err] lines, end with 'end'
  poly N         -> force polynomial degree N (manual override)
  auto           -> return to automatic degree selection
  save           -> export CSV (asks filename)
  load <file>    -> import CSV (replaces current data)
  saveplot [f]   -> save current figure (optional filename)
  clear          -> clear all data (undo not implemented)
  list           -> list current points with indices
  q              -> quit

"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


# ---------- Configuration ----------
OUTPUT_DIR = Path("./output")
CSV_DIR = OUTPUT_DIR / "csv"
OUTPUT_DIR.mkdir(exist_ok=True)
CSV_DIR.mkdir(exist_ok=True)
MAX_AUTO_DEGREE = 5
PLOT_DPI = 150
PAUSE_TIME = 0.12  # seconds to allow GUI backends (tkagg/qt) to render


# ---------- Utilities ----------

def timestamp_str() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)


def adjusted_r2(y: np.ndarray, y_pred: np.ndarray, p: int) -> float:
    """Return adjusted R²; return -inf when not applicable."""
    n = len(y)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0 or n <= p + 1:
        return float('-inf')
    r2 = 1 - ss_res / ss_tot
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


# ---------- Data manager ----------
class DataManager:
    """Simple holder for x,y,err points and CSV import/export."""

    def __init__(self):
        self.df = pd.DataFrame(columns=['x', 'y', 'err'])

    def add_point(self, x: float, y: float, err: Optional[float] = None) -> None:
        err_val = float(err) if err is not None else np.nan
        self.df.loc[len(self.df)] = [float(x), float(y), err_val]

    def clear(self) -> None:
        self.df = pd.DataFrame(columns=['x', 'y', 'err'])

    def import_csv(self, path: str) -> Tuple[bool, str]:
        if not os.path.exists(path):
            return False, f'File not found: {path}'
        try:
            df = pd.read_csv(path)
        except Exception as e:
            return False, f'Failed to read CSV: {e}'

        # Accept x,y,err or first two/three columns
        if 'x' in df.columns and 'y' in df.columns:
            if 'err' not in df.columns:
                df['err'] = np.nan
            new_df = df[['x', 'y', 'err']].copy()
        else:
            cols = list(df.columns)
            if len(cols) < 2:
                return False, 'CSV must contain at least two columns to map to x,y'
            new_df = pd.DataFrame()
            try:
                new_df['x'] = pd.to_numeric(df[cols[0]], errors='coerce')
                new_df['y'] = pd.to_numeric(df[cols[1]], errors='coerce')
                if len(cols) >= 3:
                    new_df['err'] = pd.to_numeric(df[cols[2]], errors='coerce')
                else:
                    new_df['err'] = np.nan
            except Exception as e:
                return False, f'Failed to coerce CSV columns to numeric: {e}'

        # Drop rows with missing x or y
        new_df = new_df.dropna(subset=['x', 'y']).reset_index(drop=True)
        self.df = new_df[['x', 'y', 'err']].copy()
        return True, f'Imported {len(self.df)} points.'

    def export_csv(self, filename: Optional[str] = None) -> str:
        ensure_dirs()
        if not filename:
            filename = f'lab_data_{timestamp_str()}.csv'
        out = CSV_DIR / filename
        self.df.to_csv(out, index=False)
        return str(out)


# ---------- Fitter ----------
class Fitter:
    def __init__(self, max_degree: int = MAX_AUTO_DEGREE):
        self.max_degree = max_degree

    def auto_guess_degree(self, df: pd.DataFrame) -> int:
        n = len(df)
        if n < 3:
            return 1
        max_deg = min(self.max_degree, n - 1)
        best_deg, best_adj = 1, float('-inf')
        x = df['x'].values
        y = df['y'].values
        for deg in range(1, max_deg + 1):
            coeffs = np.polyfit(x, y, deg=deg)
            y_pred = np.polyval(coeffs, x)
            adj = adjusted_r2(y, y_pred, deg)
            if adj > best_adj:
                best_adj, best_deg = adj, deg
        return best_deg

    @staticmethod
    def compute_poly_fit(df: pd.DataFrame, degree: int):
        x = df['x'].values
        y = df['y'].values
        coeffs = np.polyfit(x, y, deg=degree)
        poly = np.poly1d(coeffs)
        # R²
        y_pred = poly(x)
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_pred)**2)
        r2 = 0.0 if ss_tot == 0 else max(0.0, 1 - ss_res / ss_tot)
        return coeffs, poly, r2

    @staticmethod
    def format_poly_equation(coeffs: List[float]) -> str:
        deg = len(coeffs) - 1
        terms = []
        for i, c in enumerate(coeffs):
            power = deg - i
            if abs(c) < 1e-12:
                continue
            c_fmt = f"{c:.5g}"
            if power == 0:
                terms.append(f"{c_fmt}")
            elif power == 1:
                terms.append(f"{c_fmt}x")
            else:
                terms.append(f"{c_fmt}x^{power}")
        if not terms:
            return r"$y=0$"
        rhs = " + ".join(terms).replace('+ -', '- ')
        return r"$y = " + rhs + r"$"


# ---------- Plotter ----------
class Plotter:
    def __init__(self, interactive: bool = True):
        self.interactive = interactive
        if interactive:
            plt.ion()
        self.fig = plt.figure(figsize=(6.4, 4.8))

    def save_current_plot(self, filename: Optional[str] = None) -> str:
        ensure_dirs()
        if not filename:
            filename = f'plot_{timestamp_str()}.png'
        out = OUTPUT_DIR / filename
        plt.savefig(out, bbox_inches='tight', dpi=PLOT_DPI, facecolor='white')
        print(f'Saved current plot to: {out}')
        return str(out)

    def update_plot(self, df: pd.DataFrame, manual_degree: Optional[int], fitter: Fitter) -> None:
        plt.clf()
        ax = plt.gca()
        if df.empty:
            ax.set_title('Optics Lab Data (no points)')
            if self.interactive:
                plt.draw(); plt.pause(PAUSE_TIME)
            return

        has_err = not df['err'].isnull().all()
        if has_err:
            yerr_plot = np.where(np.isnan(df['err'].values), 0.0, df['err'].values)
            ax.errorbar(df['x'], df['y'], yerr=yerr_plot, fmt='o', markerfacecolor='blue', ecolor='gray', elinewidth=1.2, capsize=4, label='Data Points')
        else:
            ax.scatter(df['x'], df['y'], color='blue', label='Data Points')

        fit_deg = None
        if len(df) >= 2:
            if manual_degree is None and len(df) >= 3:
                fit_deg = fitter.auto_guess_degree(df)
                print(f"🔍 Auto-selected polynomial degree: {fit_deg}")
            elif manual_degree is not None:
                fit_deg = manual_degree

        eq_text = None
        r2_text = ''
        if fit_deg is not None:
            coeffs, poly, r2 = fitter.compute_poly_fit(df, fit_deg)
            x_vals = np.linspace(df['x'].min(), df['x'].max(), 300)
            ax.plot(x_vals, poly(x_vals), 'r-', label=f'Poly deg {fit_deg}')

            r2_text = f'$R^2 = {r2:.3f}$'
            eq_text = Fitter.format_poly_equation(coeffs)

            if fit_deg == 1 and len(df) >= 2:
                slope, intercept, r_val, p_value, std_err = linregress(df['x'].values, df['y'].values)
                print(f'📊 Regression slope: {slope:.6f} ± {std_err:.6f}')
                print(f'📊 Intercept: {intercept:.6f}')
                print(f'📊 R²: {r_val**2:.6f}')

        ax.set_xlabel('X (independent variable)')
        ax.set_ylabel('Y (dependent variable)')
        ax.set_title('Optics Lab Data')
        ax.grid(True)
        ax.legend()

        # Expand y-limits to leave room for errorbars and text
        y_min, y_max = ax.get_ylim()
        if has_err:
            err_max = np.nanmax(df['err'].values)
            padding = max(0.1 * (y_max - y_min), (err_max * 2.0) if not np.isnan(err_max) else 0.1 * (y_max - y_min))
        else:
            padding = 0.1 * (y_max - y_min)
        ax.set_ylim(y_min - padding, y_max + padding)

        # Put equation + R² in top-right
        if eq_text is not None:
            full_text = eq_text + '\n' + r2_text
            ax.text(0.98, 0.98, full_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', alpha=0.85))

        # Console stats
        n = len(df)
        mean_y = df['y'].mean()
        if n > 1:
            std_y = df['y'].std(ddof=1)
            uncertainty = std_y / np.sqrt(n)
            print(f'Mean of Y: {mean_y:.6f}')
            print(f'Std Dev of Y: {std_y:.6f}')
            print(f'Uncertainty (σ/√n): {uncertainty:.6f}')
        else:
            print(f'Mean of Y: {mean_y:.6f}')
            print('Std Dev of Y: N/A (need ≥2 points)')

        # Draw and give GUI loop a moment
        if self.interactive:
            plt.draw(); plt.pause(PAUSE_TIME)


# ---------- Interactive CLI ----------

def interactive_loop():
    dm = DataManager()
    fitter = Fitter()
    plotter = Plotter(interactive=True)

    print('Optics lab tool (cleaned). Type `help` for commands.')
    print("Commands: add points as `X,Y` or `X,Y,Err`  — other commands: bulk, poly N, auto, save, load <file>, saveplot [file], clear, list, q")

    manual_degree: Optional[int] = None

    while True:
        try:
            s = input('\n> ').strip()
            if not s:
                continue
            low = s.lower()
            if low in ('q', 'quit', 'exit'):
                print('Exiting.')
                break

            if low == 'help':
                print('Commands: add points as `X,Y` or `X,Y,Err`  — other commands: bulk, poly N, auto, save, load <file>, saveplot [file], clear, list, q')
                continue

            if low == 'list':
                if dm.df.empty:
                    print('No data points.')
                else:
                    print('idx\tx\ty\terr')
                    for i, r in dm.df.reset_index(drop=True).iterrows():
                        print(f"{i}\t{r['x']}\t{r['y']}\t{r['err']}")
                continue

            if low == 'clear':
                dm.clear()
                print('Data cleared.')
                plotter.update_plot(dm.df, manual_degree, fitter)
                continue

            if low.startswith('poly'):
                parts = s.split()
                if len(parts) >= 2:
                    try:
                        d = int(parts[1])
                        if d < 1:
                            print('Degree must be >= 1')
                        else:
                            manual_degree = d
                            print(f'Manual polynomial degree set to {d}')
                            plotter.update_plot(dm.df, manual_degree, fitter)
                    except Exception:
                        print('Usage: poly N')
                else:
                    print('Usage: poly N')
                continue

            if low == 'auto':
                manual_degree = None
                print('Returned to automatic degree selection.')
                plotter.update_plot(dm.df, manual_degree, fitter)
                continue

            if low == 'save':
                fn = input('Filename to save CSV (default will be auto-generated): ').strip()
                out = dm.export_csv(fn or None)
                print(f'Saved CSV -> {out}')
                continue

            if low.startswith('load'):
                parts = s.split(maxsplit=1)
                if len(parts) == 2:
                    fname = parts[1].strip('"\'')
                else:
                    fname = input('CSV filename to load: ').strip()
                ok, msg = dm.import_csv(fname)
                print(msg)
                if ok:
                    plotter.update_plot(dm.df, manual_degree, fitter)
                continue

            if low.startswith('saveplot'):
                parts = s.split(maxsplit=1)
                if len(parts) == 2:
                    fname = parts[1].strip('"\'')
                else:
                    fname = None
                out = plotter.save_current_plot(fname)
                print(f'Saved plot -> {out}')
                continue

            if low == 'bulk':
                print("Paste lines like: X,Y or X,Y,Err  — type 'end' on its own line to finish.")
                lines: List[str] = []
                while True:
                    line = input().strip()
                    if line.lower() == 'end':
                        break
                    if line:
                        lines.append(line)
                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    try:
                        if len(parts) == 2:
                            x, y = float(parts[0]), float(parts[1]); err = None
                        elif len(parts) >= 3:
                            x, y, err = float(parts[0]), float(parts[1]), float(parts[2])
                        else:
                            print(f'Skipping invalid: {line}'); continue
                        dm.add_point(x, y, err)
                    except ValueError:
                        print(f'Skipping invalid numeric: {line}')
                plotter.update_plot(dm.df, manual_degree, fitter)
                continue

            # Otherwise parse as single data line: X,Y or X,Y,Err
            parts = [p.strip() for p in s.split(',')]
            try:
                if len(parts) == 2:
                    x_val, y_val = float(parts[0]), float(parts[1]); err_val = None
                elif len(parts) == 3:
                    x_val, y_val, err_val = float(parts[0]), float(parts[1]), float(parts[2])
                else:
                    print('Unknown command or invalid data; type help.')
                    continue
                dm.add_point(x_val, y_val, err_val)
                plotter.update_plot(dm.df, manual_degree, fitter)
            except ValueError:
                print('Invalid numeric input — use floats like: 1.23, 4.56, 0.1')

        except KeyboardInterrupt:
            print('\nInterrupted by user.')
            break


if __name__ == '__main__':
    interactive_loop()

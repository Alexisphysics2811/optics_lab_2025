#!/usr/bin/env python3
"""
labtool_linearize_sympy_mixed_manual.py

Interactive linearizer for lab use.

Features:
 - SymPy-based model parsing for common forms (linear, power law, exponential).
 - Manual transforms with safe AST evaluation; supports mixed expressions
   referencing both x and y (e.g. x+y, x*y, log(x+y), np.log(x+y)).
 - Option in manual and model modes to force the slope during linear regression.
 - Safe environment for manual expressions (no imports, no lambdas, limited names).
 - Data import/export, caching, undo, plotting (TkAgg).
"""
from __future__ import annotations

import os
import math
import sys
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple, Sequence

import inspect

import numpy as np
import pandas as pd
from scipy.stats import linregress

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# SymPy is optional but recommended for `model` mode.
try:
    import sympy as sp
except Exception:
    sp = None

# ---------- Configuration ----------
OUTPUT_DIR = Path("./output")
CSV_OUTPUT_DIR = OUTPUT_DIR / "csv"
CACHE_PATH = OUTPUT_DIR / "cache.csv"
MAX_UNDO = 50
MIN_POINTS_FOR_FIT = 2

_SAFE_EVAL_NS = {
    'np': np,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'arcsin': np.arcsin,
    'arccos': np.arccos,
    'log': np.log,
    'log10': np.log10,
    'exp': np.exp,
    'sqrt': np.sqrt,
    'abs': np.abs,
    'pi': math.pi,
    'e': math.e,
}

import ast

def _validate_transform_ast(node, varnames: Sequence[str]):
    """Validate the AST of a user expression. Raises ValueError on unsafe nodes/names.

    Rules:
    - Allow arithmetic, unary ops, Calls, Names, Constants, Attribute access on `np` only.
    - Disallow imports, lambdas, function/class definitions, and any name not in the
      approved whitelist (or one of the variable names provided).
    """
    for n in ast.walk(node):
        # Disallowed structural nodes
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.Lambda, ast.FunctionDef, ast.ClassDef, ast.Global, ast.Nonlocal)):
            raise ValueError("Disallowed syntax in transform expression.")
        # Name checks
        if isinstance(n, ast.Name):
            if n.id not in varnames and n.id not in _SAFE_EVAL_NS and n.id != 'np':
                raise ValueError(f"Unauthorized name: {n.id}")
        # Attribute access only allowed for np.<name>
        if isinstance(n, ast.Attribute):
            if not (isinstance(n.value, ast.Name) and n.value.id == 'np'):
                raise ValueError("Only attribute access on 'np' is allowed (e.g., np.sin).")
            if n.attr.startswith('_'):
                raise ValueError("Private attribute access is not allowed.")
        # Call checks: ensure called function is either a Name in the safe list or np.attr
        if isinstance(n, ast.Call):
            func = n.func
            if isinstance(func, ast.Name):
                if func.id not in varnames and func.id not in _SAFE_EVAL_NS and func.id != 'np':
                    raise ValueError(f"Use of unauthorized function or name: {func.id}")
            elif isinstance(func, ast.Attribute):
                if not (isinstance(func.value, ast.Name) and func.value.id == 'np'):
                    raise ValueError("Only attribute calls on 'np' are allowed (e.g., np.sin()).")
                if func.attr.startswith('_'):
                    raise ValueError("Private attribute access is not allowed.")
            else:
                raise ValueError("Illegal function call in expression.")
    return True


def compile_transform(expr: str, varnames: Sequence[str]):
    """Validate expr AST and return a callable.

    - `varnames` is a sequence like ['x'] (old behavior) or ['x','y'] (mixed).
    - The returned callable will accept the same number of arguments as len(varnames).

    Raises ValueError on unsafe AST.
    """
    expr = expr.strip()
    # parse and validate
    tree = ast.parse(expr, mode='eval')
    _validate_transform_ast(tree, varnames)

    # Build safe globals for eval; include numpy as `np` and any approved short names
    safe_globals = {'__builtins__': {}}
    safe_globals.update(_SAFE_EVAL_NS)
    safe_globals['np'] = np

    # Build lambda arg list
    arg_list = ','.join(varnames)
    lam_src = f'lambda {arg_list}: {expr}'
    lam = eval(lam_src, safe_globals)
    return lam


# ---------- Utilities ----------
def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def timestamp_str():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def safe_log(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask = a > 0
    if np.any(mask):
        out[mask] = np.log(a[mask])
    return out


# ---------- Data manager ----------
class DataManager:
    def __init__(self):
        self.df = pd.DataFrame(columns=['x', 'y', 'err'])
        self._undo = []

    def push_undo(self):
        self._undo.append(self.df.copy(deep=True))
        if len(self._undo) > MAX_UNDO:
            self._undo.pop(0)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.df = self._undo.pop()
        return True

    def add_point(self, x: float, y: float, err: Optional[float] = None):
        self.push_undo()
        self.df.loc[len(self.df)] = [float(x), float(y), (float(err) if err is not None else np.nan)]

    def delete_index(self, idx: int) -> bool:
        if not (0 <= idx < len(self.df)):
            return False
        self.push_undo()
        self.df = self.df.drop(self.df.index[idx]).reset_index(drop=True)
        return True

    def edit_index(self, idx: int, x: float, y: float, err: Optional[float] = None) -> bool:
        if not (0 <= idx < len(self.df)):
            return False
        self.push_undo()
        self.df.loc[self.df.index[idx], ['x', 'y', 'err']] = [float(x), float(y), (float(err) if err is not None else np.nan)]
        return True

    def clear(self) -> bool:
        if len(self.df) == 0:
            return False
        self.push_undo()
        self.df = self.df.iloc[0:0].copy()
        return True

    def save_cache(self) -> bool:
        ensure_dirs()
        if len(self.df) == 0:
            if CACHE_PATH.exists():
                CACHE_PATH.unlink()
            return True
        self.df.to_csv(CACHE_PATH, index=False)
        return True

    def load_cache(self) -> bool:
        if not CACHE_PATH.exists():
            return False
        df = pd.read_csv(CACHE_PATH)
        if 'x' not in df.columns or 'y' not in df.columns:
            return False
        if 'err' not in df.columns:
            df['err'] = np.nan
        self.push_undo()
        self.df = df[['x', 'y', 'err']].copy()
        return True

    def import_csv(self, path: str) -> Tuple[bool, str]:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            return False, f'Failed to read CSV: {e}'
        if 'x' in df.columns and 'y' in df.columns:
            if 'err' not in df.columns:
                df['err'] = np.nan
            df = df[['x', 'y', 'err']].copy()
        else:
            cols = list(df.columns)
            if len(cols) >= 2:
                df2 = df.rename(columns={cols[0]: 'x', cols[1]: 'y'})
                if 'err' not in df2.columns:
                    df2['err'] = np.nan
                df = df2[['x', 'y', 'err']].copy()
            else:
                return False, 'CSV must contain at least two columns'
        self.push_undo()
        self.df = df.reset_index(drop=True)
        return True, f'Imported {len(self.df)} points.'

    def export_csv(self, filename: Optional[str] = None) -> str:
        ensure_dirs()
        if filename is None:
            filename = f'lab_data_{timestamp_str()}.csv'
        out = CSV_OUTPUT_DIR / filename
        self.df.to_csv(out, index=False)
        return str(out)


# ---------- Plotter ----------
class Plotter:
    def __init__(self, interactive: bool = True):
        self.interactive = interactive
        if interactive:
            plt.ion()

    def preview_raw(self, df: pd.DataFrame):
        plt.clf()
        if len(df) == 0:
            plt.title('No data')
            plt.draw(); plt.gcf().canvas.flush_events()
            return
        if not df['err'].isnull().all():
            yerr = np.where(np.isnan(df['err'].values), 0.0, df['err'].values)
            plt.errorbar(df['x'], df['y'], yerr=yerr, fmt='o', label='Data (raw)')
        else:
            plt.scatter(df['x'], df['y'], label='Data (raw)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Raw Data')
        plt.grid(True)
        plt.legend()
        plt.draw(); plt.gcf().canvas.flush_events()

    def _call_transform(self, fn: Callable, x_arr: np.ndarray, y_arr: np.ndarray):
        """Call transform `fn` with either one argument (array) or two (x,y arrays).

        Returns the resulting array.
        """
        try:
            sig = inspect.signature(fn)
            params = len(sig.parameters)
        except Exception:
            # fallback: try to call with two args then one arg
            params = None
        if params == 1:
            return fn(x_arr)
        if params == 2:
            return fn(x_arr, y_arr)
        # unknown signature: try two-arg, then one-arg
        try:
            return fn(x_arr, y_arr)
        except TypeError:
            return fn(x_arr)

    def plot_transformed(self, x: np.ndarray, y: np.ndarray, fx: Callable, fy: Callable,
                         fx_desc: str, fy_desc: str, err_y=None, force_slope: Optional[float] = None, save_path: Optional[str] = None) -> Dict:
        try:
            X = self._call_transform(fx, x, y)
            Y = self._call_transform(fy, x, y)
        except Exception as e:
            raise RuntimeError(f'Transform application failed: {e}')
        mask = np.isfinite(X) & np.isfinite(Y)
        if np.count_nonzero(mask) < 2:
            raise RuntimeError('Not enough finite transformed points to fit')
        Xc, Yc = X[mask], Y[mask]

        # If a forced slope is provided, compute the best intercept for that slope
        if force_slope is None:
            slope, intercept, r_val, p_value, std_err = linregress(Xc, Yc)
        else:
            slope = float(force_slope)
            # best intercept (least squares) for fixed slope: b = mean(Y) - m * mean(X)
            intercept = float(np.mean(Yc) - slope * np.mean(Xc))
            # compute correlation and set a placeholder std_err (slope std_err undefined when forced)
            r_val = float(np.corrcoef(Xc, Yc)[0, 1]) if len(Xc) > 1 else 0.0
            std_err = 0.0

        y_fit = intercept + slope * Xc
        ss_tot = np.sum((Yc - np.mean(Yc)) ** 2)
        ss_res = np.sum((Yc - y_fit) ** 2)
        r2 = 0.0 if ss_tot == 0 else max(0.0, 1 - ss_res / ss_tot)

        plt.clf()
        plt.scatter(Xc, Yc, label='Data (transformed)')
        x_vals_plot = np.linspace(np.min(Xc), np.max(Xc), 300)
        plt.plot(x_vals_plot, intercept + slope * x_vals_plot, '--', label='Linear fit')
        plt.xlabel(fx_desc)
        plt.ylabel(fy_desc)
        title_extra = f' (forced slope={force_slope:.5g})' if force_slope is not None else ''
        plt.title('Linearized Plot' + title_extra)
        plt.grid(True)
        plt.legend()

        # build equation text (show when slope forced)
        if force_slope is not None:
            eq = f'Y = {slope:.5g} X + {intercept:.5g} (slope forced)\nR^2 = {r2:.4f}'
        else:
            eq = f'Y = {slope:.5g} X + {intercept:.5g}\nR^2 = {r2:.4f}'

        plt.gca().text(
            0.98, 0.98, eq,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        if save_path:
            ensure_dirs()
            plt.savefig(save_path, bbox_inches='tight')
        if self.interactive:
            plt.draw(); plt.gcf().canvas.flush_events()
        return {'slope': slope, 'intercept': intercept, 'r_val': r_val, 'std_err': std_err, 'r2': r2, 'forced_slope': force_slope}

    def save_raw_with_model(self, x: np.ndarray, y: np.ndarray, err_y, model_params: Optional[dict], model_type: Optional[str], save_path: str):
        plt.figure()
        if err_y is not None:
            plt.errorbar(x, y, yerr=err_y, fmt='o', label='Data (raw)')
        else:
            plt.scatter(x, y, label='Data (raw)')
        if model_params and model_type:
            x_plot = np.linspace(np.min(x), np.max(x), 400)
            try:
                if model_type == 'power':
                    k = float(model_params.get('k'))
                    a = float(model_params.get('A') or model_params.get('a'))
                    plt.plot(x_plot, k * (x_plot ** a), '--', label=f'model k*x^{a:.4g}')
                elif model_type == 'exp':
                    k = float(model_params.get('k'))
                    a = float(model_params.get('a'))
                    plt.plot(x_plot, k * np.exp(a * x_plot), '--', label='model k*exp(a*x)')
                elif model_type == 'linear':
                    m = float(model_params.get('m'))
                    b = float(model_params.get('b'))
                    plt.plot(x_plot, m * x_plot + b, '--', label=f'model {m:.4g}*x + {b:.4g}')
            except Exception:
                pass
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Raw Data (with model)')
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


# ---------- SymPy model parser (central to this rewrite) ----------
class SympyModelParser:
    """Conservative recognition of common model forms using SymPy.

    Recognized forms (examples):
      - Linear: y = m*x + b  -> fx = x, fy = y, recover m,b
      - Power law: y = k * x**a  -> fx = ln(x), fy = ln(y)  (k, a recoverable)
      - Exponential: y = k * exp(a*x) -> fx = x, fy = ln(y)  (k, a recoverable)
      - Inverse power: y = k / x**a  -> detected as power with negative exponent
    """

    @staticmethod
    def parse(model_str: str) -> Optional[Dict]:
        if sp is None:
            print('SymPy not available (install sympy).')
            return None
        try:
            # Normalize
            if '=' in model_str:
                left_s, right_s = model_str.split('=', 1)
            else:
                left_s = 'y'; right_s = model_str
            left_s = left_s.strip(); right_s = right_s.strip()

            # Define symbols
            x_s, y_s = sp.symbols('x y')
            # Provide common symbols for params
            k_s, a_s, m_s, b_s = sp.symbols('k a m b')
            local = {'k': k_s, 'a': a_s, 'm': m_s, 'b': b_s, 'exp': sp.exp}

            lhs = sp.sympify(left_s, locals=local)
            rhs = sp.sympify(right_s, locals=local)

            # If lhs is not y (or single symbol) try to canonicalize
            # We want expression of the form y = f(x, params)
            # For safety, only handle single-symbol lhs
            if len(lhs.free_symbols) != 1:
                # fallback
                return None

            # get the rhs as function of x
            # Attempt linear detection: if rhs is affine in x -> linear
            # Fitable into y = m*x + b
            # Use sympy to attempt to collect rhs with x
            rhs_simpl = sp.simplify(sp.expand(rhs))

            # Linear? check whether rhs is a polynomial in x of degree 1
            if rhs_simpl.is_polynomial(x_s) and sp.degree(rhs_simpl, x_s) == 1:
                # extract coeffs
                m_coeff = sp.Poly(rhs_simpl, x_s).coeff_monomial(x_s)
                b_coeff = sp.Poly(rhs_simpl, x_s).coeff_monomial(1)
                def fx(arr): return arr
                def fy(arr): return arr
                def recover(slope, intercept):
                    return {'m': float(slope), 'b': float(intercept)}
                return {'fx': fx, 'fy': fy, 'fx_desc': 'x', 'fy_desc': 'y', 'recover': recover, 'type': 'linear', 'info': 'Linear in x'}

            # Power-law detection: attempt to take log and collect log(x)
            # Take care of cases: y = k*x**a  OR y = k / x**a
            try:
                L = sp.log(rhs_simpl)
                L_simpl = sp.simplify(sp.expand(L))
                logx = sp.log(x_s)
                A = sp.simplify(sp.expand(sp.collect(L_simpl, logx).coeff(logx, 1)))
                B = sp.simplify(L_simpl - A * logx)
                # If A and B do not contain x, it's power-like
                if not (A.free_symbols & {x_s}) and not (B.free_symbols & {x_s}):
                    def fx(arr): return safe_log(arr)
                    def fy(arr): return safe_log(arr)
                    def recover(slope, intercept):
                        return {'A': float(slope), 'k': float(math.exp(intercept))}
                    return {'fx': fx, 'fy': fy, 'fx_desc': 'ln(x)', 'fy_desc': 'ln(y)', 'recover': recover, 'type': 'power', 'info': 'Power-like'}
            except Exception:
                pass

            # Exponential detection: check if log(rhs) is affine in x
            try:
                L = sp.log(rhs_simpl)
                L_simpl = sp.simplify(sp.expand(L))
                A2 = sp.simplify(sp.expand(sp.collect(L_simpl, x_s).coeff(x_s, 1)))
                B2 = sp.simplify(L_simpl - A2 * x_s)
                if not (A2.free_symbols & {x_s}) and not (B2.free_symbols & {x_s}):
                    def fx(arr): return arr
                    def fy(arr): return safe_log(arr)
                    def recover(slope, intercept):
                        return {'a': float(slope), 'k': float(math.exp(intercept))}
                    return {'fx': fx, 'fy': fy, 'fx_desc': 'x', 'fy_desc': 'ln(y)', 'recover': recover, 'type': 'exp', 'info': 'Exponential-like'}
            except Exception:
                pass

            # Fallback: not recognized
            return None
        except Exception as e:
            print('SymPy parse error:', e)
            return None


# ---------- Interactive CLI ----------
HELP_TEXT = """
Commands:
  X,Y [,Err]       Add a point (comma separated). Examples: 1,2  |  1,2,0.1
  bulk             Paste multiline X,Y[,Err] lines; type 'end' on its own line to finish
  import           Load CSV (expects x,y[,err] or first two columns)
  save             Export CSV to ./output/csv/
  cache            Save current data to cache
  loadcache        Load cache (replaces current data)
  clearcache       Remove cache file
  list             Show current points and indices
  edit             Edit or delete a point by index
  undo             Revert last change (add/edit/delete/import/bulk/clear/loadcache)
  manual           Enter manual transform expressions (safe namespace). Mixed expressions allowed: e.g. X transform: x+y  Y transform: x*y
  model            Enter symbolic model (SymPy required) to auto-choose transforms
  clear            Clear all data (pushes undo)
  q                Quit
"""


def interactive_loop():
    dm = DataManager()
    plotter = Plotter(interactive=True)

    ensure_dirs()
    if CACHE_PATH.exists():
        try:
            df = pd.read_csv(CACHE_PATH)
            if 'x' in df.columns and 'y' in df.columns:
                if 'err' not in df.columns:
                    df['err'] = np.nan
                dm.df = df[['x', 'y', 'err']].copy()
                print(f'Loaded {len(dm.df)} cached points from {CACHE_PATH}')
        except Exception:
            pass

    print('Interactive linearizer (SymPy-based). Type `help` for commands.')
    print(HELP_TEXT)

    while True:
        try:
            user = input('> ').strip()
            if not user:
                continue
            cmd = user.lower()
            if cmd == 'q':
                print('Goodbye.')
                break
            if cmd == 'help':
                print(HELP_TEXT)
                continue

            if cmd == 'list':
                if len(dm.df) == 0:
                    print('No data points.')
                else:
                    print('idx\tx\ty\terr')
                    for i, r in dm.df.reset_index(drop=True).iterrows():
                        print(f"{i}\t{r['x']}\t{r['y']}\t{r['err']}")
                continue

            if cmd == 'undo':
                if dm.undo():
                    dm.save_cache()
                    print('Undo applied.')
                    plotter.preview_raw(dm.df)
                else:
                    print('Nothing to undo.')
                continue

            if cmd == 'save':
                path = dm.export_csv()
                print(f'Saved CSV to: {path}')
                continue

            if cmd == 'cache':
                dm.save_cache()
                print('Cache saved.')
                continue

            if cmd == 'loadcache':
                if not CACHE_PATH.exists():
                    print('No cache present.')
                    continue
                confirm = input('Load cache and replace current data? (y/N): ').strip().lower()
                if confirm != 'y':
                    print('Cancelled.')
                    continue
                dm.push_undo()
                if dm.load_cache():
                    dm.save_cache()
                    print('Cache loaded.')
                    plotter.preview_raw(dm.df)
                else:
                    print('Failed to load cache.')
                continue

            if cmd == 'clearcache':
                if CACHE_PATH.exists():
                    CACHE_PATH.unlink()
                    print('Cache removed.')
                else:
                    print('No cache to remove.')
                continue

            if cmd == 'clear':
                if dm.clear():
                    dm.save_cache()
                    print('Data cleared (undo available).')
                    plt.clf(); plt.draw()
                else:
                    print('Data already empty.')
                continue

            if cmd == 'import':
                path = input('CSV path: ').strip()
                ok, msg = dm.import_csv(path)
                print(msg)
                if ok:
                    dm.save_cache()
                    plotter.preview_raw(dm.df)
                continue

            if cmd == 'bulk':
                print("Paste lines: X,Y or X,Y,Err  (type 'end' alone to finish)")
                lines = []
                while True:
                    line = input().strip()
                    if line.lower() == 'end':
                        break
                    if line:
                        lines.append(line)
                if not lines:
                    print('No lines pasted.')
                    continue
                dm.push_undo()
                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    try:
                        if len(parts) == 2:
                            x, y = float(parts[0]), float(parts[1]); err = np.nan
                        elif len(parts) >= 3:
                            x, y, err = float(parts[0]), float(parts[1]), float(parts[2])
                        else:
                            print(f'Skipping invalid: {line}'); continue
                        dm.df.loc[len(dm.df)] = [x, y, err]
                    except Exception:
                        print(f'Skipping invalid numeric: {line}')
                dm.save_cache()
                print(f'Now {len(dm.df)} points.')
                plotter.preview_raw(dm.df)
                continue

            if cmd == 'edit':
                if len(dm.df) == 0:
                    print('No data to edit.')
                    continue
                print('Current data:')
                for i, r in dm.df.reset_index(drop=True).iterrows():
                    print(f"{i}: {r['x']}, {r['y']}, {r['err']}")
                sel = input("Index to edit (or 'd' to delete):").strip().lower()
                if sel == 'd':
                    idxs = input('Index to delete: ').strip()
                    try:
                        idx = int(idxs)
                        if dm.delete_index(idx):
                            dm.save_cache()
                            print('Deleted.')
                            plotter.preview_raw(dm.df)
                        else:
                            print('Index out of range.')
                    except Exception:
                        print('Invalid index.')
                    continue
                try:
                    idx = int(sel)
                    if not (0 <= idx < len(dm.df)):
                        print('Index out of range.')
                        continue
                except Exception:
                    print('Invalid selection.')
                    continue
                cur = dm.df.reset_index(drop=True).iloc[idx]
                print(f"Current: {cur['x']}, {cur['y']}, {cur['err']}")
                new = input('Enter new X,Y or X,Y,Err (leave blank to cancel): ').strip()
                if not new:
                    print('Cancelled.')
                    continue
                parts = [p.strip() for p in new.split(',')]
                try:
                    if len(parts) == 2:
                        x, y = float(parts[0]), float(parts[1]); err = np.nan
                    elif len(parts) >= 3:
                        x, y, err = float(parts[0]), float(parts[1]), float(parts[2])
                    else:
                        print('Invalid format.'); continue
                    dm.edit_index(idx, x, y, (None if math.isnan(err) else err))
                    dm.save_cache()
                    print('Updated.')
                    plotter.preview_raw(dm.df)
                except Exception:
                    print('Invalid numeric values.')
                continue

            if cmd == 'manual':
                if len(dm.df) < MIN_POINTS_FOR_FIT:
                    print('Need at least 2 points for manual transform.')
                    continue
                print("Enter transform expressions using 'x' and 'y' and allowed names: np, sin, cos, log, exp, sqrt, abs, pi, e, arcsin")
                print("You may use mixed expressions referencing both x and y. Examples: x+y, x*y, x/(y+1), log(x+y)")
                x_expr = input('X transform (e.g. log(x) or x+y): ').strip()
                y_expr = input('Y transform (e.g. log(y) or x*y): ').strip()
                try:
                    fx = compile_transform(x_expr, ['x', 'y'])
                    fy = compile_transform(y_expr, ['x', 'y'])
                except Exception as e:
                    print('Failed to parse transforms:', e)
                    continue
                x = dm.df['x'].values; y = dm.df['y'].values
                err_y = None
                if not dm.df['err'].isnull().all():
                    err_y = np.where(np.isnan(dm.df['err'].values), 0.0, dm.df['err'].values)
                try:
                    force_input = input("Force slope? Enter numeric slope or leave blank for none: ").strip()
                    force_slope = None
                    if force_input != '':
                        try:
                            force_slope = float(force_input)
                        except Exception:
                            print('Invalid slope value; continuing without forcing.')
                            force_slope = None
                    res = plotter.plot_transformed(x, y, fx, fy, x_expr, y_expr, err_y=err_y, force_slope=force_slope)
                    ts = timestamp_str()
                    raw_path = str(OUTPUT_DIR / f'raw_manual_{ts}.png')
                    lin_path = str(OUTPUT_DIR / f'linearized_manual_{ts}.png')
                    plotter.save_raw_with_model(x, y, err_y, None, None, raw_path)
                    plotter.plot_transformed(x, y, fx, fy, x_expr, y_expr, err_y=err_y, force_slope=force_slope, save_path=lin_path)
                    print(f'Saved raw -> {raw_path}\nSaved linearized -> {lin_path}')
                except Exception as e:
                    print('Plot/fit failed:', e)
                continue

            if cmd == 'model':
                if sp is None:
                    print('SymPy not available. Install sympy: pip install sympy')
                    continue
                model_str = input("Model (e.g. y = k*x**a or y = k*exp(a*x)): ").strip()
                parsed = SympyModelParser.parse(model_str)
                if parsed is None:
                    print('Could not parse model automatically; it may be unsupported. Try manual transforms.')
                    continue
                fx = parsed['fx']; fy = parsed['fy']; fx_desc = parsed['fx_desc']; fy_desc = parsed['fy_desc']
                x = dm.df['x'].values; y = dm.df['y'].values
                err_y = None
                if not dm.df['err'].isnull().all():
                    err_y = np.where(np.isnan(dm.df['err'].values), 0.0, dm.df['err'].values)
                try:
                    force_input = input("Force slope? Enter numeric slope or leave blank for none: ").strip()
                    force_slope = None
                    if force_input != '':
                        try:
                            force_slope = float(force_input)
                        except Exception:
                            print('Invalid slope value; continuing without forcing.')
                            force_slope = None
                    res = plotter.plot_transformed(x, y, fx, fy, fx_desc, fy_desc, err_y=err_y, force_slope=force_slope)
                    # recover
                    recovered = None
                    fit_slope = res['slope']; fit_intercept = res['intercept']
                    try:
                        if 'recover' in parsed and parsed['recover']:
                            recovered = parsed['recover'](fit_slope, fit_intercept)
                    except Exception:
                        recovered = None
                    # save
                    ts = timestamp_str()
                    raw_path = str(OUTPUT_DIR / f'raw_model_{ts}.png')
                    lin_path = str(OUTPUT_DIR / f'linearized_model_{ts}.png')
                    plotter.save_raw_with_model(x, y, err_y, recovered, parsed.get('type'), raw_path)
                    plotter.plot_transformed(x, y, fx, fy, fx_desc, fy_desc, err_y=err_y, force_slope=force_slope, save_path=lin_path)
                    print(f'Saved raw -> {raw_path}\nSaved linearized -> {lin_path}')
                    if recovered:
                        print('Recovered params:')
                        for k, v in recovered.items():
                            print(f'  {k} = {v}')
                except Exception as e:
                    print('Model fit failed:', e)
                continue

            # default: parse as data point
            parts = [p.strip() for p in user.split(',')]
            try:
                if len(parts) == 2:
                    x, y = float(parts[0]), float(parts[1]); err = np.nan
                elif len(parts) >= 3:
                    x, y, err = float(parts[0]), float(parts[1]), float(parts[2])
                else:
                    print('Unknown command. Type `help` for available commands.')
                    continue
                dm.add_point(x, y, (None if math.isnan(err) else err))
                dm.save_cache()
                print(f'Added point: x={x}, y={y}, err={err}')
                plotter.preview_raw(dm.df)
            except Exception:
                print('Invalid numeric input. Use floats like: 1.23, 4.56, 0.1')

        except KeyboardInterrupt:
            print('\nInterrupted by user. Exiting interactive loop.')
            break


if __name__ == '__main__':
    interactive_loop()

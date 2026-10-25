#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper 2 / SSTE: empirical cross-basis (DLNM) comparison.

Publication-oriented, self-contained Python implementation. This is a
transparent DLNM-style tensor cross-basis implementation, NOT a direct call to
Gasparrini's R dlnm package. For an exact R-package reproduction, independently
cross-check crossbasis construction and fitted contrasts using R/dlnm.

MODEL AND ESTIMAND
------------------
    asinh outcome, standardized within country ~ calendar spline +
    sum_{lag=0}^L f(standardized asinh cases at t-lag, lag) + ARMA(2,1) error.

The primary reference uses an unrestricted seven-position linear lag vector
(L=6); the matched DLNM uses linear exposure and a cubic B-spline lag basis.
Nonlinear DLNM contrasts compare exposure z=1 to z=0 standardized transformed
case units. They are *conditional mean contrasts*, not log-relative risks,
patient-level delays, rate ratios, causal effects, or forecasts.

No outcome coefficient or inferential finding is hardcoded. Do not interpret a
fit's 95% interval as calibrated without independent coverage validation.

Typical Google Colab commands (one cell):
    !pip -q install numpy pandas scipy statsmodels patsy matplotlib requests
    !python /content/Paper2_empirical_DLNM_comparison.py --mode quick
    !python /content/Paper2_empirical_DLNM_comparison.py --mode full
    from google.colab import files
    files.download('/content/Paper2_DLNM_comparison/Paper2_DLNM_outputs.zip')

With an earlier manually-downloaded pinned OWID CSV:
    !python /content/Paper2_empirical_DLNM_comparison.py \\
       --data /content/owid-covid-data.csv --mode full

Smoke test without downloading external inputs:
    python Paper2_empirical_DLNM_comparison.py --self-test

Interpretation: 11 geographic countries x 2 indicators (hospital admissions,
occupancy), not 22 different diseases. Dates and input definitions are
historical surveillance records; no inference about clinical delays.

Dependencies: numpy, pandas, scipy, statsmodels, patsy, matplotlib, requests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.stats import norm
import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.numdiff import approx_hess
import patsy
from patsy import dmatrix, build_design_matrices
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests

# Historical OWID commit as cited in the SSTE manuscript (not a mutable URL).
OWID_COMMIT = 'f0d53320b19703cf565940d8269181a8423d7032'
OWID_URL = ('https://raw.githubusercontent.com/owid/covid-19-data/' + OWID_COMMIT +
            '/public/data/owid-covid-data.csv')
SOURCE_REPO = ('ChildOfTheNorth/paper2-spatiotemporal-validation@'
               'df9e6feebf1ddef8da699deec8b615c92fdd278f')
INPUT_COLUMNS = ['iso_code', 'location', 'date', 'new_cases_per_million',
                 'weekly_hosp_admissions_per_million', 'hosp_patients_per_million']
COUNTRIES = ['Belgium', 'Czechia', 'Denmark', 'France', 'Israel', 'Italy',
             'Malaysia', 'Spain', 'Switzerland', 'United Kingdom', 'United States']
OUTCOMES = {'admissions': 'weekly_hosp_admissions_per_million',
            'occupancy': 'hosp_patients_per_million'}
STUDY_START = '2020-01-01'
STUDY_END = '2024-08-14'
SCHEMA_VERSION = 'dlnm_sste_v1.3.1_archived_window_penalty_rescue'
EXPECTED_WEEKS = 1882
EXPECTED_BY_COUNTRY = {
    'Belgium':172, 'Czechia':229, 'Denmark':157, 'France':157,
    'Israel':166, 'Italy':210, 'Malaysia':224, 'Spain':149,
    'Switzerland':144, 'United Kingdom':127, 'United States':147}

BLUE = '#0072B2'
ORANGE = '#D55E00'
GREEN = '#009E73'
PURPLE = '#CC79A7'
SKY = '#56B4E9'
COLORS = [BLUE, ORANGE, GREEN, PURPLE, SKY, '#E69F00', '#000000']

LOGGER = logging.getLogger('sste_dlnm')


@dataclass(frozen=True)
class Specification:
    name: str
    max_lag: int = 6
    lag_kind: str = 'identity'  # identity, cubic, cubic_knots
    lag_df: int = 4
    knots: tuple[float, ...] = ()
    exp_kind: str = 'linear'  # linear, natural_cubic
    exp_df: int = 4
    penalty_lambda: float = 0.0


def specifications(mode: str) -> list[Specification]:
    out = [Specification('primary_unrestricted', 6, 'identity', 7),
           Specification('dlnm_linear_cubic_df4', 6, 'cubic', 4),
           Specification('dlnm_nonlinear_natural_df4', 6, 'cubic', 4,
                         exp_kind='natural_cubic', exp_df=4)]
    if mode == 'full':
        out.extend([
            Specification('dlnm_linear_lag4', 4, 'cubic', 4),
            Specification('dlnm_linear_lag8', 8, 'cubic', 4),
            Specification('dlnm_linear_knots_1_3', 6, 'cubic_knots', 6, (1, 3)),
            Specification('dlnm_linear_knots_2_4', 6, 'cubic_knots', 6, (2, 4)),
            Specification('dlnm_linear_knots_3_5', 6, 'cubic_knots', 6, (3, 5)),
            Specification('dlnm_nonlinear_knots_2_4', 6, 'cubic_knots', 6, (2, 4),
                          'natural_cubic', 4),
            Specification('dlnm_penalized_lag_lambda1', 6, 'cubic_knots', 6,
                          (2, 4), 'linear', 4, 1.0),
        ])
    return out

def make_dirs(root: Path):
    for sub in ['', 'data', 'tables', 'figures', 'text', 'logs', 'checkpoints']:
        (root/sub).mkdir(parents=True, exist_ok=True)


def setup_logging(root: Path):
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.propagate = False  # Avoid duplicate Colab/Jupyter root-logger lines.
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOGGER.addHandler(sh)
    fh = logging.FileHandler(root/'logs'/'run.log', encoding='utf-8')
    fh.setFormatter(fmt)
    LOGGER.addHandler(fh)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download_source(path: Path):
    if path.exists() and path.stat().st_size > 10_000_000:
        LOGGER.info('Using downloaded input: %s', path)
        return
    LOGGER.info('Downloading pinned OWID file (~100 MB): %s', OWID_URL)
    tmp = path.with_suffix('.download')
    with requests.get(OWID_URL, stream=True, timeout=(40, 240)) as r:
        r.raise_for_status()
        with tmp.open('wb') as f:
            for block in r.iter_content(1024*1024):
                if block:
                    f.write(block)
    if tmp.stat().st_size < 10_000_000:
        raise IOError('Unexpectedly small OWID CSV; download incomplete')
    tmp.replace(path)


def collapse_duplicates(raw: pd.DataFrame) -> pd.DataFrame:
    """Merge identical/compatible country-date duplicate records; reject conflict."""
    key = ['iso_code', 'date']
    if not raw.duplicated(key).any():
        return raw
    count = int(raw.duplicated(key).sum())
    LOGGER.warning('Found %s duplicate country-date rows; auditing them.', count)
    result = []
    for _, g in raw.groupby(key, sort=False, dropna=False):
        if len(g) == 1:
            result.append(g.iloc[0].to_dict())
            continue
        out = g.iloc[0].to_dict()
        for col in raw.columns:
            values = g[col].dropna().unique()
            if len(values) > 1:
                raise ValueError(f'Conflicting duplicate {tuple(g.iloc[0][key])} / {col}: {values[:3]}')
            out[col] = values[0] if len(values) else np.nan
        result.append(out)
    return pd.DataFrame(result, columns=raw.columns)


def load_daily(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    LOGGER.info('Reading source CSV %s', source)
    raw = pd.read_csv(source, usecols=lambda c: c in INPUT_COLUMNS,
                      low_memory=False, dtype={'iso_code': 'string', 'location': 'string'})
    absent = set(INPUT_COLUMNS)-set(raw.columns)
    if absent:
        raise ValueError(f'Missing required columns: {sorted(absent)}')
    raw['date'] = pd.to_datetime(raw['date'], errors='raise')
    raw = raw[raw.iso_code.notna() & ~raw.iso_code.str.startswith('OWID_')].copy()
    for col in INPUT_COLUMNS[3:]:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')
        raw.loc[raw[col] < 0, col] = np.nan
    raw = collapse_duplicates(raw)
    # The manuscript screened nonmissing DAILY records *before* truncating to
    # the 2020-2024 study window. Hold that order exactly.
    daily_counts = raw.groupby('iso_code')[INPUT_COLUMNS[3:]].count()
    eligible_codes = set(daily_counts.index[daily_counts.min(axis=1) >= 300])
    screen = raw[raw.iso_code.isin(eligible_codes)].copy()
    screen = screen[(screen.date >= STUDY_START) & (screen.date <= STUDY_END)]
    mapping = screen[['iso_code', 'location']].drop_duplicates()
    LOGGER.info('Daily screening retained %d ISO codes before weekly eligibility',
                len(eligible_codes))
    # Use manuscript countries, and log additions/omissions rather than
    # quietly forcing the expected answer.
    summary = (daily_counts.assign(screen_pass=daily_counts.index.isin(eligible_codes))
               .reset_index())
    summary['location'] = summary.iso_code.map(
        raw.groupby('iso_code')['location'].first().to_dict())
    screen = screen.sort_values(['iso_code', 'date'])
    return screen, summary


def strict_weekly(country: pd.DataFrame) -> pd.DataFrame:
    country = country.sort_values('date').set_index('date')
    idx = pd.date_range(country.index.min(), country.index.max(), freq='D')
    country = country.reindex(idx)
    # Monday-Sunday W-SUN; cases and occupancy require precisely 7 observed
    # daily values, unlike admissions, which is already a rolling 7-day total.
    cases = country['new_cases_per_million'].resample('W-SUN')
    occ = country['hosp_patients_per_million'].resample('W-SUN')
    w = pd.DataFrame(index=cases.sum().index)
    w['cases'] = cases.sum(min_count=7).where(cases.count().eq(7))
    w['occupancy'] = occ.mean().where(occ.count().eq(7))
    sundays = country.loc[country.index.weekday == 6,
                          'weekly_hosp_admissions_per_million']
    w['admissions'] = sundays.reindex(w.index)
    w = w[w.index <= pd.Timestamp(STUDY_END)].copy()
    w.index.name = 'week_sunday'
    return w


def scale_series(z: pd.Series) -> tuple[pd.Series, dict]:
    transformed = pd.Series(np.arcsinh(z.astype(float)), index=z.index)
    mean = float(transformed.mean(skipna=True))
    sd = float(transformed.std(skipna=True, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError('Invalid outcome/exposure standard deviation')
    return (transformed-mean)/sd, {'asinh_mean':mean,'asinh_sd':sd,
                                   'weekly_nonmissing':int(transformed.count())}


def build_countries(daily: pd.DataFrame, root: Path):
    """Build entire weekly calendar, fixed 8-week mask, standardized outcomes."""
    out = {}
    audit = []
    for name in COUNTRIES:
        g = daily.loc[daily.location.eq(name)]
        if g.empty:
            audit.append(dict(country=name, status='not_in_daily_screen', n_common=0))
            continue
        weekly = strict_weekly(g)
        weekly['x'], sc_x = scale_series(weekly.cases)
        sc_y = {}
        for o in OUTCOMES:
            weekly['y_'+o], sc_y[o] = scale_series(weekly[o])
        available = weekly[['cases','admissions','occupancy']].notna().all(axis=1)
        history = weekly.cases.notna().copy()
        for lag in range(1,9):
            history &= weekly.cases.shift(lag).notna()
        weekly['common'] = available & history
        n = int(weekly.common.sum())
        audit.append(dict(country=name, status='included' if n>=80 else 'under_80',
                          n_common=n, reference_n=EXPECTED_BY_COUNTRY[name],
                          matches_reference=(n==EXPECTED_BY_COUNTRY[name]), n_weeks=len(weekly),
                          n_paired=int(available.sum()),
                          cases_available=int(weekly.cases.count()),
                          admissions_available=int(weekly.admissions.count()),
                          occupancy_available=int(weekly.occupancy.count()),
                          x_mean=sc_x['asinh_mean'], x_sd=sc_x['asinh_sd'],
                          adm_mean=sc_y['admissions']['asinh_mean'],
                          adm_sd=sc_y['admissions']['asinh_sd'],
                          occ_mean=sc_y['occupancy']['asinh_mean'],
                          occ_sd=sc_y['occupancy']['asinh_sd']))
        if n >= 80:
            out[name] = weekly
            weekly.to_csv(root/'data'/f"weekly_{name.lower().replace(' ','_')}.csv")
        LOGGER.info('%s: %d paired-common weeks; %d weekly calendar entries',
                    name,n,len(weekly))
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(root/'tables'/'01_data_inclusion_audit.csv', index=False)
    total = int(audit_df.n_common.loc[audit_df.status.eq('included')].sum())
    included = int(audit_df.status.eq('included').sum())
    country_match=all(int(r.n_common)==EXPECTED_BY_COUNTRY[r.country]
                      for r in audit_df.itertuples() if r.country in EXPECTED_BY_COUNTRY)
    if total != EXPECTED_WEEKS or included != 11 or not country_match:
        LOGGER.error('MANUSCRIPT-REPLICATION WARNING: calculated %s country-weeks/%s countries, '
                     'expected 1882/11. One or more country counts may also disagree; '
                     'inspect 01_data_inclusion_audit.csv. Do not claim archival replication.',total,included)
    else:
        LOGGER.info('Inclusion check matches manuscript: 11 countries, 1882 country-weeks.')
    return out, audit_df, (total == EXPECTED_WEEKS and included == 11 and country_match)

def make_trend(n: int) -> np.ndarray:
    # Calendar time is the full W-SUN index, not the compressed observed rows.
    t = np.linspace(0.0, 1.0, n)
    tr = np.asarray(dmatrix('bs(t, df=8, degree=3, include_intercept=False) - 1',
                            {'t': t}))
    return np.column_stack([np.ones(n), tr])


def identifiable_trend_columns(tr: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    """Remove only trend columns that are unidentifiable on the shared sample.

    The calendar spline is constructed on the full weekly time axis, but many
    countries only report for part of that axis. Some columns can therefore be
    zero or linearly dependent *on observed weeks*. Greedy rank selection,
    with the intercept retained first, preserves the column span of the trend
    design on those weeks; no observations or exposure lags are changed.

    This is an identifiability repair, not an exact archived-model replay.
    Archived coefficient validation is still required before publication.
    """
    observed = np.asarray(observed_mask, bool)
    if observed.ndim != 1 or len(observed) != tr.shape[0]:
        raise ValueError('Invalid observed mask for trend identification')
    if observed.sum() <= tr.shape[1]:
        raise ValueError('Too few observations for specified trend model')
    kept = [0]  # Always keep the intercept, which is column 0.
    current_rank = np.linalg.matrix_rank(tr[observed][:, kept])
    for col in range(1, tr.shape[1]):
        candidate = kept + [col]
        rank = np.linalg.matrix_rank(tr[observed][:, candidate])
        if rank > current_rank:
            kept.append(col)
            current_rank = rank
    dropped = [i for i in range(tr.shape[1]) if i not in kept]
    if dropped:
        LOGGER.warning('Calendar spline rank repair: %d/%d identified columns; '
                       'dropped trend-only columns %s. Verify reference refits.',
                       len(kept),tr.shape[1],dropped)
    out = tr[:, kept]
    if np.linalg.matrix_rank(out[observed]) != out.shape[1]:
        raise RuntimeError('Rank repair failed for observed trend')
    return out


def make_lag_basis(spec: Specification) -> np.ndarray:
    lag = np.arange(spec.max_lag + 1, dtype=float)
    if spec.name == 'primary_unrestricted':
        # Match original empirical notebook: QR Legendre basis spanning 7 lags.
        from numpy.polynomial.legendre import legvander
        return np.linalg.qr(legvander(np.linspace(-1,1,len(lag)),len(lag)-1))[0]
    if spec.lag_kind == 'identity':
        return np.eye(len(lag))
    if spec.lag_kind == 'cubic':
        expr = (f'bs(lag, df={spec.lag_df}, degree=3, '
                'include_intercept=True, lower_bound=0, '
                f'upper_bound={spec.max_lag}) - 1')
        return np.asarray(dmatrix(expr, {'lag':lag}))
    if spec.lag_kind == 'cubic_knots':
        if not all(0 < k < spec.max_lag for k in spec.knots):
            raise ValueError('Lag knot outside support')
        expr = (f'bs(lag, knots={tuple(spec.knots)!r}, degree=3, '
                'include_intercept=True, lower_bound=0, '
                f'upper_bound={spec.max_lag}) - 1')
        mat = np.asarray(dmatrix(expr, {'lag':lag}))
        if mat.shape[1] != spec.lag_df:
            raise ValueError('Lag knots and lag_df do not match')
        return mat
    raise ValueError(spec.lag_kind)


class ExposureBasis:
    def __init__(self, x: np.ndarray, spec: Specification):
        self.kind = spec.exp_kind
        self.design_info = None
        if self.kind == 'linear':
            self.ncol = 1
        elif self.kind == 'natural_cubic':
            finite = x[np.isfinite(x)]
            if len(finite) < 30:
                raise ValueError('Too few finite exposures for natural spline')
            expr = f'cr(z, df={spec.exp_df}, constraints="center") - 1'
            try:
                mat = dmatrix(expr, {'z':finite})
            except Exception as exc:
                raise ValueError(f'Could not build nonlinear exposure basis: {exc}')
            self.design_info = mat.design_info
            self.ncol = np.asarray(mat).shape[1]
        else:
            raise ValueError(self.kind)

    def evaluate(self, x):
        arr = np.asarray(x, dtype=float)
        shape = arr.shape
        flat = arr.reshape(-1)
        if not np.isfinite(flat).all():
            raise ValueError('Exposure basis evaluated at missing values')
        if self.kind == 'linear':
            val = flat[:,None]
        else:
            val = np.asarray(build_design_matrices([self.design_info], {'z':flat})[0])
        return val.reshape(*shape, self.ncol)

    def contrast(self, x1: float, x0: float = 0.0):
        return (self.evaluate([x1])[0]-self.evaluate([x0])[0])


def make_crossbasis(x: np.ndarray, spec: Specification):
    """Z[t,a,b] = sum_l {g_a(x[t-l])-g_a(0)} H[l,b].

    Centering exposure basis at x0 does not change the effect contrasts;
    importantly, it removes a constant component that can be collinear with
    calendar trend. Keep missing histories as placeholder zeros; their outcomes
    are masked out of the likelihood by the paired common mask.
    """
    lag_mat = make_lag_basis(spec)
    e_basis = ExposureBasis(x, spec)
    n = len(x)
    x_lags = np.column_stack([pd.Series(x).shift(l).to_numpy()
                               for l in range(spec.max_lag+1)])
    valid = np.isfinite(x_lags).all(axis=1)
    flat = x_lags[valid].reshape(-1)
    centered = e_basis.evaluate(flat)-e_basis.evaluate([0.0])[0]
    centered = centered.reshape(valid.sum(), spec.max_lag+1, -1)
    z = np.zeros((n, e_basis.ncol * lag_mat.shape[1]), dtype=float)
    # einsum -> rows x exposure-basis x lag-basis; flatten C-order
    z[valid] = np.einsum('nla,lb->nab', centered, lag_mat).reshape(
        valid.sum(), -1)
    if np.linalg.matrix_rank(z[valid]) < z.shape[1]:
        LOGGER.warning('Cross-basis has rank deficiency (%d of %d columns)',
                       np.linalg.matrix_rank(z[valid]),z.shape[1])
        # Do not silently discard columns. SARIMAX convergence log captures it.
    return z, lag_mat, e_basis, valid


def penalty_block(spec: Specification, n_trend: int, n_exp: int,
                  n_lag: int, total_params: int):
    # P penalizes second differences *across lag-basis coefficients*, for every
    # exposure dimension. Uses the *fixed* coefficient order from einsum.
    P = np.zeros((total_params, total_params))
    D = np.diff(np.eye(n_lag), n=2, axis=0)
    block = np.kron(np.eye(n_exp), D.T@D)
    start = n_trend
    P[start:start+n_exp*n_lag, start:start+n_exp*n_lag] = block
    return P


def safe_cov(cov: np.ndarray):
    cov = np.asarray(cov, float)
    if not np.isfinite(cov).all():
        raise ValueError('Nonfinite covariance')
    cov = (cov+cov.T)/2
    ev = np.linalg.eigvalsh(cov)
    if ev[0] < -1e-6*max(ev[-1],1):
        raise ValueError(f'Materially indefinite covariance: min {ev[0]}')
    if ev[0] < 0:
        v, Q = np.linalg.eigh(cov)
        cov = (Q*np.maximum(v,0))@Q.T
    return cov, float(ev[0])


def orthogonalize_observed_exog(X: np.ndarray, mask: np.ndarray):
    """Invertibly reparameterize the COMPLETE regression design using observed rows.

    X_stable = X @ T; thus X_stable @ b_stable = X @ (T @ b_stable).
    The transformation includes the trend and cross-basis together, preserves
    their span, fitted values and lag-response estimand exactly in algebra.
    We use the OBSERVED weeks to choose T; no missing weeks are removed or
    compressed, and regression parameters/covariances are mapped back to the
    original cross-basis before computing curves and confidence intervals.

    This is a numerical conditioning step, not smoothing or penalization.
    """
    X=np.asarray(X,float)
    mask=np.asarray(mask,dtype=bool)
    obs=X[mask]
    if not np.isfinite(obs).all():
        raise ValueError('Non-finite observed regression design')
    if obs.shape[0] <= obs.shape[1] or np.linalg.matrix_rank(obs)!=obs.shape[1]:
        raise ValueError('Observed model matrix is not identifiable; inspect design')
    cond=float(np.linalg.cond(obs))
    _,R=np.linalg.qr(obs,mode='reduced')
    T=np.linalg.solve(R,np.eye(R.shape[0]))*np.sqrt(obs.shape[0])
    stable=X @ T
    if not np.isfinite(stable).all():
        raise ValueError('Numerical QR transform produced nonfinite design')
    if not np.allclose(stable[mask].T@stable[mask],
                       np.eye(R.shape[0])*obs.shape[0],rtol=1e-7,atol=1e-6):
        raise RuntimeError('QR transformation did not whiten observed design')
    return stable,T,cond


def fit_archived_reference(y, X):
    """Use the original Paper 2 empirical optimizer policy for the 7-DF reference.

    Purposefully separate from generic DLNM numerical optimization. See the
    directly runnable Paper2_direct_Colab_empirical_checks.ipynb reference
    fitter, and the pinned original model snapshot.
    """
    model=SARIMAX(y, exog=X, order=(2,0,1), trend='n',
                  enforce_stationarity=True,enforce_invertibility=True)
    messages=[]; successes=[]
    def attempt(method,limit,start=None):
        kw=dict(method=method,maxiter=limit,disp=False,cov_type='robust')
        if start is not None: kw['start_params']=np.asarray(start.params,float)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                r=model.fit(**kw)
            messages.append(dict(method=method,converged=bool(r.mle_retvals.get('converged',False)),llf=float(r.llf)))
            valid=bool(np.isfinite(r.llf) and np.isfinite(np.asarray(r.params)).all())
            if valid and messages[-1]['converged']:
                v=np.asarray(r.cov_params(),float)
                if np.isfinite(v).all() and np.linalg.eigvalsh((v+v.T)/2).min()>-1e-5:
                    successes.append(r)
            return r if valid else None
        except Exception as e:
            messages.append(dict(method=method,reason=str(e)[:160]));return None
    first=attempt('lbfgs',1500)
    if not successes:
        second=attempt('powell',2500,first)
        if second is not None: attempt('lbfgs',2500,second)
    if not successes: raise RuntimeError('Reference optimizer failed: '+str(messages))
    best=max(successes,key=lambda r:r.llf)
    return model,best,'archived_reference_optimization='+str(messages)


def fit_sarimax_baseline(y, X, maxiter=130, cov_type='robust'):
    """Retain the original v1.1.1 quick-mode fitting path for easy models.

    The 59 successful fits from the previous run should not be perturbed by
    applying a new numerical parameterization everywhere. We therefore use
    the original retry budgets and native regression coordinates first.
    Only failed cases enter the observed-design QR/warm-start rescue.
    """
    model = SARIMAX(y, exog=X, order=(2,0,1), trend='n',
                    enforce_stationarity=True,enforce_invertibility=True,
                    simple_differencing=False)
    best=None; messages=[]
    for method in ('lbfgs','powell','lbfgs'):
        try:
            with warnings.catch_warnings(record=True) as msgs:
                warnings.simplefilter('always')
                opts={'maxiter':int(maxiter),'disp':False}
                if method=='powell':opts['maxiter']=min(int(maxiter),35)
                candidate=model.fit(method=method,cov_type=cov_type,**opts)
            messages += [str(m.message)[:180] for m in msgs]
            good=(bool(candidate.mle_retvals.get('converged',False)) and
                  np.isfinite(candidate.llf) and
                  np.isfinite(np.asarray(candidate.params,float)).all())
            if good:
                cov=np.asarray(candidate.cov_params(),float)
                if not np.isfinite(cov).all():good=False
            if good:
                best=candidate
                break
            messages.append(method+' did not converge')
        except Exception as exc:
            messages.append(f'{method}: {type(exc).__name__}: {exc}')
    if best is None:
        raise RuntimeError('; '.join(messages[-5:]) or 'No converged baseline fit')
    return model,best,'; '.join(messages[-5:])


def fit_sarimax(y, X, maxiter=130, cov_type='robust'):
    """ARMA(2,1) optimization with finite-parameter warm starts and real retries.

    Original Paper 2 audit notebooks allow 1500+ L-BFGS iterations and
    warm-start Powell and final L-BFGS from the previous estimate. The old
    DLNM implementation used 130/35 iterations and independently reinitialized
    all three attempts. Here the first attempt gets >=700 iterations, and
    harder cases receive a warm-start Powell rescue followed by >=1000
    L-BFGS iterations. Failed or non-finite fits are never marked successful.
    """
    model = SARIMAX(y, exog=X, order=(2,0,1), trend='n',
                    enforce_stationarity=True, enforce_invertibility=True,
                    simple_differencing=False)
    history=[]; best=None; warnings_all=[]; starter=None
    attempts=[('lbfgs',max(700,int(maxiter))),
              ('powell',max(120,min(250,int(maxiter)))),
              ('lbfgs',max(1100,int(maxiter)))]
    for method,budget in attempts:
        try:
            kw=dict(method=method,maxiter=budget,disp=False,cov_type=cov_type)
            if starter is not None:
                kw['start_params']=starter.copy()
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter('always')
                result=model.fit(**kw)
            warnings_all += [str(w.message)[:160] for w in recorded]
            finite=bool(np.isfinite(result.llf) and
                        np.isfinite(np.asarray(result.params,float)).all())
            if finite:
                starter=np.asarray(result.params,float).copy()
            ok=bool(result.mle_retvals.get('converged',False)) and finite
            history.append(dict(method=method,budget=budget,
                                iterations=result.mle_retvals.get('iterations'),
                                llf=round(float(result.llf),7) if finite else None,
                                converged=ok))
            if ok:
                # Ensure covariance is suitable for lag-response intervals.
                V=np.asarray(result.cov_params(),float)
                if not np.isfinite(V).all():
                    history[-1]['converged']=False
                    history[-1]['reason']='Nonfinite covariance'
                    continue
                if best is None or result.llf>best.llf:
                    best=result
                break
        except Exception as exc:
            history.append(dict(method=method,budget=budget,converged=False,
                                reason=f'{type(exc).__name__}: {str(exc)[:300]}'))
    if best is None:
        raise RuntimeError('No converged fit after QR and warm-start retries: '+
                           json.dumps(history))
    detail='; '.join(dict.fromkeys(warnings_all[-4:]))
    detail+='; optimizer_history='+json.dumps(history)
    return model,best,detail


def penalized_refit(model, initial, spec: Specification, P: np.ndarray,
                    maxiter: int):
    """Fit fixed-lambda penalized Gaussian ARMA likelihood.

    The penalty is defined in coefficient space. Unpenalized likelihood at the
    fitted penalized parameters is used for fit comparison, but its usual AIC
    cannot be interpreted with k instead of effective degrees of freedom.
    A rough edf-adjusted AIC is returned separately; all penalized inferences
    are conditional on the preselected lambda and rely on Hessian approximation.
    """
    u0 = np.asarray(model.untransform_params(initial.params),float)
    if not np.isfinite(u0).all():
        raise ValueError('Cannot invert SARIMAX parameter transformation')
    def obj(u):
        p = model.transform_params(u)
        ll = model.loglike(p, transformed=True)
        return -float(ll)+0.5*spec.penalty_lambda*float(p @ P @ p)
    t0 = time.perf_counter()
    opt = minimize(obj, u0, method='L-BFGS-B',
                   options={'maxiter':max(450,4*maxiter), 'ftol':1e-8})
    used_score_rescue=False
    if not opt.success:
        # Gradient-based rescue is only invoked when the original finite-
        # difference optimizer fails. The scientific likelihood and fixed
        # lambda remain unchanged; analytic SARIMAX score/Jacobian avoids
        # the finite-difference function-evaluation limit in hard fits.
        def score_objective(u):
            p = np.asarray(model.transform_params(u),float)
            val = -float(model.loglike(p,transformed=True)) + .5*spec.penalty_lambda*float(p@P@p)
            score=np.asarray(model.score(p,transformed=True),float)
            jac=np.asarray(model.transform_jacobian(u),float)
            grad=jac.T @ (-score+spec.penalty_lambda*(P@p))
            return val,grad
        starter=opt.x if np.isfinite(opt.x).all() else u0
        rescue = minimize(score_objective, starter, jac=True, method='L-BFGS-B',
                 options={'maxiter':3500,'maxfun':70000,'ftol':1e-9,'gtol':1e-5})
        if rescue.success and np.isfinite(rescue.x).all():
            opt=rescue
            used_score_rescue=True
    p = np.asarray(model.transform_params(opt.x),float)
    if not opt.success or not np.isfinite(p).all():
        raise RuntimeError('Penalized optimizer failed even after score-gradient rescue: '+str(opt.message))
    ll = float(model.loglike(p, transformed=True))
    # loglikelihood curvature includes all regression and nuisance parameters;
    # this is computationally expensive and numerically approximate.
    Hdata = -approx_hess(p, lambda v: float(model.loglike(v, transformed=True)))
    Hdata = (Hdata+Hdata.T)/2
    Hpen = Hdata + spec.penalty_lambda*P
    V = np.linalg.pinv(Hpen)@Hdata@np.linalg.pinv(Hpen)
    V, _ = safe_cov(V)
    edf = float(np.trace(np.linalg.pinv(Hpen)@Hdata))
    k = len(p)
    neff = max(int(np.isfinite(model.endog).sum()),2)
    return {'params':p, 'cov':V, 'llf':ll, 'aic':-2*ll+2*edf,
            'bic':-2*ll+np.log(neff)*edf,
            'deviance':-2*ll, 'edf':edf, 'k':k,
            'elapsed_penalty':time.perf_counter()-t0,
            'message':str(opt.message) + ('; score-gradient rescue' if used_score_rescue else ''),
            'covariance_method':'conditional_hessian_sandwich_fixed_lambda',
            'fit_note':'AIC/BIC approximate with effective df; inference conditional on fixed lambda'}


def lin_contrast(g1_minus_g0, h):
    return np.outer(g1_minus_g0,h).ravel()


def predict_from_fit(params, V, ntrend, H, exposure, xs=(0.,1.)):
    g=exposure.contrast(xs[1],xs[0])
    C=np.vstack([lin_contrast(g, H[l]) for l in range(H.shape[0])])
    beta=params[ntrend:ntrend+C.shape[1]]
    point=C @ beta
    if V is None:
        se=np.full(H.shape[0],np.nan)
    else:
        Ve=V[ntrend:ntrend+C.shape[1], ntrend:ntrend+C.shape[1]]
        se=np.sqrt(np.maximum(np.einsum('ik,kl,il->i',C,Ve,C),0))
    sumC=C.sum(axis=0)
    total=float(sumC@beta)
    totse=(float(np.sqrt(max(0,sumC@Ve@sumC))) if V is not None else np.nan)
    return point,se,total,totse

def make_fit_output(country: str, outcome: str, w: pd.DataFrame,
                    spec: Specification, args):
    # Reconstruct the archived empirical model's trend-time support. Its
    # weekly series is cropped from the first to last common observed week
    # *before* generating the 8-df calendar spline. The previous DLNM code
    # fitted the trend over the entire ~240-week study calendar, which changed
    # coefficients even though the 1,882 paired observations were identical.
    #
    # Exposure histories MUST be generated before this crop, because the
    # first retained observed week depends on cases in preceding calendar weeks.
    full_x = w.x.to_numpy(float)
    full_y = w['y_'+outcome].to_numpy(float)
    full_mask = w.common.to_numpy(bool)
    observed_positions = np.flatnonzero(full_mask)
    if not len(observed_positions):
        raise ValueError('Country has no usable observations')
    sl = slice(int(observed_positions[0]), int(observed_positions[-1])+1)
    Z_full,H,exp_basis,valid_full = make_crossbasis(full_x,spec)
    Z = Z_full[sl].copy()
    valid = valid_full[sl]
    x = full_x[sl]
    y = full_y[sl].copy()
    mask = full_mask[sl]
    y[~mask] = np.nan
    n = len(y)
    tr = identifiable_trend_columns(make_trend(n),mask)
    if not np.all(valid[mask]):
        raise RuntimeError('Common mask contains incomplete exposure histories')
    exog=np.column_stack([tr,Z])
    if np.linalg.matrix_rank(exog[mask]) < exog.shape[1]:
        raise ValueError('Rank-deficient observed model matrix')
    # The calendar-support repair alone did not resolve optimization
    # failure: poorly conditioned exposure-lag and trend columns made the
    # 130-iteration SARIMAX optimizer stop prematurely. QR is an invertible
    # reparameterization of the *whole* observed design (no scientific change).
    # Keep the original numerical fit whenever it converges, reproducing
    # the successful v1.1.1 pathway. For failures ONLY, apply the invertible
    # observed-design QR and longer, warm-started optimizer sequence.
    design_condition=float(np.linalg.cond(exog[mask]))
    t0=time.perf_counter()
    qr_used=False
    T=np.eye(exog.shape[1])
    try:
        if spec.name == 'primary_unrestricted':
            model,fit,warn=fit_archived_reference(y,exog)
        else:
            model,fit,warn=fit_sarimax_baseline(y,exog,args.maxiter)
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as baseline_exc:
        LOGGER.warning('%s/%s/%s: baseline optimization failed (%s); '
                       'retrying with observed-design QR and warm starts',
                       country,outcome,spec.name,str(baseline_exc)[:140])
        stable_exog,T,_=orthogonalize_observed_exog(exog,mask)
        model,fit,warn=fit_sarimax(y,stable_exog,args.maxiter)
        qr_used=True
        warn='QR rescue after baseline failure; '+warn
    duration=time.perf_counter()-t0
    fitted_params=np.asarray(fit.params,float)
    fitted_cov=np.asarray(fit.cov_params(),float)
    covariance_method='robust_sarimax'
    edf=float(len(fitted_params))
    llf,aic,bic = float(fit.llf),float(fit.aic),float(fit.bic)
    deviance=-2*llf
    fit_note='Unpenalized Gaussian ARMA likelihood; observed-design QR; robust covariance'
    # Original coordinates = G @ fitted coordinates. Only regression
    # coefficients transform, ARMA parameters and innovation variance do not.
    G=np.eye(len(fitted_params))
    G[:exog.shape[1],:exog.shape[1]]=T
    if spec.penalty_lambda > 0:
        # Original lag-penalty is defined on the untransformed cross-basis;
        # map it to QR fit coordinates before numerical penalized refitting.
        P=penalty_block(spec,tr.shape[1],exp_basis.ncol,H.shape[1],len(fitted_params))
        P_qr=G.T @ P @ G
        penal=penalized_refit(model,fit,spec,P_qr,args.maxiter)
        fitted_params,fitted_cov=penal['params'],penal['cov']
        duration+=penal['elapsed_penalty']
        llf,aic,bic,deviance=(penal['llf'],penal['aic'],penal['bic'],penal['deviance'])
        edf,covariance_method,fit_note=(penal['edf'],penal['covariance_method'],penal['fit_note'])
        warn+='; '+penal['message']
    params=G @ fitted_params
    V,mineig=safe_cov(G @ fitted_cov @ G.T)
    warn=f'Observed design condition={design_condition:.4g}; '+warn
    ntr=tr.shape[1]
    point,se,cum,cumse=predict_from_fit(params,V,ntr,H,exp_basis)
    rows=[]
    for l in range(len(point)):
        rows.append({'country':country,'outcome':outcome,'specification':spec.name,
                     'lag':l, 'effect':float(point[l]), 'se':float(se[l]),
                     'ci_lower':float(point[l]-1.96*se[l]),
                     'ci_upper':float(point[l]+1.96*se[l]),
                     'exposure_low':0.,'exposure_high':1.,
                     'units':'standardized_asinh_outcome_difference'})
    gdata=[]
    surface=[]
    # Exposure-specific *cumulative* effects of changing all case lags from
    # 0 to v (i.e. sustained exposure-history contrast), plus a full surface.
    xfinite=x[np.isfinite(x)]
    low=float(np.quantile(xfinite,.05))
    hi=float(np.quantile(xfinite,.95))
    grid=np.linspace(min(low,0),max(hi,1),31)
    for value in grid:
        surf,serr,total,totse=predict_from_fit(params,V,ntr,H,exp_basis,(0.,float(value)))
        gdata.append(dict(country=country,outcome=outcome,specification=spec.name,
                          x_contrast=float(value),cumulative_effect=float(total),
                          cumulative_lower=float(total-1.96*totse),
                          cumulative_upper=float(total+1.96*totse)))
        for ell in range(len(surf)):
            surface.append(dict(country=country, outcome=outcome, specification=spec.name,
                  x_contrast=float(value), lag=int(ell), effect=float(surf[ell]),
                  se=float(serr[ell]), lower=float(surf[ell]-1.96*serr[ell]),
                  upper=float(surf[ell]+1.96*serr[ell])))
    stat=dict(country=country,outcome=outcome,specification=spec.name,
              max_lag=spec.max_lag,exposure_form=spec.exp_kind,
              lag_basis=spec.lag_kind, lag_df=spec.lag_df,
              lag_knots=str(spec.knots),penalty_lambda=spec.penalty_lambda,
              effective_df=edf,n_common=int(mask.sum()),n_calendar=int(n),
              n_coefficients=len(params),log_likelihood=llf,
              aic=aic,bic=bic,minus2loglik=deviance,
              optimizer_converged=True, cov_min_eigenvalue=mineig,
              covariance_method=covariance_method,elapsed_seconds=duration,
              design_condition_before_qr=design_condition,
              used_observed_design_qr=qr_used,
              warnings=warn,fit_note=fit_note,
              peak_lag=int(np.argmax(point)),
              dominant_abs_peak_lag=int(np.argmax(np.abs(point))),
              peak_effect=float(np.max(point)),
              peak_effect_se=float(se[np.argmax(point)]),
              peak_effect_ci_width=float(3.92*se[np.argmax(point)]),
              cumulative_effect=cum,cumulative_se=cumse,
              cumulative_lower=cum-1.96*cumse,
              cumulative_upper=cum+1.96*cumse,
              cumulative_ci_width=3.92*cumse,
              cumulative_sign='positive' if cum>0 else ('negative' if cum<0 else 'zero'),
              sign_at_peak='positive' if point[np.argmax(point)]>0 else 'nonpositive',
              n_secondary_peaks=0)
    peaks, prop=find_peaks(np.abs(point), prominence=max(.05,.25*float(np.max(np.abs(point)))))
    peaks=[int(p) for p in peaks if p != int(np.argmax(np.abs(point)))]
    stat['n_secondary_peaks']=len(peaks)
    stat['secondary_peak_lags']=','.join(map(str,peaks))
    return {'stat':stat,'lagrows':rows,'gridrows':gdata,'surface_rows':surface}


def data_signature(country: str, outcome: str, w: pd.DataFrame, spec: Specification):
    bits = (w[['x','y_'+outcome,'common']].to_csv(index=True).encode()+
            json.dumps(asdict(spec),sort_keys=True).encode()+
            SCHEMA_VERSION.encode())
    return hashlib.sha256(bits).hexdigest()


def safe_name(s):
    return re.sub(r'[^a-zA-Z0-9_-]','_',s)


def cached_fit(country,outcome,w,spec,args,root):
    cache=root/'checkpoints'/f'{safe_name(country)}__{outcome}__{spec.name}.json'
    sig=data_signature(country,outcome,w,spec)
    if cache.exists() and not args.force:
        try:
            obj=json.loads(cache.read_text(encoding='utf-8'))
            if obj.get('signature')==sig and 'fit' in obj and 'surface_rows' in obj['fit']:
                LOGGER.info('Reusing valid checkpoint %s/%s/%s',country,outcome,spec.name)
                return obj['fit']
        except Exception:
            pass
    start=time.perf_counter()
    try:
        fitted=make_fit_output(country,outcome,w,spec,args)
        msg=f"{country}/{outcome}/{spec.name}: success, AIC={fitted['stat']['aic']:.2f}, "
        LOGGER.info('%s%.1f seconds',msg,time.perf_counter()-start)
    except Exception as exc:
        LOGGER.error('%s/%s/%s: %s',country,outcome,spec.name,exc)
        fitted={'stat':dict(country=country,outcome=outcome,
                            specification=spec.name,optimizer_converged=False,
                            n_common=int(w.common.sum()),
                            error_type=type(exc).__name__,error_message=str(exc),
                            elapsed_seconds=time.perf_counter()-start),
                'lagrows':[],'gridrows':[],'surface_rows':[]}
        LOGGER.debug(traceback.format_exc())
    temp=cache.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(signature=sig,fit=fitted),allow_nan=False,
                               indent=1),encoding='utf-8')
    temp.replace(cache)
    return fitted


def profile_vectors(lag_df,pair,spec):
    d=lag_df.loc[(lag_df.country==pair[0]) & (lag_df.outcome==pair[1]) &
                 (lag_df.specification==spec)].sort_values('lag')
    return d.effect.to_numpy(float),d.se.to_numpy(float),d


def compare_models(stats,lag_df,args):
    rows=[]
    pairs=stats[['country','outcome']].drop_duplicates().itertuples(index=False,name=None)
    baseline='primary_unrestricted'
    for country,outcome in pairs:
        primary=stats[(stats.country==country)&(stats.outcome==outcome)&
                      (stats.specification==baseline)&(stats.optimizer_converged==True)]
        if primary.empty:
            continue
        rp=primary.iloc[0]
        p,pse,_=profile_vectors(lag_df,(country,outcome),baseline)
        for _,r in stats.loc[(stats.country==country)&(stats.outcome==outcome)&
                              (stats.specification!=baseline)&
                              (stats.optimizer_converged==True)].iterrows():
            q,qse,_=profile_vectors(lag_df,(country,outcome),r.specification)
            h=min(len(p),len(q))
            if h==0:continue
            p0,q0=p[:h],q[:h]
            sim=float(np.corrcoef(p0,q0)[0,1]) if np.std(p0)>1e-12 and np.std(q0)>1e-12 else np.nan
            euclidean=float(np.linalg.norm(p0-q0))
            rmse=float(np.sqrt(np.mean((p0-q0)**2)))
            denom=np.linalg.norm(p0)*np.linalg.norm(q0)
            cosine=float(p0@q0/denom) if denom>1e-12 else np.nan
            same_lag=(r.max_lag==rp.max_lag)
            drift=int(r.peak_lag)-int(rp.peak_lag)
            cdiff=float(r.cumulative_effect-rp.cumulative_effect)
            # Heuristic materiality flags, *not* statistical significance of
            # correlated-model contrasts. Reporting threshold configurable.
            flagged=(abs(drift)>args.peak_tolerance or
                     (same_lag and abs(cdiff)>args.cumulative_tolerance))
            row=dict(country=country,outcome=outcome,
                     specification=r.specification,
                     primary_peak_lag=int(rp.peak_lag),alternative_peak_lag=int(r.peak_lag),
                     peak_lag_difference=drift,
                     peak_agreement_within_tolerance=abs(drift)<=args.peak_tolerance,
                     primary_peak_effect=float(rp.peak_effect),
                     alternative_peak_effect=float(r.peak_effect),
                     primary_cumulative=float(rp.cumulative_effect),
                     alternative_cumulative=float(r.cumulative_effect),
                     cumulative_difference=cdiff,
                     cumulative_comparable_same_horizon=bool(same_lag),
                     cumulative_same_sign=bool(np.sign(r.cumulative_effect)==np.sign(rp.cumulative_effect)),
                     lag_profile_correlation=sim, cosine_similarity=cosine,
                     lag_profile_rmse=rmse,euclidean_distance=euclidean,
                     primary_peak_ci_width=float(rp.peak_effect_ci_width),
                     alternative_peak_ci_width=float(r.peak_effect_ci_width),
                     primary_cumulative_ci_width=float(rp.cumulative_ci_width),
                     alternative_cumulative_ci_width=float(r.cumulative_ci_width),
                     primary_secondary_peak_lags=rp.secondary_peak_lags,
                     alternative_secondary_peak_lags=r.secondary_peak_lags,
                     delta_aic=float(r.aic-rp.aic) if r.penalty_lambda==0 else np.nan,
                     delta_bic=float(r.bic-rp.bic) if r.penalty_lambda==0 else np.nan,
                     different_scientific_interpretation_flag=bool(flagged),
                     flag_definition='descriptive thresholds, not hypothesis tests')
            rows.append(row)
    return pd.DataFrame(rows)


def load_archived_primary(path: str, lag: pd.DataFrame, root: Path):
    if not path:
        return {'audit':'not_performed',
                'reason':'An original archived primary coefficient CSV was not provided; '
                'primary model is a reconstruction, not independently confirmed archival refit.'}
    archive=pd.read_csv(path)
    mandatory={'country','outcome','lag','effect'}
    if not mandatory.issubset(archive.columns):
        raise ValueError('Archival CSV must have country,outcome,lag,effect columns')
    recent=lag[lag.specification.eq('primary_unrestricted')]
    matched=recent.merge(archive,on=['country','outcome','lag'],
                         suffixes=('_refit','_archived'))
    if matched.empty:
        raise ValueError('Zero matching archived primary coefficient rows')
    matched['abs_difference']=(matched.effect_refit-matched.effect_archived).abs()
    matched.to_csv(root/'tables'/'02_archived_primary_coefficient_audit.csv',index=False)
    out={'audit':'compared','matched_rows':len(matched),
         'max_abs_effect_difference':float(matched.abs_difference.max()),
         'missing_refit_rows':int(len(recent)-len(matched))}
    if len(matched)!=len(recent):
        LOGGER.warning('Archival audit incomplete: %s/%s rows matched',len(matched),len(recent))
    return out


def df_csv(df,root,name):
    path=root/'tables'/name
    df.to_csv(path,index=False)
    return path

def set_plot_style():
    plt.rcParams.update({'figure.dpi':120, 'savefig.dpi':360,
        'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':False, 'legend.frameon':False})

def save_fig(fig,root,number,caption):
    fig.tight_layout(pad=1.4)
    for ext in ('png','pdf'):
        fig.savefig(root/'figures'/f'{number}.{ext}',dpi=360,
                    bbox_inches='tight',facecolor='white')
    plt.close(fig)
    with (root/'figures'/'figure_captions.txt').open('a',encoding='utf-8') as f:
        f.write(f'{number}: {caption}\n\n')


def trace(ax,d,label,color,ci=True):
    if d.empty:return
    d=d.sort_values('lag')
    ax.plot(d.lag,d.effect,marker='o',ms=3,color=color,label=label,lw=1.5)
    if ci and d.ci_lower.notna().any():
        ax.fill_between(d.lag.to_numpy(float),d.ci_lower.to_numpy(float),
                        d.ci_upper.to_numpy(float),alpha=.17,color=color)


def figures(stats,lag,grid,root,featured='United States'):
    set_plot_style()
    good=stats.loc[stats.optimizer_converged.eq(True)]
    if good.empty or lag.empty:return
    pairs={(r.country,r.outcome) for r in good.itertuples()}
    if (featured,'admissions') not in pairs:
        featured=next(iter(sorted(pairs)))[0]
    outcome='admissions' if (featured,'admissions') in pairs else 'occupancy'
    pick=lag[(lag.country==featured)&(lag.outcome==outcome)]
    a=pick[pick.specification.eq('primary_unrestricted')]
    b=pick[pick.specification.eq('dlnm_linear_cubic_df4')]
    # Fig 1 point profiles, without CIs, to separate shape from inference.
    fig,ax=plt.subplots(figsize=(7.2,4.1))
    trace(ax,a,'Primary unrestricted',BLUE,False)
    trace(ax,b,'Linear-exposure DLNM',ORANGE,False)
    ax.axhline(0,color='gray',lw=.7)
    ax.set(xlabel='Case lag (weeks)',ylabel='Conditional standardized effect',
           title=f'Lag representation | {featured}: {outcome}')
    ax.legend()
    save_fig(fig,root,'Figure_1_primary_vs_DLNM',
             'Unrestricted seven-position lag curve and linear-exposure DLNM, matched data and error model. No intervals in this representation-only figure.')
    # Fig2 interval overlay
    fig,ax=plt.subplots(figsize=(7.2,4.1))
    trace(ax,a,'Primary 95% pointwise interval',BLUE,True)
    trace(ax,b,'DLNM 95% pointwise interval',ORANGE,True)
    ax.axhline(0,color='gray',lw=.7)
    ax.set(xlabel='Case lag (weeks)',ylabel='Conditional standardized effect ± pointwise 95% CI',
           title=f'Lag profiles with estimated uncertainty | {featured}: {outcome}')
    ax.legend(fontsize=8)
    save_fig(fig,root,'Figure_2_overlay_pointwise_CI',
             'Gaussian ARMA-model estimates; shaded intervals are nominal pointwise 95% Wald intervals, not calibrated or simultaneous bands.')
    # Fig3 cumulative contrasts across countries, two outcomes separately
    sub=good[good.specification.isin(['primary_unrestricted','dlnm_linear_cubic_df4'])]
    fig,axs=plt.subplots(1,2,figsize=(13,5.7),sharex=False)
    for ax,o in zip(axs,OUTCOMES):
        dd=sub[sub.outcome.eq(o)]
        names=[c for c in COUNTRIES if c in set(dd.country)]
        y=np.arange(len(names))
        for off,name,color in [(-.15,'primary_unrestricted',BLUE),(.15,'dlnm_linear_cubic_df4',ORANGE)]:
            dat=dd[dd.specification.eq(name)].set_index('country')
            v=np.array([dat.loc[c,'cumulative_effect'] if c in dat.index else np.nan for c in names])
            e=np.array([1.96*dat.loc[c,'cumulative_se'] if c in dat.index else np.nan for c in names])
            ax.errorbar(v,y+off,xerr=e,fmt='o',ms=3.5,lw=1,capsize=2,
                        color=color,label='Primary' if name=='primary_unrestricted' else 'DLNM')
        ax.set_yticks(y,labels=names)
        ax.axvline(0,color='gray',lw=.7)
        ax.invert_yaxis()
        ax.set(xlabel='Cumulative standardized effect (0 → 1 exposure contrast)',title=o.title())
        ax.legend(fontsize=8)
    save_fig(fig,root,'Figure_3_cumulative_effects',
             'Sustained 0-to-1 standardized asinh-case contrast, summed across lags 0–6, with marginal pointwise 95% intervals. Not original-scale patient counts.')
    # Fig4 sensitivity to lag horizon (only in full mode).
    sub=good[good.specification.isin(['dlnm_linear_lag4','dlnm_linear_cubic_df4','dlnm_linear_lag8'])]
    if sub.specification.nunique()>=2:
        fig,axs=plt.subplots(1,2,figsize=(13,5.0),sharey=True)
        for ax,o in zip(axs,OUTCOMES):
            d=sub[sub.outcome.eq(o)]
            for j,country in enumerate(COUNTRIES):
                v=d[d.country.eq(country)].sort_values('max_lag')
                if len(v)>=2:
                    ax.plot(v.max_lag,v.peak_lag,color=COLORS[j%len(COLORS)],
                            marker='o',ms=3,alpha=.8,label=country)
            ax.set(xticks=[4,6,8],xlabel='Maximum lag (weeks)',
                   title=o.title(),ylabel='Peak lag (weeks)')
        handles,labels=axs[1].get_legend_handles_labels()
        if handles:fig.legend(handles,labels,loc='lower center',ncol=6,fontsize=7,
                              bbox_to_anchor=(.5,-.04))
        save_fig(fig,root,'Figure_4_lag_horizon_sensitivity',
                 'Peak lag of linear-exposure cubic DLNM under maximum lags of 4, 6 and 8 weeks; common eight-week sample mask.')
    # Fig5 knots (only in full mode).
    knotnames=['dlnm_linear_knots_1_3','dlnm_linear_knots_2_4','dlnm_linear_knots_3_5']
    if lag.specification.isin(knotnames).any():
        fig,axs=plt.subplots(1,2,figsize=(12,4.5))
        for ax,o in zip(axs,OUTCOMES):
            d=lag[(lag.country==featured)&(lag.outcome.eq(o))]
            for i,nm in enumerate(knotnames):
                trace(ax,d[d.specification.eq(nm)],nm.rsplit('_',2)[-2]+' / '+nm.rsplit('_',2)[-1],COLORS[i],True)
            ax.set(xlabel='Case lag (weeks)',ylabel='Standardized effect',title=f'{featured}: {o}')
            ax.axhline(0,color='gray',lw=.7)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(title='Interior knots',fontsize=8)
        save_fig(fig,root,'Figure_5_lag_knot_sensitivity',
                 'Six-column cubic B-spline lag cross-bases with alternative interior-knot locations; fixed horizon and observation mask.')
    # Fig6 country panel, two outcome panels per country, paginated
    for o in OUTCOMES:
        fig,axs=plt.subplots(4,3,figsize=(12.5,12),sharex=True)
        for ax,country in zip(axs.flat,COUNTRIES):
            d=lag[(lag.country==country)&(lag.outcome.eq(o))]
            trace(ax,d[d.specification.eq('primary_unrestricted')],'Primary',BLUE,False)
            trace(ax,d[d.specification.eq('dlnm_linear_cubic_df4')],'DLNM',ORANGE,False)
            ax.axhline(0,color='gray',lw=.6)
            ax.set_title(country)
            ax.set_xlabel('Lag (weeks)')
        for ax in list(axs.flat)[len(COUNTRIES):]:ax.axis('off')
        fig.legend(['Primary','Linear DLNM'],loc='lower center',ncol=2,fontsize=9)
        save_fig(fig,root,f'Figure_6_country_level_{o}',
                 'Country panels for '+o+'; each line represents a separate national conditional association; do not interpret as causal or cross-border dependence.')
    # Fig7 outcome-level panels: country x lag *difference* heat map
    fig,axs=plt.subplots(1,2,figsize=(13,7.2),sharey=True)
    for ax,o in zip(axs,OUTCOMES):
        diff=np.full((len(COUNTRIES),7),np.nan)
        for j,country in enumerate(COUNTRIES):
            pr=lag[(lag.country==country)&(lag.outcome.eq(o))&
                   (lag.specification.eq('primary_unrestricted'))].sort_values('lag')
            dl=lag[(lag.country==country)&(lag.outcome.eq(o))&
                   (lag.specification.eq('dlnm_linear_cubic_df4'))].sort_values('lag')
            if len(pr)==len(dl)==7:diff[j,:]=dl.effect.to_numpy()-pr.effect.to_numpy()
        vmax=np.nanmax(np.abs(diff)) if np.isfinite(diff).any() else 1
        im=ax.imshow(diff,cmap='PuOr',vmin=-vmax,vmax=vmax,aspect='auto')
        ax.set_yticks(np.arange(len(COUNTRIES)),labels=COUNTRIES)
        ax.set_xticks(range(7))
        ax.set(xlabel='Lag (weeks)',title=o.title()+' (DLNM − primary)')
        fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label='Standardized effect difference')
    save_fig(fig,root,'Figure_7_outcome_level_comparison',
             'Difference in conditional lag effects by country, shown separately for admissions and occupancy. No pooling or spatial-dependence estimate.')
    # Additional supplemental nonlinear exposure-response curve (if fitted)
    fig,axs=plt.subplots(1,2,figsize=(12,4.3))
    for ax,o in zip(axs,OUTCOMES):
        g=grid[(grid.country==featured)&(grid.outcome.eq(o))]
        for name,col in [('dlnm_linear_cubic_df4',BLUE),
                         ('dlnm_nonlinear_natural_df4',ORANGE)]:
            d=g[g.specification.eq(name)].sort_values('x_contrast')
            if len(d):
                ax.plot(d.x_contrast,d.cumulative_effect,label=name,color=col)
                ax.fill_between(d.x_contrast.to_numpy(),d.cumulative_lower.to_numpy(),
                                d.cumulative_upper.to_numpy(),color=col,alpha=.14)
        ax.axhline(0,color='gray',lw=.7)
        ax.set(xlabel='Standardized asinh cases vs reference 0',
               ylabel='Cumulative standardized outcome contrast',title=f'{featured}: {o}')
        ax.legend(fontsize=7)
    save_fig(fig,root,'Supplementary_nonlinear_exposure_response',
             'Cumulative exposure-response contrast compared with reference standardized cases zero, with nominal pointwise CIs.')

def reviewer_tables(stats,comparison,root,args):
    if stats.empty or comparison.empty:return
    keys=['dlnm_linear_cubic_df4','dlnm_nonlinear_natural_df4']
    agree=comparison[comparison.specification.eq('dlnm_linear_cubic_df4')].copy()
    df_csv(agree,root,'06_main_agreement_country_outcome.csv')
    summary=[]
    for spec,g in comparison.groupby('specification'):
        comparable=g[g.cumulative_comparable_same_horizon.eq(True)]
        summary.append(dict(specification=spec,country_outcome_pairs=len(g),
            peak_agreement_pct=100*g.peak_agreement_within_tolerance.mean(),
            same_cumulative_sign_pct=100*comparable.cumulative_same_sign.mean() if len(comparable) else np.nan,
            median_abs_cumulative_difference=float(comparable.cumulative_difference.abs().median()) if len(comparable) else np.nan,
            median_profile_correlation=float(g.lag_profile_correlation.median()),
            median_profile_rmse=float(g.lag_profile_rmse.median()),
            flagged_differences=int(g.different_scientific_interpretation_flag.sum())))
    df_csv(pd.DataFrame(summary),root,'07_reviewer_sensitivity_summary.csv')
    # Nonlinearity: ΔAIC for same horizon and same outcome, not proof of
    # exposure nonlinearity and not a likelihood-ratio test in this script.
    df_csv(comparison[comparison.specification.eq('dlnm_nonlinear_natural_df4')],
           root,'08_nonlinearity_diagnostic.csv')
    df_csv(comparison[comparison.specification.str.contains('knots')],
           root,'09_knot_placement_diagnostic.csv')
    df_csv(comparison[comparison.specification.str.contains('lag4|lag8')],
           root,'10_lag_horizon_diagnostic.csv')
    # Secondary head-to-head comparisons use *the same linear DLNM lag basis*
    # where possible. Comparing nonlinear DLNM only with the unrestricted
    # primary would confound the effects of nonlinearity and lag restriction.
    ok=stats.loc[stats.optimizer_converged.eq(True)]
    lin=ok[ok.specification.eq('dlnm_linear_cubic_df4')][
        ['country','outcome','aic','bic','n_common']].rename(columns={
        'aic':'linear_aic','bic':'linear_bic','n_common':'linear_n'})
    non=ok[ok.specification.eq('dlnm_nonlinear_natural_df4')][
        ['country','outcome','aic','bic','n_common']].rename(columns={
        'aic':'nonlinear_aic','bic':'nonlinear_bic','n_common':'nonlinear_n'})
    compare_nl=lin.merge(non,on=['country','outcome'],how='inner')
    if len(compare_nl):
        compare_nl['delta_aic_nonlinear_minus_linear']=compare_nl.nonlinear_aic-compare_nl.linear_aic
        compare_nl['delta_bic_nonlinear_minus_linear']=compare_nl.nonlinear_bic-compare_nl.linear_bic
        compare_nl['lower_aic_nonlinear_by_at_least_2']=(compare_nl.delta_aic_nonlinear_minus_linear<=-2)
        compare_nl['same_sample']=(compare_nl.linear_n==compare_nl.nonlinear_n)
    df_csv(compare_nl,root,'11_nonlinearity_direct_DLNM_fit_test.csv')
    ka=ok[ok.specification.eq('dlnm_linear_knots_2_4')][
        ['country','outcome','aic','bic']].rename(columns={'aic':'knot24_aic','bic':'knot24_bic'})
    knots=ok[ok.specification.isin(['dlnm_linear_knots_1_3','dlnm_linear_knots_3_5'])][
        ['country','outcome','specification','aic','bic']].merge(ka,on=['country','outcome'])
    if len(knots):
        knots['delta_aic_vs_knots24']=knots.aic-knots.knot24_aic
        knots['delta_bic_vs_knots24']=knots.bic-knots.knot24_bic
    df_csv(knots,root,'12_equal_df_knot_head_to_head.csv')
    pen=ok[ok.specification.eq('dlnm_penalized_lag_lambda1')][
        ['country','outcome','aic','bic','effective_df']].rename(columns={
            'aic':'approx_penalized_aic','bic':'approx_penalized_bic',
            'effective_df':'penalized_effective_df'})
    pen_cmp=ka.merge(pen,on=['country','outcome'],how='inner')
    if len(pen_cmp):
        pen_cmp['approx_aic_difference']=pen_cmp.approx_penalized_aic-pen_cmp.knot24_aic
        pen_cmp['warning']='Penalized IC uses fixed-lambda approximate effective degrees of freedom; exploratory only'
    df_csv(pen_cmp,root,'13_penalization_exploratory_comparison.csv')
    # Percentages always conditional on converged paired fits.


def write_text(root,stats,comp,match,audit,args,archived):
    path=root/'text'
    primary=comp[comp.specification.eq('dlnm_linear_cubic_df4')] if len(comp) else pd.DataFrame()
    n=int(len(primary))
    n_ag=int(primary.peak_agreement_within_tolerance.sum()) if n else 0
    n_sign=int(primary.cumulative_same_sign.sum()) if n else 0
    n_flags=int(primary.different_scientific_interpretation_flag.sum()) if n else 0
    n_fit=int(stats.optimizer_converged.eq(True).sum()) if len(stats) else 0
    n_fail=int(stats.optimizer_converged.eq(False).sum()) if len(stats) else 0
    note=('Matches manuscript inclusion count' if match else
          'DOES NOT MATCH manuscript inclusion count; NOT archival replication')
    txt=[f'SSTE DLNM evidence summary | {datetime.now(timezone.utc).isoformat()}',
         f'Input QC: {note}. Successful specification fits={n_fit}; failures={n_fail}.',
         f'Pinned OWID commit: {OWID_COMMIT}',
         f'Archived-primary audit: {json.dumps(archived)}',
         '', 'EMPIRICAL PRIMARY-VS-DLNM COMPARISON (linear exposure; L=6)',
         f'Matched country-outcome pairs: {n}',
         f'Peak lag within ±{args.peak_tolerance} week(s): {n_ag}/{n}',
         f'Same sign of sustained cumulative contrast: {n_sign}/{n}',
         f'Descriptive materiality flags: {n_flags}/{n}',
         'These are estimates from observed surveillance data, not known-target simulation.',
         'Peak locations are point-estimate features; their uncertainty was not estimated.',
         'Coefficient-correlation and RMSE measure curve similarity; they do not test causal or clinical agreement.',
         'The two fits share cases and outcomes; naïve independent-model difference tests are invalid.',
         'Nominal intervals are not calibrated bands: primary manuscript simulations documented undercoverage.',
         'Changing lag horizon changes the estimand; cumulative contrasts over L=4 and L=8 are not directly comparable to L=6.',
         'Penalized model uncertainty and edf-based IC, if requested, are conditional approximations.', '',
         'REVIEWER QUESTIONS',
         'Q1. Substantively different? See main agreement table and flagged pairs, with explicit descriptive thresholds.',
         'Q2. Do peaks change? See peak lag delta, Figure 4 and Table 06.',
         'Q3. Do cumulative effects change? See Figure 3, differences and CI widths (not cross-model contrast CIs).',
         'Q4. Does nonlinearity improve fit? Compare within-pair AIC/BIC in model statistics; this is in-sample evidence only.',
         'Q5. Robust to DLNM specifications? See the sensitivity table and all failed fits.',
         'Q6. Knot selection? See Figure 5 and Table 09.',
         'Q7. Lag horizon? See Figure 4 and Table 10.', '',
         'COUNTRY-OUTCOME AGREEMENTS/DISAGREEMENTS']
    if n:
        for r in primary.itertuples():
            flag='FLAG' if r.different_scientific_interpretation_flag else 'similar within predefined descriptive tolerance'
            txt.append(f'{r.country} | {r.outcome}: peak {r.primary_peak_lag} vs {r.alternative_peak_lag}; '
                       f'cumulative Δ={r.cumulative_difference:+.3f}; profile r={r.lag_profile_correlation:.3f}; {flag}')
    else:
        txt.append('Not estimable: no pairs had two converged fits.')
    txt += ['', 'SCIENTIFIC CONCLUSION',
        'Do NOT automatically claim findings remain robust. Check flagged pairs, QC status, and failed fits.',
        'This empirical comparison does not validate coverage, forecast skill, patient-level timing or causality.',
        'If estimates differ, report divergence and revisit individual country/outcome interpretation; '
        'the separate known-target simulation evidence on basis trade-offs is not replaced by these fits.']
    (path/'empirical_interpretation.txt').write_text('\n'.join(txt)+'\n',encoding='utf-8')
    support='No blanket robustness conclusion can be inferred from this empirical sensitivity analysis alone.'
    if n and match and archived.get('audit')=='compared' and archived.get('missing_refit_rows',1)==0:
        support='Empirical shape agreement can be described for the matched pairs below, subject to the residual and interval limitations.'
    method=("Methods - Additional DLNM comparison. We fitted a secondary, matched Gaussian "
      "distributed-lag nonlinear cross-basis analysis to the same weekly standardized asinh-transformed case, "
      "admissions and occupancy series as the main model. We retained the common complete eight-week exposure "
      "history mask, eight-degree-of-freedom cubic calendar spline design and ARMA(2,1) error model. Calendar trend columns not identifiable on the paired sample were removed identically across matched specifications. The reference unrestricted "
      "seven-position lag vector was compared with a cubic-spline lag cross-basis under a linear exposure effect. "
      "Exposure nonlinearities were assessed with centered natural cubic regression splines evaluated as "
      "standardized exposure contrasts from 0 to 1. Alternative lag supports and interior knots were sensitivity "
      "analyses. Wald intervals were derived from model-estimated parameter covariance; penalized estimates, "
      "where run, used a fixed second-difference penalty and approximate conditional curvature covariance. "
      "All comparisons are retrospective and noncausal. The implementation uses a Python tensor cross-basis "
      "and does not claim bitwise equivalence with the R dlnm package.")
    if n:
        results=(f"Results - In {n} matched country-outcome pairs with converged primary and "
            f"linear-exposure DLNM fits, peak lag estimates agreed within ±{args.peak_tolerance} week(s) "
            f"for {n_ag} pairs, and cumulative effect signs agreed for {n_sign}. "
            f"{n_flags} pairs exceeded the prespecified descriptive sensitivity thresholds. "
            "Country-specific comparisons and model-fit failures are reported in the accompanying tables. "
            "These are in-sample conditional association contrasts rather than patient-level delay or forecasting results.")
    else:
        results='Results - Not generated: no matched successful country-outcome comparisons.'
    discussion=("Discussion - DLNM sensitivity provides a check on whether country-specific surveillance "
       "patterns persist under an alternative lag representation and exposure-response specification. "
       "Apparent agreement should not be read as confirmation of calibrated uncertainty, since nominal "
       "intervals were shown to undercover in the original known-target simulations. Where profiles differ, "
       "lag peaks should not be reported as stable surveillance characteristics without resolving basis, "
       "measurement and temporal-support sensitivities. This analysis does not directly quantify spatial "
       "dependence or evaluate an operational early-warning system. "+support)
    supplement=("Supplementary methods and results - The frozen OWID source, screening audit, complete weekly "
       "calendar definitions, fit-status records, exposure and lag spline basis dimensions and knots, pointwise "
       "contrasts, cumulative standardized contrasts, uncertainty estimates, AIC/BIC, computational times and "
       "sensitivity comparisons are provided as machine-readable files. Models are fitted per country and "
       "outcome; no cross-country or cross-outcome joint covariance is estimated. Knot and horizon sensitivity "
       "are descriptive; multiplicity and model selection are not accounted for by nominal intervals. "
       "Penalized information criteria and their confidence intervals are approximation-based and should not "
       "be used as confirmatory evidence.")
    (path/'suggested_manuscript_text.txt').write_text('\n\n'.join([method,results,discussion,supplement])+'\n',encoding='utf-8')
    (path/'reviewer_response_draft.txt').write_text(
        'We thank the reviewer for suggesting comparison with distributed lag nonlinear models. '
        'We added a matched empirical sensitivity analysis retaining the primary country-outcome observation '
        'masks, temporal adjustment and ARMA error structure. The supplemental models compare unrestricted '
        'and spline-restricted lag profiles, examine exposure nonlinearity and explore knot and horizon choices. '
        'Results are summarized in the accompanying country-specific tables. We have not interpreted in-sample '
        'fit improvements as forecasting gains or nominal confidence intervals as calibrated. '
        'Please insert verified numerical findings from the executed tables before submission.\n',
        encoding='utf-8')


def environment_record(root,args,source: Path, inclusion_match, archived):
    libs={'python':platform.python_version(), 'numpy':np.__version__,
          'pandas':pd.__version__,'scipy':scipy.__version__,
          'statsmodels':statsmodels.__version__,'patsy':patsy.__version__,
          'matplotlib':matplotlib.__version__,'requests':requests.__version__}
    d={'schema':SCHEMA_VERSION,'created_utc':datetime.now(timezone.utc).isoformat(),
       'source_url':OWID_URL,'source_commit':OWID_COMMIT,
       'source_sha256':sha256(source), 'source_bytes':source.stat().st_size,
       'original_repo':SOURCE_REPO,'input_path':str(source),
       'output_path':str(root), 'inclusion_matches_manuscript':inclusion_match,
       'archival_primary_audit':archived, 'software':libs,
       'command_line':sys.argv,'args':vars(args),
       'warning':'DLNM-style explicit Python cross-basis, not an R dlnm package call.'}
    (root/'data'/'provenance.json').write_text(json.dumps(d,indent=2),encoding='utf-8')


def zip_results(root):
    z=root/'Paper2_DLNM_outputs.zip'
    # Avoid zipping the full ~100 MB input or the ZIP itself; include hashes and
    # derived weekly data so result size is manageable for Colab downloads.
    with __import__('zipfile').ZipFile(z,'w',compression=__import__('zipfile').ZIP_DEFLATED) as f:
        for path in sorted(root.rglob('*')):
            if path==z or not path.is_file() or path.name=='owid_pinned.csv':
                continue
            f.write(path,path.relative_to(root))
    LOGGER.info('Wrote output archive: %s (%0.1f MB)',z,z.stat().st_size/1e6)

def run(args):
    root=Path(args.output).expanduser().resolve()
    make_dirs(root)
    setup_logging(root)
    LOGGER.info('Starting %s (mode=%s)',SCHEMA_VERSION,args.mode)
    src=Path(args.data).expanduser().resolve() if args.data else root/'data'/'owid_pinned.csv'
    if not args.data:download_source(src)
    if not src.exists():raise FileNotFoundError(src)
    if args.expected_sha256 and sha256(src).lower()!=args.expected_sha256.lower():
        raise ValueError('Source SHA256 mismatch. Aborting.')
    daily,daily_counts=load_daily(src)
    df_csv(daily_counts,root,'00_daily_record_screen.csv')
    datasets,audit,match=build_countries(daily,root)
    if args.strict and not match:
        raise RuntimeError('Strict manuscript reproduction QC failed: n_countryweeks != 1882 or countries != 11.')
    specs=specifications(args.mode)
    stats_rows=[]; lag_rows=[]; grid_rows=[]; surface_rows=[]
    for country,w in datasets.items():
        if args.country and country not in args.country:
            continue
        for outcome in OUTCOMES:
            for spec in specs:
                fitted=cached_fit(country,outcome,w,spec,args,root)
                stats_rows.append(fitted['stat'])
                lag_rows.extend(fitted['lagrows'])
                grid_rows.extend(fitted['gridrows'])
                surface_rows.extend(fitted.get('surface_rows', []))
                # Per-fit CSV checkpoint to safeguard Colab interruptions.
                pd.DataFrame(stats_rows).to_csv(root/'tables'/'03_model_statistics.csv',index=False)
    stats=pd.DataFrame(stats_rows)
    lag=pd.DataFrame(lag_rows)
    grid=pd.DataFrame(grid_rows)
    surface=pd.DataFrame(surface_rows)
    if stats.empty:
        raise RuntimeError('No country-outcome models requested or eligible')
    df_csv(stats,root,'03_model_statistics.csv')
    # Explicit reviewer-facing completeness gate; unsuccessful models are
    # retained and never counted as evidence of model agreement.
    n_requested=len(stats)
    n_ok=int(stats.optimizer_converged.fillna(False).astype(bool).sum())
    fit_audit=pd.DataFrame([dict(schema=SCHEMA_VERSION,mode=args.mode,
                     requested=n_requested,converged=n_ok,failed=n_requested-n_ok,
                     complete=bool(n_ok==n_requested),
                     note=('Convergence audit only; primary coefficient '
                           'archive reconciliation is separate.'))])
    df_csv(fit_audit,root,'02_fit_completeness_audit.csv')
    if n_ok < n_requested:
        LOGGER.error('FIT COMPLETENESS: %s of %s models converged. Inspect 03_model_statistics.csv; do not describe the comparison as complete.', n_ok, n_requested)
    else:
        LOGGER.info('FIT COMPLETENESS: all %s model fits converged. Archived-primary validation remains separate.',n_ok)
    df_csv(lag,root,'04_lag_response_pointwise.csv')
    df_csv(grid,root,'05_cumulative_exposure_response.csv')
    df_csv(surface,root,'05b_exposure_lag_response_surface.csv')
    if not lag.empty:
        comp=compare_models(stats,lag,args)
    else:
        comp=pd.DataFrame()
    df_csv(comp,root,'06_all_primary_DLNM_comparisons.csv')
    archived=load_archived_primary(args.archived_primary,lag,root)
    if len(comp):reviewer_tables(stats,comp,root,args)
    write_text(root,stats,comp,match,audit,args,archived)
    if not lag.empty and not grid.empty:
        figures(stats,lag,grid,root,args.featured_country)
    environment_record(root,args,src,match,archived)
    zip_results(root)
    LOGGER.info('Completed; review failed fits, inclusion QC and warnings before drawing conclusions.')


def self_test():
    """Offline unit/smoke checks; does not pretend to validate empirical estimates."""
    s=Specification('test',6,'cubic',4)
    x=np.random.default_rng(21123).normal(size=160)
    z,H,e,v=make_crossbasis(x,s)
    assert H.shape==(7,4) and z.shape==(160,4) and v.sum()==154
    unres=Specification('unres',6,'identity',7)
    zz,HH,ee,vv=make_crossbasis(x,unres)
    for lag_i in range(7):
        np.testing.assert_allclose(zz[15:,lag_i],x[15-lag_i:len(x)-lag_i],atol=1e-12)
    s2=Specification('test_nonlin',6,'cubic',4,exp_kind='natural_cubic',exp_df=4)
    z2,H2,e2,v2=make_crossbasis(x,s2)
    assert np.isfinite(z2).all() and z2.shape[0]==160 and e2.ncol==4
    assert np.linalg.norm(e2.contrast(0,0))<1e-9
    for loc in [(1,3),(2,4),(3,5)]:
        sk=Specification('test_knots',6,'cubic_knots',6,loc)
        assert make_lag_basis(sk).shape==(7,6)
    rng=np.random.default_rng(20261023)
    # Regression test for incomplete national calendars: unrestricted time
    # splines may be rank deficient when only an interval has observations.
    calendar_trend=make_trend(240)
    sampled=np.zeros(240,dtype=bool)
    sampled[30:205]=True
    repaired=identifiable_trend_columns(calendar_trend,sampled)
    assert repaired.shape[1] <= calendar_trend.shape[1]
    assert np.linalg.matrix_rank(repaired[sampled]) == repaired.shape[1]
    assert np.linalg.matrix_rank(calendar_trend[sampled]) == repaired.shape[1]
    # Use non-degenerate correlated exposures for the numerical fit test.
    # The earlier deterministic straight-line x was intentionally rank
    # deficient and cannot support observed-design QR.
    xfit=np.zeros(160)
    for i in range(1,len(xfit)):
        xfit[i]=.65*xfit[i-1]+rng.normal()
    zfit,Hfit,efit,vfit=make_crossbasis(xfit,s)
    trend=make_trend(160)
    X=np.column_stack([trend,zfit])
    yy=.4*zfit[:,1]+rng.normal(0,.3,160)
    yy[:8]=np.nan
    xx,trans,cond=orthogonalize_observed_exog(X, np.isfinite(yy))
    test_coef=rng.normal(size=X.shape[1])
    np.testing.assert_allclose(xx @ test_coef,X @ (trans @ test_coef),
                               rtol=1e-9,atol=1e-9)
    assert np.linalg.cond(xx[np.isfinite(yy)]) < 1.000001
    model,fit,warn=fit_sarimax(yy,X,maxiter=35,cov_type='oim')
    assert np.isfinite(fit.llf)
    cov,_=safe_cov(fit.cov_params())
    eff,se,cumul,cumse=predict_from_fit(fit.params,cov,trend.shape[1],Hfit,efit)
    assert len(eff)==7 and np.isfinite(cumul) and np.isfinite(cumse)
    print('SELF-TEST PASS: cross-basis, nonlinear contrast, knots, Gaussian ARMA fit, covariance, curve and cumulative predictions.')
    print('The test uses synthetic data and cannot verify national OWID input reproduction or archive identity.')

def main():
    default_out='/content/Paper2_DLNM_comparison' if Path('/content').exists() else './Paper2_DLNM_comparison'
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data',default='',help='Local OWID historical daily CSV; otherwise fetch pinned historical OWID file')
    ap.add_argument('--output',default=default_out)
    ap.add_argument('--mode',choices=['quick','full'],default='quick',
                    help='Quick: primary+linear+nonlinear; Full: adds horizon, knot and fixed-lambda sensitivity')
    ap.add_argument('--country',nargs='*',default=None,help='Optional country names for debugging; all by default')
    ap.add_argument('--featured-country',default='United States')
    ap.add_argument('--archived-primary',default='',
                    help='Optional CSV with columns country,outcome,lag,effect from the original archived primary fits')
    ap.add_argument('--expected-sha256',default='',help='Optional pinned source SHA256 assertion')
    ap.add_argument('--maxiter',type=int,default=130)
    ap.add_argument('--peak-tolerance',type=int,default=1,
                    help='Heuristic allowed lag difference in weeks, not a significance test')
    ap.add_argument('--cumulative-tolerance',type=float,default=.10,
                    help='Heuristic cumulative effect difference in standardized units')
    ap.add_argument('--strict',action='store_true',
                    help='Abort if 11 countries and 1882 country-weeks are not reconstructed')
    ap.add_argument('--force',action='store_true',help='Recompute valid checkpoints')
    ap.add_argument('--self-test',action='store_true',help='Offline implementation tests; no data download')
    args=ap.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)


# -------------------------------------------------------------------------
# Reviewer follow-up: reproducible US multistart and ARCHIVED-REFERENCE full
# model comparison. Numerical restarts NEVER use the archived response curve
# as optimizer starts or select solutions based on their archive closeness.
# -------------------------------------------------------------------------
import base64
from types import SimpleNamespace

PIPELINE_VERSION = 'dlnm_sste_v1.4.0_us_multistart_archived_reporting'
ARCHIVED_REFERENCE_SHA256 = '9cd49ceedf7e65956491ecb31e6288d088431de0d6603d062a4eea392535210a'
ARCHIVED_REFERENCE_B64 = 'Y291bnRyeSxvdXRjb21lLGxhZyxlZmZlY3QsY2lfbG93ZXIsY2lfdXBwZXIsaW50ZXJ2YWxfc3RhdHVzCkJlbGdpdW0sYWRtaXNzaW9ucywwLDEuMDc3MzgyODE5MzcwMjM5NSwwLjg2MTM4NDM1MTUxMTQ1MTQsMS4yOTMzODEyODcyMjkwMjc2LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkJlbGdpdW0sYWRtaXNzaW9ucywxLDAuMzg5ODEwNjI1MjM1MDUzNCwwLjEzODM3ODA3MjE3ODAyMzYsMC42NDEyNDMxNzgyOTIwODMyLFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkJlbGdpdW0sYWRtaXNzaW9ucywyLDAuNzc3Nzg4ODk3OTk1ODg5LDAuNTQwODUwNzI1MDA0NTMwNCwxLjAxNDcyNzA3MDk4NzI0NzgsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKQmVsZ2l1bSxhZG1pc3Npb25zLDMsMC4wMjQ4NDQzNzYxNTA4NzMzLC0wLjExNzIwOTU2ODM1MjUwOTUsMC4xNjY4OTgzMjA2NTQyNTYyLFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkJlbGdpdW0sYWRtaXNzaW9ucyw0LC0wLjExMDYwOTcxNTc4ODE3NTEsLTAuMjcwNDg5MjgyODMxODEwOCwwLjA0OTI2OTg1MTI1NTQ2MDUsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKQmVsZ2l1bSxhZG1pc3Npb25zLDUsLTAuMDcxNjI1NjY4MTYyODE3NiwtMC4yODQ3MTE2NjkxNjEzMzkxLDAuMTQxNDYwMzMyODM1NzA0LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkJlbGdpdW0sYWRtaXNzaW9ucyw2LDAuMDc5NTg0NzE1NTY1NTkxMywtMC4xMDY5MDEwNDkyMTAyNDIxLDAuMjY2MDcwNDgwMzQxNDI0OCxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApCZWxnaXVtLG9jY3VwYW5jeSwwLDAuMzMwMjQwNDQwNzAxNzM5MSwwLjE4ODI4NzE0MzMzOTA5NTUsMC40NzIxOTM3MzgwNjQzODI3LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkJlbGdpdW0sb2NjdXBhbmN5LDEsMC43MDg1ODU0NTk5Mzc4MzAzLDAuNTQzNzY3MTQxODQ2NTAyOCwwLjg3MzQwMzc3ODAyOTE1NzgsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKQmVsZ2l1bSxvY2N1cGFuY3ksMiwwLjY5MjkzOTQxNDUzMjU3MDIsMC40MTI2MDE3NjgzNjE3MjM3LDAuOTczMjc3MDYwNzAzNDE2OCxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApCZWxnaXVtLG9jY3VwYW5jeSwzLDAuMTQ4MjEzMDA4MTA2NjIyNCwwLjA4NTUwNTc3MTU5NTQzNjEsMC4yMTA5MjAyNDQ2MTc4MDg2LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkJlbGdpdW0sb2NjdXBhbmN5LDQsLTAuMDA1MzMxNzkyMDY3NzEwNCwtMC4xNzgzMDc5MzM2MTEzMjcxLDAuMTY3NjQ0MzQ5NDc1OTA2MSxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApCZWxnaXVtLG9jY3VwYW5jeSw1LDAuMDM0MjE2MDMzNzIzNTU5NCwtMC4wOTgzOTkzNDgzMzc5MTksMC4xNjY4MzE0MTU3ODUwMzc5LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkJlbGdpdW0sb2NjdXBhbmN5LDYsMC4xMDk5OTE3MTg3MDQ3ODMzLDAuMDM4NDc2NTYxNjA1NjM4NSwwLjE4MTUwNjg3NTgwMzkyODEsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKQ3plY2hpYSxhZG1pc3Npb25zLDAsMC43Nzg3MDU1NTI1NTg5MzU4LDAuNjU1OTM0MTI1MTU0NjgyMiwwLjkwMTQ3Njk3OTk2MzE4OTQsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKQ3plY2hpYSxhZG1pc3Npb25zLDEsMC4wNzAxNjY2NzYyOTc5MzgsLTAuMDg5OTk5MzQ2MzUxOTg2MSwwLjIzMDMzMjY5ODk0Nzg2MjMsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKQ3plY2hpYSxhZG1pc3Npb25zLDIsMC4yOTUzNzMzNzA0MDYwOTM0LDAuMTI5OTYwODU0ODgyMjYwNSwwLjQ2MDc4NTg4NTkyOTkyNjMsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKQ3plY2hpYSxhZG1pc3Npb25zLDMsMC4wMzQ1ODk4NDA0MDAyMjA4LC0wLjA5NTg4NDk0MDkxMDQ2OTMsMC4xNjUwNjQ2MjE3MTA5MTExLFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkN6ZWNoaWEsYWRtaXNzaW9ucyw0LDAuMDE2MzgyNzUxODQyMjYzMSwtMC4xMDk1ODU5NjA4OTUyMjkyLDAuMTQyMzUxNDY0NTc5NzU1NixQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApDemVjaGlhLGFkbWlzc2lvbnMsNSwwLjA3NTgxMzM5ODM4MDAzMDIsLTAuMDY5NjE4NTY4ODE1NjkzMSwwLjIyMTI0NTM2NTU3NTc1MzUsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKQ3plY2hpYSxhZG1pc3Npb25zLDYsLTAuMDU3MDE4NDM4OTgzMzYzLC0wLjE1NTI1MTM1NDQwNTgzNTksMC4wNDEyMTQ0NzY0MzkxMDk4LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkN6ZWNoaWEsb2NjdXBhbmN5LDAsMC40NTM1NzExOTc0NDg2MTgzLDAuMzQ2OTQ4NTkxODY2MDUzNiwwLjU2MDE5MzgwMzAzMTE4MyxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApDemVjaGlhLG9jY3VwYW5jeSwxLDAuMjQ2Mzg3MTI4NjUzMjUzNywwLjEzNjQ5OTA0NjU0NjI5ODQsMC4zNTYyNzUyMTA3NjAyMDksUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKQ3plY2hpYSxvY2N1cGFuY3ksMiwwLjIxNDg3Mjc2NzA4NDA0MjEsMC4xMDk2NzQxMjg0NTkzMTI3LDAuMzIwMDcxNDA1NzA4NzcxNSxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApDemVjaGlhLG9jY3VwYW5jeSwzLDAuMjA5MjU2MDk2MDg2MTc0NCwwLjEwOTk2NDM5NzQyNzkzMTYsMC4zMDg1NDc3OTQ3NDQ0MTcyLFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkN6ZWNoaWEsb2NjdXBhbmN5LDQsMC4wNDQyOTE2MTM4Nzg1MzAxLC0wLjA1NjM2NDcxNTExMTM0NTYsMC4xNDQ5NDc5NDI4Njg0MDU5LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkN6ZWNoaWEsb2NjdXBhbmN5LDUsMC4wOTIzMzMxMDIwOTg1NjksMC4wMTA0MDMyMDAxNjEyMjQ2LDAuMTc0MjYzMDA0MDM1OTEzNSxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApDemVjaGlhLG9jY3VwYW5jeSw2LC0wLjAxNTA4MDMyNTQ5OTQzMzksLTAuMDk1MjQ5MjY2MDc2NTQ0NywwLjA2NTA4ODYxNTA3NzY3NjcsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKRGVubWFyayxhZG1pc3Npb25zLDAsMS4yNDQzODEzMDM2Nzg2NzMzLDAuOTYyNzQ3NzQ2NzAwNTgyNCwxLjUyNjAxNDg2MDY1Njc2NDUsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKRGVubWFyayxhZG1pc3Npb25zLDEsMC4zMjg5MDE4NjMxMDMxODkzLC0wLjAwMDE5NzcyNzA3NjY0NTMsMC42NTgwMDE0NTMyODMwMjQsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKRGVubWFyayxhZG1pc3Npb25zLDIsMC4zNjk2NTIxNzg5NjI2MDE0LDAuMDY3Mjk5NjM5OTYwMDcwNSwwLjY3MjAwNDcxNzk2NTEzMjMsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKRGVubWFyayxhZG1pc3Npb25zLDMsLTAuMDcwMDc3NzkxOTA3MjMwOSwtMC4zNTUyNjM0ODQzODU5NjMyLDAuMjE1MTA3OTAwNTcxNTAxMyxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApEZW5tYXJrLGFkbWlzc2lvbnMsNCwtMC4xNzI5MjUzMTAxOTI2MDE2LC0wLjQyMzg5MDg3ODU2NTEwNzcsMC4wNzgwNDAyNTgxNzk5MDQ1LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkRlbm1hcmssYWRtaXNzaW9ucyw1LDAuMDIzMTM2MDAxNzczMzUsLTAuMTEyODMzNzIzNzE1ODgxNiwwLjE1OTEwNTcyNzI2MjU4MTgsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKRGVubWFyayxhZG1pc3Npb25zLDYsMC4xMjgwMjcxMzI5ODYxMjMxLC0wLjAwMzI3MTA4MTY2NDIwNjUsMC4yNTkzMjUzNDc2MzY0NTI3LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkRlbm1hcmssb2NjdXBhbmN5LDAsMC40MTM4NjIyMDA5NTYxMDAxLDAuMjU0MTM5NzE2MTczMjI4MywwLjU3MzU4NDY4NTczODk3MixQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApEZW5tYXJrLG9jY3VwYW5jeSwxLDAuNTgxOTE2NTgxNDU0NTExMywwLjMzNDA4NzQyNDM0MzkxOTMsMC44Mjk3NDU3Mzg1NjUxMDMzLFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkRlbm1hcmssb2NjdXBhbmN5LDIsMC40MzA4MTk5NTI4NjM2NjUxLDAuMjA1MDg1MzkwMTc3NjY5NCwwLjY1NjU1NDUxNTU0OTY2MDcsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKRGVubWFyayxvY2N1cGFuY3ksMywwLjE5MjA4ODg1MDU3OTQ3OSwtMC4wMjM3NjA2OTk0OTk0OTU3LDAuNDA3OTM4NDAwNjU4NDUzNyxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApEZW5tYXJrLG9jY3VwYW5jeSw0LDAuMTYzNzYyMTIwMDQ4NzI0NCwtMC4wODUxNTIxNTA0NjcxMDc0LDAuNDEyNjc2MzkwNTY0NTU2MyxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApEZW5tYXJrLG9jY3VwYW5jeSw1LDAuMDA3ODY5OTU4OTkxMzIxMSwtMC4wNzc2NTM0ODg1MDE0MDQsMC4wOTMzOTM0MDY0ODQwNDYzLFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkRlbm1hcmssb2NjdXBhbmN5LDYsMC4xNTg1NjI3MjcxNzI5MTcsMC4wMzkzMTA3ODQxNTA5MTU4LDAuMjc3ODE0NjcwMTk0OTE4MixQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApGcmFuY2UsYWRtaXNzaW9ucywwLDAuNjE2Mjk3NTM0MDQ3NjU2NiwwLjE1ODUzMDMxNDQzMzUxNTQsMS4wNzQwNjQ3NTM2NjE3OTc4LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkZyYW5jZSxhZG1pc3Npb25zLDEsMC4yMjE4MjIwMDA2MTY2OTgyLC0wLjM1NDQ0Njk3ODc0NDExNzQsMC43OTgwOTA5Nzk5Nzc1MTM5LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkZyYW5jZSxhZG1pc3Npb25zLDIsMC41MTkyODUwNTY2MzA1OTk3LDAuMDI3NjA1MTU3MDg1NTk4MSwxLjAxMDk2NDk1NjE3NTYwMTEsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKRnJhbmNlLGFkbWlzc2lvbnMsMywtMC4wMzUzMjU5MDM0NjU3OTYyLC0wLjM1OTAxNzA2MTEzOTU2NzQsMC4yODgzNjUyNTQyMDc5NzUsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKRnJhbmNlLGFkbWlzc2lvbnMsNCwwLjAzMTU2OTM1NzY4OTUwMTgsLTAuMzc5NzI3OTc4NjI5MDc3LDAuNDQyODY2Njk0MDA4MDgwNixQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApGcmFuY2UsYWRtaXNzaW9ucyw1LDAuMDg2NDU1OTUzMDQ3NDI0OSwtMC4zODMxOTY2MDYyNjg5ODIyLDAuNTU2MTA4NTEyMzYzODMyMSxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApGcmFuY2UsYWRtaXNzaW9ucyw2LC0wLjE2NTYwMDYyMjcxMDMyNywtMC42Nzg4OTQ1MDI0MzU3OTUyLDAuMzQ3NjkzMjU3MDE1MTQxMSxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApGcmFuY2Usb2NjdXBhbmN5LDAsMC4xNjE0Mjg3MTc5OTU0MjYzLDAuMDUyODQzNDE1OTY4MzYxOCwwLjI3MDAxNDAyMDAyMjQ5MDgsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKRnJhbmNlLG9jY3VwYW5jeSwxLDAuMjMzMDQ4OTYwNjk1OTY3MSwwLjEzODA4MzQ5MDE5MTc0NzYsMC4zMjgwMTQ0MzEyMDAxODY2LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkZyYW5jZSxvY2N1cGFuY3ksMiwwLjI4MzY5MTQ5MDU3MTAxODgsMC4xODYxOTk1ODM1MjczNDc5LDAuMzgxMTgzMzk3NjE0Njg5NixQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApGcmFuY2Usb2NjdXBhbmN5LDMsMC4yNDQ4OTU3NDEwNTgxNTY1LDAuMTQzNjU0Njk3MzQ4ODQzNSwwLjM0NjEzNjc4NDc2NzQ2OTYsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKRnJhbmNlLG9jY3VwYW5jeSw0LDAuMTc0NDg4Njc4OTAzMTY0NywwLjA3ODYzODY4NjcyMDc1NjYsMC4yNzAzMzg2NzEwODU1NzI4LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkZyYW5jZSxvY2N1cGFuY3ksNSwwLjEyOTM4MzI2MDY1NjQ2OTYsMC4wNTA5Nzg3MTAzNjAyMzgxLDAuMjA3Nzg3ODEwOTUyNzAxMixQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApGcmFuY2Usb2NjdXBhbmN5LDYsMC4wNjgyMzgzMjUzMTcwNTY4LDAuMDExOTM1MDg5NDU2NjA1NSwwLjEyNDU0MTU2MTE3NzUwODEsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKSXNyYWVsLGFkbWlzc2lvbnMsMCwxLjM2OTk3MDgzNjg2ODczMDQsMC45NjU3NzQ4Mzg3OTU1NDEyLDEuNzc0MTY2ODM0OTQxOTE5NSxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApJc3JhZWwsYWRtaXNzaW9ucywxLDAuNjYxMDEwODQ4MDU2Mjc2NiwwLjEwNTY2ODAwMTkxMzIyMiwxLjIxNjM1MzY5NDE5OTMzMTEsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKSXNyYWVsLGFkbWlzc2lvbnMsMiwwLjMwNzA0NzcwMjExODgzMTUsLTAuMjU1NjE2NDY2NTE4NDExMiwwLjg2OTcxMTg3MDc1NjA3NDEsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKSXNyYWVsLGFkbWlzc2lvbnMsMywtMC42MjgzNDYzMDMwMDU3OTQsLTEuNjE3NTY1ODUwMTE5OTg4MywwLjM2MDg3MzI0NDEwODQwMDcsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKSXNyYWVsLGFkbWlzc2lvbnMsNCwwLjc1NTI0NzcwNjQwNzgxNSwtMC4xNzY3NTgwODExNzk5MTkzLDEuNjg3MjUzNDkzOTk1NTQ5NyxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApJc3JhZWwsYWRtaXNzaW9ucyw1LC0wLjMyMzQxMDQ5NTU3NzUxMzEsLTAuODI0NjQyODE4MjAwNjI1MywwLjE3NzgyMTgyNzA0NTU5OTEsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKSXNyYWVsLGFkbWlzc2lvbnMsNiwtMC4wNDgxNDE3OTczNjA2NTQ0LC0wLjMxNzg2MjczNjUyMTkwMjgsMC4yMjE1NzkxNDE4MDA1OTQsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKSXNyYWVsLG9jY3VwYW5jeSwwLDAuNjY3MzM1Mzk2OTYzMjAyMiwwLjM2MTEzNzc3Nzg2ODk0NDQsMC45NzM1MzMwMTYwNTc0NixQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApJc3JhZWwsb2NjdXBhbmN5LDEsMC41NTE2NzI2NTkzNzUyODkyLDAuMzY5ODQ4NDc2MTY1NjU0NywwLjczMzQ5Njg0MjU4NDkyMzYsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKSXNyYWVsLG9jY3VwYW5jeSwyLDAuNTExMDIzMDIzODczMDY5MywwLjI1NjcyNzc2MDc0ODY3NjgsMC43NjUzMTgyODY5OTc0NjE3LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCklzcmFlbCxvY2N1cGFuY3ksMywwLjE0OTMyMTU5ODMzMzIxNTQsLTAuMjQzNjgwMTA0MTIyNzkzMiwwLjU0MjMyMzMwMDc4OTIyNDIsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKSXNyYWVsLG9jY3VwYW5jeSw0LDAuNTYzNzQzNDY0OTg0MDA5NCwwLjA5OTExOTY3MDI3MjU3NjksMS4wMjgzNjcyNTk2OTU0NDE4LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCklzcmFlbCxvY2N1cGFuY3ksNSwwLjA2MTM2ODQ3MzIxNTcyNTEsLTAuMTYyNTMwOTc4NjY0NTk4OCwwLjI4NTI2NzkyNTA5NjA0OTEsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKSXNyYWVsLG9jY3VwYW5jeSw2LC0wLjMwODU0OTU5NzA3NzEwNjUsLTAuNjI0NjQyNDE2NTQ2MTMwMiwwLjAwNzU0MzIyMjM5MTkxNzEsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKSXRhbHksYWRtaXNzaW9ucywwLDAuOTY2MTU5MjAxNjM5NTUwNiwwLjgxNjI0MDkyMjkzMDgwMzksMS4xMTYwNzc0ODAzNDgyOTcxLFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkl0YWx5LGFkbWlzc2lvbnMsMSwwLjIyOTEwNjc1MzE0NTI5OTYsMC4wNzYyNjYzNjA0Mjg0NDU5LDAuMzgxOTQ3MTQ1ODYyMTUzMyxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApJdGFseSxhZG1pc3Npb25zLDIsMC4xMjAxOTI1NTczMDY2MTUzLC0wLjAxMzM2MzMxNDExNzAwMDgsMC4yNTM3NDg0Mjg3MzAyMzE2LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkl0YWx5LGFkbWlzc2lvbnMsMywtMC4wMTE3MjA3MjAwMzA2NTk3LC0wLjEzNzU0MDgwODQxMDAzNjYsMC4xMTQwOTkzNjgzNDg3MTcxLFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkl0YWx5LGFkbWlzc2lvbnMsNCwtMC4wNTQ2MTE2ODE4MzMzNTUzLC0wLjE5MjY0OTUzMTAxMjIwMjIsMC4wODM0MjYxNjczNDU0OTE2LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkl0YWx5LGFkbWlzc2lvbnMsNSwtMC4wNjI0NDEyNDQwNjI2NjQ2LC0wLjIyODgzOTA0MzMyNzExNTMsMC4xMDM5NTY1NTUyMDE3ODYsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKSXRhbHksYWRtaXNzaW9ucyw2LDAuMDMxOTM5Mzg4MjAwMTI1NCwtMC4wNjkxNjE4MDYxMTE0MjQ4LDAuMTMzMDQwNTgyNTExNjc1NyxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApJdGFseSxvY2N1cGFuY3ksMCwwLjM1MTAxOTI5MTE2OTM1MDUsMC4yNjA3Mzg5NDU1NzUwNzQxLDAuNDQxMjk5NjM2NzYzNjI2OSxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApJdGFseSxvY2N1cGFuY3ksMSwwLjUwMDI0MzEyMjk5NzE5MjgsMC4zOTYxNzQwMzA4OTE2MDE3LDAuNjA0MzEyMjE1MTAyNzgzOSxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApJdGFseSxvY2N1cGFuY3ksMiwwLjM3NTU2MzY0ODIyMjUwMTUsMC4yNzAzNDI5NzI3NDcwOTc4LDAuNDgwNzg0MzIzNjk3OTA1MyxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApJdGFseSxvY2N1cGFuY3ksMywwLjIxMDIzMTk2ODExNzE1MzksMC4wNzU1MDM1NjAwMTM2MTM3LDAuMzQ0OTYwMzc2MjIwNjk0LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCkl0YWx5LG9jY3VwYW5jeSw0LDAuMDU4NTIyNzQwODkyNzczMSwtMC4wMzUwODM5MTI1MjMzMjg0LDAuMTUyMTI5Mzk0MzA4ODc0NixQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApJdGFseSxvY2N1cGFuY3ksNSwwLjA0MzIzMTU2MTAyNTI3MDEsLTAuMDM4NzkxMzI1NzY1OTI3OSwwLjEyNTI1NDQ0NzgxNjQ2ODMsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKSXRhbHksb2NjdXBhbmN5LDYsMC4xMDA2OTAzOTYzMTY4NTQ0LDAuMDIzNTkzMjEzNjUyNjAzOCwwLjE3Nzc4NzU3ODk4MTEwNTEsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKTWFsYXlzaWEsYWRtaXNzaW9ucywwLDAuMTk0OTk0MTE0NjQ2NTc4MywwLjAxODgxOTEwOTI0ODkyOTQsMC4zNzExNjkxMjAwNDQyMjcxLFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCk1hbGF5c2lhLGFkbWlzc2lvbnMsMSwwLjIyODU3OTM5Mjk0NjA5NjIsMC4xMjY3NjM0NzkwOTkwMjM1LDAuMzMwMzk1MzA2NzkzMTY4OCxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApNYWxheXNpYSxhZG1pc3Npb25zLDIsMC4xODcwMjcwODAzMDAxNDE2LDAuMTE1MTEzODA4NjcxMzMwMywwLjI1ODk0MDM1MTkyODk1MyxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApNYWxheXNpYSxhZG1pc3Npb25zLDMsMC4xNjk3MzAzMDQ5MTE4Mjc4LDAuMDkxODA4NTkxMzAyMDExOCwwLjI0NzY1MjAxODUyMTY0MzgsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKTWFsYXlzaWEsYWRtaXNzaW9ucyw0LDAuMDcxMDYxODU4NzEwMTc4OSwtMC4wMjcwNzYwMjc3Nzk5MzYxLDAuMTY5MTk5NzQ1MjAwMjk0LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCk1hbGF5c2lhLGFkbWlzc2lvbnMsNSwwLjAyMDIwMTMzOTkyMjIyNDEsLTAuMDY4NDI1NTgwMzY4OTgzMywwLjEwODgyODI2MDIxMzQzMTYsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKTWFsYXlzaWEsYWRtaXNzaW9ucyw2LC0wLjA3MDk5MDczNDQzNzUsLTAuMTM2NjQwNTkxNTI2OTk5MywtMC4wMDUzNDA4NzczNDgwMDA4LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCk1hbGF5c2lhLG9jY3VwYW5jeSwwLDAuMDk1Mzk2NzY1MzQxMzQ5NCwtMC4wMDY5NzAxODgwNzQ0MTg2LDAuMTk3NzYzNzE4NzU3MTE3NSxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApNYWxheXNpYSxvY2N1cGFuY3ksMSwwLjE1MDcxNjkzMTQ0MDIzLDAuMDU2ODcyNjUyNjE2Njg2LDAuMjQ0NTYxMjEwMjYzNzczOSxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApNYWxheXNpYSxvY2N1cGFuY3ksMiwwLjE3NTI5MjA0ODA2NjUyMjQsMC4wOTM3MTQwMDQ2MTQwNjQxLDAuMjU2ODcwMDkxNTE4OTgwNyxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApNYWxheXNpYSxvY2N1cGFuY3ksMywwLjE0NzQzNjU4MTAzNDA5OTUsMC4wNzI5MDcxNDA4NDgyMTUzLDAuMjIxOTY2MDIxMjE5OTgzNyxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApNYWxheXNpYSxvY2N1cGFuY3ksNCwwLjEwMTA3OTQyMTM5NzI4NjcsMC4wMjE4OTcxMjgyNDEzNDYxLDAuMTgwMjYxNzE0NTUzMjI3MyxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApNYWxheXNpYSxvY2N1cGFuY3ksNSwtMC4wMDg3NTU3ODI4NTMxODc4LC0wLjA4NDg0ODEyMDYzODgzODksMC4wNjczMzY1NTQ5MzI0NjMyLFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsCk1hbGF5c2lhLG9jY3VwYW5jeSw2LC0wLjAxMDk5MTY3MTQ4MTI5MTcsLTAuMDY3NTU3MDAyNDQ2NDU3NCwwLjA0NTU3MzY1OTQ4Mzg3MzksUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKU3BhaW4sYWRtaXNzaW9ucywwLDAuOTk1ODM5Mzc1Mjc0OTk5NSwwLjY4OTI4ODk2MzIzNDMwNCwxLjMwMjM4OTc4NzMxNTY5NSxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApTcGFpbixhZG1pc3Npb25zLDEsMC41NTIxMDYyODE3ODcwODM5LDAuMzIzMDcyMzE0Nzk0NjkzNCwwLjc4MTE0MDI0ODc3OTQ3NDMsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKU3BhaW4sYWRtaXNzaW9ucywyLDAuNDM5MzI1NTcyNTk4NjMzNiwwLjIxMzQxOTEzNDcxMjQzNCwwLjY2NTIzMjAxMDQ4NDgzMzIsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKU3BhaW4sYWRtaXNzaW9ucywzLDAuMTg3Mzk0NTUxODMxMTU3NCwtMC4wNzkzNDc0NTMzMDY0OTg5LDAuNDU0MTM2NTU2OTY4ODEzOCxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApTcGFpbixhZG1pc3Npb25zLDQsLTAuMTUxOTY0MTI3NjU1MjUzNSwtMC40MTUxNDg5ODg3NTc4NCwwLjExMTIyMDczMzQ0NzMzMjgsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKU3BhaW4sYWRtaXNzaW9ucyw1LC0wLjAwOTY3MzY1MTkzNzQ1MzQsLTAuMTg1NjM3MDYzMzc4MjM3MSwwLjE2NjI4OTc1OTUwMzMzMDIsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKU3BhaW4sYWRtaXNzaW9ucyw2LC0wLjAxMjY4MTExMzc1NjA5MywtMC4xNzc1OTgyMDc4NTk0NTUxLDAuMTUyMjM1OTgwMzQ3MjY4OSxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApTcGFpbixvY2N1cGFuY3ksMCwwLjUwMzg4NzA5NDE0MTM3NjIsMC4zMTQyMjMyODM4NzM1NDA5LDAuNjkzNTUwOTA0NDA5MjExNSxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApTcGFpbixvY2N1cGFuY3ksMSwwLjQ5MzgwNjQ0Njk2NTkzNTgsMC4zMzUyMTA2NTA1MDI2OTk2LDAuNjUyNDAyMjQzNDI5MTcyMSxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApTcGFpbixvY2N1cGFuY3ksMiwwLjYyMTA4NzIwNzQyNDM4NDEsMC40NDUwMDk2MDg4NTgwNjU3LDAuNzk3MTY0ODA1OTkwNzAyNCxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApTcGFpbixvY2N1cGFuY3ksMywwLjIwMzczMjk3NTUzNzU1MTksLTAuMTAzNjY0NzY3NTg5MjY1MiwwLjUxMTEzMDcxODY2NDM2OTIsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKU3BhaW4sb2NjdXBhbmN5LDQsMC4wNTgyNTk1Mjc1NTYxODM4LC0wLjE0NzUyMzE3NjAyNDQwMTUsMC4yNjQwNDIyMzExMzY3NjkzLFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsClNwYWluLG9jY3VwYW5jeSw1LC0wLjAwMDcyNTA4MTY4NDM2NiwtMC4xMDk2Mjk3MDc0Nzg0OTA5LDAuMTA4MTc5NTQ0MTA5NzU4OCxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApTcGFpbixvY2N1cGFuY3ksNiwtMC4wMTUzMjUzNzY4NTI5MjA1LC0wLjEyMzQzODgyMDk0MTcyNSwwLjA5Mjc4ODA2NzIzNTg4MzksUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKU3dpdHplcmxhbmQsYWRtaXNzaW9ucywwLDEuMDUzNzgyMjExNTQzNzQ3OCwwLjY4NDU1MDgwMDA4Mzc0MTksMS40MjMwMTM2MjMwMDM3NTM4LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsClN3aXR6ZXJsYW5kLGFkbWlzc2lvbnMsMSwwLjY3NTY2NTIwODc0NDE4NDcsMC4xNTg1ODIwMTYyMTAyNjMyLDEuMTkyNzQ4NDAxMjc4MTA2MixQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApTd2l0emVybGFuZCxhZG1pc3Npb25zLDIsLTAuMjc4NzMzNDExODA1MDk3MiwtMC43ODc3OTE3NzE2ODAzNDA0LDAuMjMwMzI0OTQ4MDcwMTQ1OSxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApTd2l0emVybGFuZCxhZG1pc3Npb25zLDMsMC4wNDA1NDAzODQ5MDQ4NTEsLTAuNDMzNTUxOTg3Mzk0MTY5MywwLjUxNDYzMjc1NzIwMzg3MTUsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKU3dpdHplcmxhbmQsYWRtaXNzaW9ucyw0LDAuMjcwMjIzMzQzNjc4OTIyNywtMC4xNjU4Mzg5ODUxNjkxMzgyLDAuNzA2Mjg1NjcyNTI2OTgzNyxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApTd2l0emVybGFuZCxhZG1pc3Npb25zLDUsLTAuMjEyODM0NjM4NDgxOTE3MywtMC42MTE0OTY4MjM5NTkyNTY2LDAuMTg1ODI3NTQ2OTk1NDIxOSxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApTd2l0emVybGFuZCxhZG1pc3Npb25zLDYsLTAuMDQ4NzAzMzI0NzIxODgwNSwtMC4zMDg3NDQ0OTYxMTMzMzE0LDAuMjExMzM3ODQ2NjY5NTcwMyxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApTd2l0emVybGFuZCxvY2N1cGFuY3ksMCwwLjQ0Mzg1MTA5NDAzNDA5MjQsMC4yNDIxNjA1NDAwMjg2MzgsMC42NDU1NDE2NDgwMzk1NDY4LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsClN3aXR6ZXJsYW5kLG9jY3VwYW5jeSwxLDAuNTUxODQ0NzA1MDkyNzUsMC4zMTc5ODQzNjU3MjAxMDc4LDAuNzg1NzA1MDQ0NDY1MzkyMSxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApTd2l0emVybGFuZCxvY2N1cGFuY3ksMiwwLjUxMjM0NDY5NDI3NjgxNjYsMC4zMjA0NjM3MTY2NDE2MzQ3LDAuNzA0MjI1NjcxOTExOTk4NSxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApTd2l0emVybGFuZCxvY2N1cGFuY3ksMywwLjMxNTgzMDE1Njg4MjYzLDAuMDc3NTM1MDY1MjE3Mjk4OSwwLjU1NDEyNTI0ODU0Nzk2MTEsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKU3dpdHplcmxhbmQsb2NjdXBhbmN5LDQsLTAuMTEwODQ0OTQzODExNDk3NCwtMC4zMDk5NDQ5MTk0MTY2MjQ5LDAuMDg4MjU1MDMxNzkzNjMsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKU3dpdHplcmxhbmQsb2NjdXBhbmN5LDUsMC4xMjczNjE0NDU0OTMxMzUsLTAuMDczMjQ3MjY2MjYzODc4NywwLjMyNzk3MDE1NzI1MDE0ODgsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKU3dpdHplcmxhbmQsb2NjdXBhbmN5LDYsMC4wNTYwNjg2MTU4MzAxNDExLC0wLjEwOTg0MDUyMTcxMDI3NzksMC4yMjE5Nzc3NTMzNzA1NjAzLFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsClVuaXRlZCBLaW5nZG9tLGFkbWlzc2lvbnMsMCwwLjY5NjAyNTg4NzI3ODIxMzksMC41MTg3MjYwNDM2NTUyMzcxLDAuODczMzI1NzMwOTAxMTkwNyxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApVbml0ZWQgS2luZ2RvbSxhZG1pc3Npb25zLDEsMC41MzczMTAzOTExNTU4NzU0LDAuMzQwNzQzNDYyOTc4NDA5NSwwLjczMzg3NzMxOTMzMzM0MTMsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKVW5pdGVkIEtpbmdkb20sYWRtaXNzaW9ucywyLDAuMjQyODg5NzM4NjQ2NDg3NCwwLjA0NzczOTYxMTkwMjk4NiwwLjQzODAzOTg2NTM4OTk4ODksUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKVW5pdGVkIEtpbmdkb20sYWRtaXNzaW9ucywzLC0wLjAxMzE1MTEzMjQ4NDEzNDMsLTAuMjAyNzk1ODg1Nzk4MzU2NywwLjE3NjQ5MzYyMDgzMDA4OCxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApVbml0ZWQgS2luZ2RvbSxhZG1pc3Npb25zLDQsMC4xMjgwNDM2MDUzNDIyNDExLC0wLjAxOTM2OTEwOTI3ODUxMDksMC4yNzU0NTYzMTk5NjI5OTMyLFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsClVuaXRlZCBLaW5nZG9tLGFkbWlzc2lvbnMsNSwtMC4xMDg2MTIzMzMxNDc0MDkyLC0wLjI2MTQ1NDgwODY5NTk5MDcsMC4wNDQyMzAxNDI0MDExNzIyLFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsClVuaXRlZCBLaW5nZG9tLGFkbWlzc2lvbnMsNiwtMC4wNTI4MTE3NTkzOTU2Mzc3LC0wLjE4MTkxMzY3NjQ5MjQ1ODksMC4wNzYyOTAxNTc3MDExODMzLFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsClVuaXRlZCBLaW5nZG9tLG9jY3VwYW5jeSwwLDAuMzM4MDExNjg5NTQwMjE1MiwwLjIxNTM3OTEzNzc3Mjk5MDYsMC40NjA2NDQyNDEzMDc0Mzk5LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsClVuaXRlZCBLaW5nZG9tLG9jY3VwYW5jeSwxLDAuNDIwNjc1Nzk5MTcwOTkyNCwwLjMwMDIzMzY4MzAxNDcxMjEsMC41NDExMTc5MTUzMjcyNzI3LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsClVuaXRlZCBLaW5nZG9tLG9jY3VwYW5jeSwyLDAuNTQ1NTA2ODcxMTE1OTM1MSwwLjM5NTAyNDgxNzM4NjA0MzUsMC42OTU5ODg5MjQ4NDU4MjY2LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsClVuaXRlZCBLaW5nZG9tLG9jY3VwYW5jeSwzLDAuMTkyNDI5MDkzNjU2MzkyNywwLjA3MDM2NzE1MzEzODU0MjksMC4zMTQ0OTEwMzQxNzQyNDI2LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsClVuaXRlZCBLaW5nZG9tLG9jY3VwYW5jeSw0LDAuMTQyODkxNDA1NjU4MDEwMiwwLjAyMzI5ODc0NDk3Njc0NDQsMC4yNjI0ODQwNjYzMzkyNzYxLFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsClVuaXRlZCBLaW5nZG9tLG9jY3VwYW5jeSw1LC0wLjAxMDY5NDg1NzQwMzUyNzMsLTAuMTE5MDYxNjk5NzE4MzM0MiwwLjA5NzY3MTk4NDkxMTI3OTUsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKVW5pdGVkIEtpbmdkb20sb2NjdXBhbmN5LDYsMC4wMDg1NDk5MTAwMDU5MDI4LC0wLjA4Njk5MTc0NzY5MzAxMzQsMC4xMDQwOTE1Njc3MDQ4MTkyLFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsClVuaXRlZCBTdGF0ZXMsYWRtaXNzaW9ucywwLDAuODA1MjE4MDUyNjU0ODAwMSwwLjQ5MTY0NjgyNTE2ODExNzgsMS4xMTg3ODkyODAxNDE0ODI1LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsClVuaXRlZCBTdGF0ZXMsYWRtaXNzaW9ucywxLDAuNzY3NjkzNjIxMjYyNzg3MiwwLjYwMDQzMTE2NDU5NDM1MDQsMC45MzQ5NTYwNzc5MzEyMjQsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKVW5pdGVkIFN0YXRlcyxhZG1pc3Npb25zLDIsMC41MTAxNTkwNjI0NjkzMTk5LDAuMjM1Njc0Njk2MjE3NDc3OCwwLjc4NDY0MzQyODcyMTE2MTksUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKVW5pdGVkIFN0YXRlcyxhZG1pc3Npb25zLDMsMC4yODgwODE0NzY1NzE2MjI1LDAuMDMxNzI3NjYwOTAwMzc2NywwLjU0NDQzNTI5MjI0Mjg2ODQsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKVW5pdGVkIFN0YXRlcyxhZG1pc3Npb25zLDQsLTAuMDY4MTYzNjgwMzQ2NzUxNiwtMC4yNTA0MDAyMzM1MjgyMjk2LDAuMTE0MDcyODcyODM0NzI2MyxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApVbml0ZWQgU3RhdGVzLGFkbWlzc2lvbnMsNSwtMC4xNDY3NDM4MjI5MDQ1OTcyLC0wLjMzMDE0MTI5NzEyNjgzNTQsMC4wMzY2NTM2NTEzMTc2NDEsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKVW5pdGVkIFN0YXRlcyxhZG1pc3Npb25zLDYsLTAuMDg2MDM2MzgxNTg2ODEwMiwtMC4zMjM2NTMwMDc3MzIwNDczLDAuMTUxNTgwMjQ0NTU4NDI2NyxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApVbml0ZWQgU3RhdGVzLG9jY3VwYW5jeSwwLDAuMzE1OTQ2OTEzMDkzMzQzOSwwLjE4MDM4NjkwMzY1Njk3MjksMC40NTE1MDY5MjI1Mjk3MTQ5LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsClVuaXRlZCBTdGF0ZXMsb2NjdXBhbmN5LDEsMC42MjA1MjI0NTQ2MTI5MDM3LDAuNDY5MjgyNDE5NzIxODI5NiwwLjc3MTc2MjQ4OTUwMzk3NzcsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKVW5pdGVkIFN0YXRlcyxvY2N1cGFuY3ksMiwwLjUzNTExMTQxNTQ5MjQ5MDcsMC40MDA5NjI3OTQ1NTkwODI4LDAuNjY5MjYwMDM2NDI1ODk4NyxQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbApVbml0ZWQgU3RhdGVzLG9jY3VwYW5jeSwzLDAuMzU5NzM0NTI1OTY0Njk0OSwwLjE4NTkwNTc5NzExMTcxODIsMC41MzM1NjMyNTQ4MTc2NzE2LFByb3Zpc2lvbmFsIG1vZGVsLWJhc2VkIGludGVydmFsClVuaXRlZCBTdGF0ZXMsb2NjdXBhbmN5LDQsMC4wOTU0Njg5MjMxMDkzNjg3LC0wLjAwODc2ODA0NTA1Mjc3NywwLjE5OTcwNTg5MTI3MTUxNDUsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKVW5pdGVkIFN0YXRlcyxvY2N1cGFuY3ksNSwtMC4xMDkwNjQ4MDk4NTg2OTg4LC0wLjIxOTQxMDI3MTY5NDc0OCwwLjAwMTI4MDY1MTk3NzM1MDMsUHJvdmlzaW9uYWwgbW9kZWwtYmFzZWQgaW50ZXJ2YWwKVW5pdGVkIFN0YXRlcyxvY2N1cGFuY3ksNiwtMC4wNTQ5MTI5MDgxNTIzNCwtMC4xNDIyNTQzMTQ1Mjc3NDc3LDAuMDMyNDI4NDk4MjIzMDY3NixQcm92aXNpb25hbCBtb2RlbC1iYXNlZCBpbnRlcnZhbAo='


def restore_embedded_archive(path: Path):
    raw=base64.b64decode(ARCHIVED_REFERENCE_B64)
    if hashlib.sha256(raw).hexdigest()!=ARCHIVED_REFERENCE_SHA256:
        raise RuntimeError('Embedded historical reference checksum mismatch')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_bytes()!=raw:
        raise RuntimeError('Refusing to overwrite an inconsistent reference CSV: '+str(path))
    path.write_bytes(raw)
    data=pd.read_csv(path)
    req={'country','outcome','lag','effect','ci_lower','ci_upper'}
    if not req.issubset(data.columns):raise ValueError('Archived reference schema mismatch')
    if len(data)!=154 or data.duplicated(['country','outcome','lag']).any():
        raise ValueError('Archived reference must contain exactly 22 x 7 unique points')
    if set(data.country)!=set(COUNTRIES) or set(data.outcome)!=set(OUTCOMES):
        raise ValueError('Archived reference contains unexpected country/outcome')
    if any(set(g.lag)!={0,1,2,3,4,5,6} for _,g in data.groupby(['country','outcome'])):
        raise ValueError('Archived reference has missing lag positions')
    if not np.isfinite(data[['effect','ci_lower','ci_upper']].to_numpy(float)).all():
        raise ValueError('Nonfinite archived primary reference')
    return data


def load_us_weekly(root:Path, source_csv:str='', weekly_us:str=''):
    if weekly_us:
        w=pd.read_csv(weekly_us)
        if not {'week_sunday','x','y_admissions','common'}.issubset(w.columns):
            raise ValueError('Local US weekly file has invalid columns')
        w['week_sunday']=pd.to_datetime(w.week_sunday)
        w=w.set_index('week_sunday').sort_index()
        return w
    weekly=root/'data'/'weekly_united_states.csv'
    if not weekly.exists():
        src=Path(source_csv).expanduser().resolve() if source_csv else root/'data'/'owid_pinned.csv'
        if not source_csv:download_source(src)
        daily,_=load_daily(src)
        countries,audit,passed=build_countries(daily,root)
        if not passed: raise RuntimeError('Country-week inclusion audit failed; US audit halted')
        return countries['United States']
    w=pd.read_csv(weekly)
    w['week_sunday']=pd.to_datetime(w.week_sunday)
    return w.set_index('week_sunday').sort_index()


def us_reference_design(w):
    """Exactly the same country-specific trend window and Legendre lag basis."""
    spec=Specification('primary_unrestricted',6,'identity',7)
    ix=np.flatnonzero(w.common.to_numpy(bool))
    if len(ix)!=EXPECTED_BY_COUNTRY['United States']:
        raise ValueError('US observed sample differs from archived count of 147 weeks')
    full_x=w.x.to_numpy(float)
    Zfull,H,exposure,valid=make_crossbasis(full_x,spec)
    sl=slice(int(ix[0]),int(ix[-1])+1)
    obs=w.common.to_numpy(bool)[sl]
    y=w.y_admissions.to_numpy(float)[sl].copy(); y[~obs]=np.nan
    trend=identifiable_trend_columns(make_trend(len(y)),obs)
    X=np.column_stack([trend,Zfull[sl]])
    if np.linalg.matrix_rank(X[obs])!=X.shape[1]:
        raise RuntimeError('US primary design rank failure')
    if not bool(valid[sl][obs].all()):
        raise RuntimeError('Missing lag histories on US observed weeks')
    return y,X,obs,trend,H,exposure


def evaluate_us_fit(model, result, T, ntrend,H,exposure, archive):
    """All results expressed in original model coordinates, never QR columns."""
    k=T.shape[0]
    params=np.asarray(result.params,float)
    G=np.eye(len(params)); G[:k,:k]=T
    native=G@params
    V=np.asarray(result.cov_params(),float)
    V=G@V@G.T
    V,min_eig=safe_cov(V)
    point,se,cum,cumse=predict_from_fit(native,V,ntrend,H,exposure)
    original=archive.sort_values('lag').effect.to_numpy(float)
    native_llf=float(model.loglike(params))
    return dict(peak_lag=int(np.argmax(point)),cumulative_effect=float(cum),
        max_abs_archived_difference=float(np.max(np.abs(point-original))),
        archived_curve_rmse=float(np.sqrt(np.mean((point-original)**2))),
        robust_cov_min_eig=float(min_eig),
        effect_profile=point.tolist(),pointwise_se=se.tolist(),
        original_params=native.tolist(),llf_model_evaluation=native_llf)


def us_multistart_audit(w,archive:pd.DataFrame,root:Path,
                        n_starts:int=6,seed:int=20261025,
                        maxiter:int=1500):
    """Deterministic multistart diagnostics, with native and QR coordinates.

    Scientific guard: starts are built from the data-derived optimizer result,
    with fixed random perturbations of regression coefficients only. Neither
    archived lag effects nor their CIs enter the likelihood optimization.
    Compare the best candidate *by Gaussian log-likelihood*, not by reference
    agreement. A numerical-identity claim requires the post-fit archive audit.
    """
    if n_starts<2: raise ValueError('Use at least two independent perturbation starts')
    output=root/'tables'; output.mkdir(parents=True,exist_ok=True)
    target=archive[(archive.country=='United States') & (archive.outcome=='admissions')]
    if len(target)!=7:raise ValueError('Missing archived US admissions coefficients')
    y,X,mask,trend,H,exp=us_reference_design(w)
    native_model=SARIMAX(y,exog=X,order=(2,0,1),trend='n',
                         enforce_stationarity=True,enforce_invertibility=True)
    # Original optimization policy first (not counted as an independent random start).
    baseline_model,baseline_result,base_note=fit_archived_reference(y,X)
    pbase=np.asarray(baseline_result.params,float)
    k=X.shape[1]
    transformed_X,T,cond=orthogonalize_observed_exog(X,mask)
    qr_model=SARIMAX(y,exog=transformed_X,order=(2,0,1),trend='n',
                    enforce_stationarity=True,enforce_invertibility=True)
    # Draw perturbations in both coordinate systems from a frozen seed.
    rng=np.random.default_rng(seed)
    candidates=[]; successful=[]
    def record(label,coord,method,start,model,mat,limit):
        t0=time.perf_counter()
        meta=dict(run_id=label,coordinates=coord,optimizer=method,
                  maxiter=limit,start_seed=seed,design_condition=cond)
        try:
            kw=dict(method=method,maxiter=int(limit),disp=False,cov_type='robust',
                    start_params=np.asarray(start,float))
            with warnings.catch_warnings(record=True) as warn:
                warnings.simplefilter('always')
                fit=model.fit(**kw)
            meta.update(elapsed_seconds=float(time.perf_counter()-t0),
                converged=bool(fit.mle_retvals.get('converged',False)),
                log_likelihood=float(fit.llf),aic=float(fit.aic),bic=float(fit.bic),
                iterations=str(fit.mle_retvals.get('iterations','')),
                mle_warnflag=str(fit.mle_retvals.get('warnflag','')),
                warnings=' | '.join(str(x.message)[:110] for x in warn)[:350])
            if meta['converged'] and np.isfinite(fit.llf):
                diagnostic=evaluate_us_fit(model,fit,mat,trend.shape[1],H,exp,target)
                native_ll=float(native_model.loglike(diagnostic['original_params']))
                meta.update(peak_lag=diagnostic['peak_lag'],
                    cumulative_effect=diagnostic['cumulative_effect'],
                    max_abs_archived_difference=diagnostic['max_abs_archived_difference'],
                    archived_curve_rmse=diagnostic['archived_curve_rmse'],
                    log_likelihood_native_coordinates=native_ll,
                    parameterization_llf_gap=float(fit.llf-native_ll),
                    robust_cov_min_eig=diagnostic['robust_cov_min_eig'])
                successful.append((meta,diagnostic))
            else:
                meta['reason']='Optimizer did not converge to a finite solution'
        except Exception as exc:
            meta.update(converged=False,elapsed_seconds=float(time.perf_counter()-t0),
                        reason=f'{type(exc).__name__}: {str(exc)[:250]}')
        candidates.append(meta)
        LOGGER.info('US multistart: %s converged=%s llf=%s peak=%s',label,
                    meta.get('converged',False),meta.get('log_likelihood'),meta.get('peak_lag'))
        pd.DataFrame(candidates).to_csv(output/'20_us_multistart_candidates.csv',index=False)
        return meta

    # Preserve historical native baseline as a separate diagnostic row.
    baseline_diagnostic=evaluate_us_fit(native_model,baseline_result,np.eye(k),
                                        trend.shape[1],H,exp,target)
    baseline_row=dict(run_id='historical_policy_baseline',coordinates='native',
        optimizer='original_lbfgs_powell_policy',converged=True,
        log_likelihood=float(baseline_result.llf),aic=float(baseline_result.aic),
        bic=float(baseline_result.bic),design_condition=float(cond),
        max_abs_archived_difference=baseline_diagnostic['max_abs_archived_difference'],
        archived_curve_rmse=baseline_diagnostic['archived_curve_rmse'],
        peak_lag=baseline_diagnostic['peak_lag'],
        cumulative_effect=baseline_diagnostic['cumulative_effect'])
    candidates.append(baseline_row)
    successful.append((baseline_row,baseline_diagnostic))
    pd.DataFrame(candidates).to_csv(output/'20_us_multistart_candidates.csv',index=False)

    # Independent model-coordinate seeds, alternating native and QR.
    # One zero-perturbation start in QR checks reparameterization invariance.
    scales=[0.,.02,.06,.12,.025,.08,.15,.04]
    for idx in range(n_starts):
        coord='qr' if idx%2==0 else 'native'
        model=qr_model if coord=='qr' else native_model
        mat=T if coord=='qr' else np.eye(k)
        start=pbase.copy()
        if coord=='qr':
            start[:k]=np.linalg.solve(T,start[:k])
        s=scales[idx%len(scales)]
        if s:
            # Perturbation relative to the regression parameter scale;
            # ARMA and variance parameters are held feasible and unchanged.
            scale_vec=.05+np.abs(start[:k])
            start[:k] += rng.standard_normal(k)*scale_vec*s
        record(f'{coord}_start_{idx:02d}_scale_{s:g}',coord,'lbfgs',start,model,mat,maxiter)
    if not successful:raise RuntimeError('No converged US optimizer solution')
    bestmeta,best=max(successful,key=lambda x:x[0]['log_likelihood'])
    best_ll=float(bestmeta['log_likelihood'])
    df=pd.DataFrame(candidates)
    df['delta_loglik_from_best']=best_ll-pd.to_numeric(df.log_likelihood,errors='coerce')
    df['is_max_likelihood_fit']=df.run_id.eq(bestmeta['run_id'])
    df.to_csv(output/'20_us_multistart_candidates.csv',index=False)
    llfs=np.array([float(z['log_likelihood']) for z,_ in successful])
    stable=bool(np.max(llfs)-np.min(llfs)<=.01)
    summary={
        'pipeline':PIPELINE_VERSION,'n_random_starts':n_starts,
        'n_attempts_including_baseline':len(candidates),
        'n_converged':int(df.converged.fillna(False).sum()),
        'baseline_loglik':float(baseline_result.llf),
        'best_loglik':best_ll,'best_run_id':bestmeta['run_id'],
        'best_gain_vs_baseline':best_ll-float(baseline_result.llf),
        'max_converged_loglik_spread':float(np.ptp(llfs)),
        'all_converged_starts_near_equal_loglik':stable,
        'baseline_max_abs_archive_difference':baseline_diagnostic['max_abs_archived_difference'],
        'best_max_abs_archive_difference':best['max_abs_archived_difference'],
        'best_curve_rmse_archive':best['archived_curve_rmse'],
        'archive_peak_lag':int(target.loc[target.effect.idxmax(),'lag']),
        'best_peak_lag':best['peak_lag'],
        'archive_identity_at_tolerance_1e_minus_3':bool(best['max_abs_archived_difference']<=.001),
        'archive_was_not_used_to_choose_optimum':True,
        'interpretation_note':('Optimizer selected by maximum likelihood, not by closeness to archive. '
              'If independent starts disagree, inspect local optima; if all stabilize but archives differ, '
              'investigate software versions/initialization/covariance/design and input semantics.')}
    pd.DataFrame([summary]).to_csv(output/'21_us_multistart_summary.csv',index=False)
    fig,ax=plt.subplots(1,2,figsize=(10.6,3.9))
    gg=target.sort_values('lag')
    ax[0].plot(gg.lag,gg.effect,'o-',color=BLUE,label='Archived primary')
    ax[0].plot(range(7),best['effect_profile'],'s-',color=ORANGE,
               label='Best multistart refit')
    ax[0].axhline(0,lw=.6,color='.55');ax[0].legend()
    ax[0].set(xlabel='Lag (weeks)',ylabel='Standardized asinh contrast',title='US admissions: lag profile')
    conv=df[df.converged.eq(True)].copy()
    ax[1].scatter(range(len(conv)),conv.log_likelihood,color=BLUE)
    ax[1].axhline(best_ll,color=ORANGE,ls='--',lw=1,label='Best likelihood')
    ax[1].set(xlabel='Converged candidate index',ylabel='Log-likelihood',title='Multistart objective diagnostics')
    ax[1].legend();fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(root/'figures'/('Supplement_US_admissions_multistart.'+ext),
                    dpi=360,bbox_inches='tight')
    plt.close(fig)
    (root/'text'/'US_admissions_multistart_interpretation.txt').write_text(
        json.dumps(summary,indent=2)+ '\n\nDifferent likelihood maxima are a numerical issue, '
        'not evidence for different epidemiological processes. A stable numerical optimum '
        'need not replicate historical coefficients under a different software environment.\n',
        encoding='utf-8')
    return summary


def archive_anchored_reporting(root:Path,archive:pd.DataFrame,mode:str,
                              peak_tolerance:int=1,cumulative_tolerance:float=.1):
    """Replace the PUBLIC comparison with the archived fixed reference.

    Refit-primary comparisons are kept as *diagnostics* with explicit labels.
    Different maximum lags do NOT have a comparable cumulative estimand.
    """
    tab=root/'tables'
    lag=pd.read_csv(tab/'04_lag_response_pointwise.csv')
    stats=pd.read_csv(tab/'03_model_statistics.csv')
    ref=archive.rename(columns={'effect':'archive_effect',
                                'ci_lower':'archive_lower','ci_upper':'archive_upper'})
    diagnostic=lag[lag.specification=='primary_unrestricted']
    check=ref.merge(diagnostic[['country','outcome','lag','effect','se','ci_lower','ci_upper']],
                   on=['country','outcome','lag'],how='left',validate='one_to_one')
    check['difference']=check.effect-check.archive_effect
    check['abs_difference']=check.difference.abs()
    check.to_csv(tab/'14_primary_archived_coefficient_audit.csv',index=False)
    primary=[]
    for (country,outcome),g in check.groupby(['country','outcome']):
        g=g.sort_values('lag');o=g.archive_effect.to_numpy(float)
        z=g.effect.to_numpy(float)
        primary.append(dict(country=country,outcome=outcome,n_matched=int(np.isfinite(z).sum()),
            archive_peak=int(g.iloc[np.argmax(o)].lag),
            refit_peak=int(g.iloc[np.argmax(z)].lag) if np.isfinite(z).all() else np.nan,
            max_abs_effect_difference=float(np.nanmax(abs(z-o))),
            curve_rmse=float(np.sqrt(np.nanmean((z-o)**2)))))
    audit=pd.DataFrame(primary)
    audit.to_csv(tab/'15_primary_archived_reconciliation_summary.csv',index=False)
    rows=[]
    for (country,outcome),rg in ref.groupby(['country','outcome']):
        rg=rg.sort_values('lag');base=rg.archive_effect.to_numpy(float)
        for specname,alt in lag[(lag.country==country)&(lag.outcome==outcome)&
                               (lag.specification!='primary_unrestricted')].groupby('specification'):
            alt=alt.sort_values('lag')
            if len(alt)!=7 or set(alt.lag)!={0,1,2,3,4,5,6}:
                continue  # horizon 4/8 is separate, not an equivalent 7-week integral.
            eff=alt.effect.to_numpy(float)
            corr=float(np.corrcoef(base,eff)[0,1]) if np.std(base)>1e-10 and np.std(eff)>1e-10 else np.nan
            drift=int(alt.iloc[np.argmax(eff)].lag-rg.iloc[np.argmax(base)].lag)
            delta=float(eff.sum()-base.sum())
            rows.append(dict(country=country,outcome=outcome,
                specification=specname,comparison_reference='archived_original',
                archive_peak=int(rg.iloc[np.argmax(base)].lag),
                alternative_peak=int(alt.iloc[np.argmax(eff)].lag),
                peak_lag_difference=drift,within_one_week=abs(drift)<=1,
                archive_cumulative=float(base.sum()),alternative_cumulative=float(eff.sum()),
                cumulative_difference=delta,
                same_cumulative_sign=bool(np.sign(base.sum())==np.sign(eff.sum())),
                lag_profile_correlation=corr,
                lag_profile_rmse=float(np.sqrt(np.mean((base-eff)**2))),
                descriptive_flag=bool(abs(drift)>peak_tolerance or abs(delta)>cumulative_tolerance),
                descriptive_threshold='peak >%d weeks OR |delta cumulative|>%.3f; not statistical significance'%(peak_tolerance,cumulative_tolerance)))
    out=pd.DataFrame(rows)
    if out.empty:raise RuntimeError('No archived-anchored same-horizon comparisons')
    out.to_csv(tab/'16_archived_primary_vs_DLNM.csv',index=False)
    summ=[]
    for sp,g in out.groupby('specification'):
        summ.append(dict(specification=sp,n=len(g),exact_peak=int(g.peak_lag_difference.eq(0).sum()),
            within_one_week=int(g.within_one_week.sum()),same_cumulative_sign=int(g.same_cumulative_sign.sum()),
            median_correlation=float(g.lag_profile_correlation.median()),
            median_abs_cumulative_difference=float(g.cumulative_difference.abs().median()),
            flagged_descriptively=int(g.descriptive_flag.sum())))
    summary=pd.DataFrame(summ)
    summary.to_csv(tab/'17_archived_anchored_summary.csv',index=False)
    # Preserve the original generic script's refitted-primary reports for audit.
    for suffix in ('06_main_agreement_country_outcome.csv','07_reviewer_sensitivity_summary.csv'):
        f=tab/suffix
        if f.exists():
            f.replace(tab/(suffix.replace('.csv','_REFIT_DIAGNOSTIC.csv')))
    main=out[out.specification.eq('dlnm_linear_cubic_df4')]
    if len(main)!=22:
        raise RuntimeError('Incomplete matched archived comparison: %s/22'%len(main))
    main.to_csv(tab/'06_main_agreement_country_outcome.csv',index=False)
    summary.to_csv(tab/'07_reviewer_sensitivity_summary.csv',index=False)
    # AIC/BIC are meaningful ONLY within series, matching samples, and
    # unpenalized likelihood models; archive lacks archived likelihood.
    stats.query('specification in ["dlnm_linear_cubic_df4", "dlnm_nonlinear_natural_df4"]')\
         .to_csv(tab/'18_within_DLNM_information_criteria.csv',index=False)
    total=22*len(specifications(mode))
    completed=int(stats.optimizer_converged.fillna(False).astype(bool).sum())
    requested_keys={(c,o,s.name) for c in COUNTRIES for o in OUTCOMES for s in specifications(mode)}
    actual_keys={(str(row.country),str(row.outcome),str(row.specification)) for row in stats.itertuples()}
    gate=dict(requested=total,observed_fit_rows=len(stats),converged=completed,
              missing_keys=len(requested_keys-actual_keys),unexpected_keys=len(actual_keys-requested_keys),
              duplicate_model_keys=len(stats)-len(actual_keys))
    pd.DataFrame([gate]).to_csv(tab/'19_full_completion_gate.csv',index=False)
    if (len(stats)!=total or completed!=total or actual_keys!=requested_keys):
        raise RuntimeError('Refusing publication-facing completeness claim: '+str(gate))
    comp=dict(mode=mode,requested=total,completed=completed,
              archive_rows=len(ref),archived_matched_lags=int(np.isfinite(check.effect).sum()),
              main_matched_country_outcomes=len(main),
              exact_peak_agreement=int(main.peak_lag_difference.eq(0).sum()),
              within_one_week=int(main.within_one_week.sum()),
              same_cumulative_sign=int(main.same_cumulative_sign.sum()),
              median_profile_correlation=float(main.lag_profile_correlation.median()),
              flagged_main=int(main.descriptive_flag.sum()),
              reference_reproduced_to_1e_minus_3=bool(audit.max_abs_effect_difference.max()<=.001),
              max_primary_refit_archive_difference=float(audit.max_abs_effect_difference.max()),
              archived_reference_sha256=ARCHIVED_REFERENCE_SHA256)
    (root/'text'/'OFFICIAL_ARCHIVED_REFERENCE_SUMMARY.json').write_text(json.dumps(comp,indent=2),encoding='utf-8')
    textlines=[
        'Post-hoc empirical DLNM sensitivity: archived original unpenalized baseline',
        'The 154 reference coefficients are fixed, checksum-verified, and not used to choose an optimizer.',
        'Matched 0–6-week linear-exposure DLNM comparison across %s of 22 country–outcome series.'%len(main),
        'Exact peak agreement: %s/22; agreement within one week: %s/22.'%(comp['exact_peak_agreement'],comp['within_one_week']),
        'Cumulative association sign agreement: %s/22.'%comp['same_cumulative_sign'],
        'Median lag correlation: %.4f.'%comp['median_profile_correlation'],
        'Descriptively flagged magnitude/peak differences: %s/22.'%comp['flagged_main'],
        'Model-based confidence limits are not validated simultaneous or calibrated intervals.',
        'A retrospective conditional association is not a causal effect, patient delay, or demonstrated forecast.',
        'The original primary refit is an independent numerical diagnostic, NOT the reference in these summaries.',
        'The maximum archived-versus-refitted coefficient difference was %.5f; exact identity cannot be claimed unless this is reduced.'%comp['max_primary_refit_archive_difference'],
        'Lag horizons 4 and 8 are separate cumulative estimands and not compared on equal cumulative support.'
    ]
    (root/'text'/'OFFICIAL_reviewer_methods_results_notes.txt').write_text('\n'.join(textlines)+'\n',encoding='utf-8')
    LOGGER.info('OFFICIAL archived reference: %s matched, %s within 1-week, %s sign agreement; refit max discrepancy=%.4g',
                len(main),comp['within_one_week'],comp['same_cumulative_sign'],comp['max_primary_refit_archive_difference'])
    return comp


def archived_overlay(root:Path,archive:pd.DataFrame):
    lag=pd.read_csv(root/'tables'/'04_lag_response_pointwise.csv')
    fig,axs=plt.subplots(2,2,figsize=(10.5,7),sharex=True)
    for ax,(country,outcome) in zip(axs.flatten(),[('United States','admissions'),('United States','occupancy'),
                                ('Israel','occupancy'),('Belgium','occupancy')]):
        base=archive[(archive.country==country)&(archive.outcome==outcome)].sort_values('lag')
        alt=lag[(lag.country==country)&(lag.outcome==outcome)&
                (lag.specification=='dlnm_linear_cubic_df4')].sort_values('lag')
        ax.plot(base.lag,base.effect,'o-',color=BLUE,label='Archived primary')
        ax.fill_between(base.lag.to_numpy(float),base.ci_lower.to_numpy(float),
                        base.ci_upper.to_numpy(float),color=BLUE,alpha=.12)
        ax.plot(alt.lag,alt.effect,'s-',color=ORANGE,label='Linear DLNM')
        ax.fill_between(alt.lag.to_numpy(float),alt.ci_lower.to_numpy(float),
                        alt.ci_upper.to_numpy(float),color=ORANGE,alpha=.12)
        ax.axhline(0,lw=.6,color='.6')
        ax.set(title=country+' - '+outcome,xlabel='Lag (weeks)',
               ylabel='Standardized asinh contrast')
    axs[0,0].legend(fontsize=8)
    fig.suptitle('Original archived coefficients versus fitted linear DLNM',fontsize=12)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(root/'figures'/('Supplement_OFFICIAL_archive_anchored_comparison.'+ext),
                    dpi=360,bbox_inches='tight')
    plt.close(fig)


def advanced_pipeline(output_dir:str,mode='full',data_csv='',weekly_us='',
                      n_starts=6,random_seed=20261025,audit_maxiter=1500,
                      strict=True,force=False,run_models=True,run_audit=True,
                      peak_tolerance=1,cumulative_tolerance=.1):
    root=Path(output_dir).expanduser().resolve()
    make_dirs(root);setup_logging(root)
    LOGGER.info('Starting %s (mode=%s)',PIPELINE_VERSION,mode)
    ref_file=root/'data'/'original_archived_primary_154_coefficients.csv'
    archived=restore_embedded_archive(ref_file)
    if run_audit:
        us=load_us_weekly(root,source_csv=data_csv,weekly_us=weekly_us)
        us_multistart_audit(us,archived,root,n_starts=n_starts,seed=random_seed,maxiter=audit_maxiter)
    if run_models:
        args=SimpleNamespace(output=str(root),data=data_csv,mode=mode,
            country=None,featured_country='United States',archived_primary=str(ref_file),
            expected_sha256='',maxiter=130,peak_tolerance=peak_tolerance,
            cumulative_tolerance=cumulative_tolerance,strict=strict,force=force,self_test=False)
        run(args)
    required=root/'tables'/'03_model_statistics.csv'
    if not run_models:
        LOGGER.info('Multistart-only audit complete; full model fits were not requested.')
        return None
    if not required.exists():
        raise RuntimeError('The requested full/quick modelling run did not produce model statistics')
    report=archive_anchored_reporting(root,archived,mode,peak_tolerance,cumulative_tolerance)
    archived_overlay(root,archived)
    if run_audit and (root/'tables'/'21_us_multistart_summary.csv').exists():
        report['us_multistart_audit']=pd.read_csv(root/'tables'/'21_us_multistart_summary.csv').iloc[0].to_dict()
    (root/'text'/'PIPELINE_status.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    zip_results(root)
    LOGGER.info('Pipeline archived-reference reconciliation and packaging complete')
    return report


def advanced_cli():
    ap=argparse.ArgumentParser(description='Paper 2 multistart US audit plus full DLNM and archived reference reports')
    ap.add_argument('--output',default='/content/Paper2_DLNM_reference_audit' if Path('/content').exists() else './Paper2_DLNM_reference_audit')
    ap.add_argument('--data',default='',help='Optional pinned OWID daily CSV; downloaded otherwise')
    ap.add_argument('--weekly-us',default='',help='Optional locally cached US weekly CSV for audit-only tests')
    ap.add_argument('--mode',choices=['quick','full'],default='full')
    ap.add_argument('--n-starts',type=int,default=6)
    ap.add_argument('--random-seed',type=int,default=20261025)
    ap.add_argument('--audit-maxiter',type=int,default=1500)
    ap.add_argument('--audit-only',action='store_true')
    ap.add_argument('--skip-us-audit',action='store_true')
    ap.add_argument('--force',action='store_true')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:
        self_test()
        restore_embedded_archive(Path(args.output)/'data'/'original_archived_primary_154_coefficients.csv')
        print('ARCHIVED-REFERENCE SELF-TEST PASS')
    else:
        advanced_pipeline(output_dir=args.output,mode=args.mode,data_csv=args.data,
            weekly_us=args.weekly_us,n_starts=args.n_starts,random_seed=args.random_seed,
            audit_maxiter=args.audit_maxiter,run_audit=not args.skip_us_audit,
            run_models=not args.audit_only,force=args.force)

if __name__=='__main__':
    advanced_cli()
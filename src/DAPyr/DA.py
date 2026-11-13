import numpy as np
import copy
from . import MISC
from . import INFLATION
import warnings
import functools

#TODO Clean up and comment inside functions

def EnSRF_update(xf : np.ndarray, hx : np.ndarray, 
                 y : np.ndarray, HC : np.ndarray, HCH: np.ndarray,
                 var_y : float, inf_flag:int,
                 e_flag : int, qc : np.ndarray, **kwargs):
    '''Performs an Ensemble Square Root Filter update based on Whitaker and Hamill (2002).
    
    Parameters
    ----------
    xf : np.ndarray
        Array of size Nx x Ne containing the ensemble members
    hx : np.ndarray
        Array of size Ny x Ne containing the ensemble members projected into obs-space
    y : np.ndarray
        Array of size Ny x 1 containing the observations at time T
    HC : np.ndarray
        Localization matrix of size Ny x Nx in state-space
    HCH : np.ndarray
        Localization matrix of size Ny x Ny in obs-space
    var_y : float
        Observation variance
    inf_flag : int
        Flag to choose what inflation scheme to choose
    e_flag : int
        Error flag
    qc : np.ndarray
        Array of length Ny repesenting quality control of observations

    Returns
    --------
    xa : np.ndarray
        Array of size Nx x Ne representing analysis ensemble states
    e_flag : int
        Error flag after data assimilation step

    Other Parameters
    -----------------
    **kwargs
        Extra arguments to `EnSRF_update` to pass to inflation parameters. See documentation for `EnSRF_update` for a list of all possible arguments.
    '''
        
    if len(y.shape) == 1: #If Y is 1-dimensional, make it the correct dimension
        y = y[:, None]

    #Ensemble mean
    Ny = len(y[:, 0])
    Nx, Ne = xf.shape

    if inf_flag == 3: #Anderson Inflation
        infs, infs_y = kwargs['infs'], kwargs['infs_y']
        var_infs, var_infs_y = kwargs['var_infs'], kwargs['var_infs_y']
        xf, infs, var_infs = INFLATION.do_Anderson2009(xf, hx, y, infs, var_infs, HC, var_y)
        hx, infs_y, var_infs_y = INFLATION.do_Anderson2009(hx, hx, y, infs_y, var_infs_y, HCH, var_y)
    
    xm = np.mean(xf, axis = -1)[:,None]
    hxm = np.mean(hx, axis = -1)[:, None]

    xp = xf - xm #Nx x Ne
    xpo = copy.deepcopy(xp) #Original perturbation
    hxp = hx - hxm # Ny x Ne
    #one obs at a time
    if e_flag !=0:
        return np.nan, e_flag
    if np.sum(qc) == Ny:
        warnings.warn('No observations pass QAQC, error_flag set to 1.')
        e_flag = 1
        return np.nan, e_flag

    for i in range(Ny):
        d = (y[i, :] - hxm[i, :])
        hxo = hxp[i, :]
        var_den = np.dot(hxo, hxo)/(Ne-1) + var_y
        P = np.dot(xp, hxo)/(Ne - 1)
        P = P*HC[i, :]
        K = P/var_den
        xm = xm + K[:, np.newaxis]*d[:, np.newaxis]

        beta = 1/(1 + np.sqrt(var_y/var_den))
        xp = xp - beta*np.dot(K[:, np.newaxis], hxo[np.newaxis, :])

        P = np.dot(hxp, hxo)/(Ne - 1)
        P = P*HCH[i, :]
        K = P/var_den

        hxm = hxm + K[:, np.newaxis]*d[:, np.newaxis]
        beta = 1/(1 + np.sqrt(var_y/var_den))
        hxp = hxp - beta*np.dot(K[:, np.newaxis], hxo[np.newaxis, :])

    match inf_flag:
        case 0: #Nothing
            xa_new = xm + xp # Do Nothing
        case 1: #RTPS
            gamma = kwargs['gamma']
            xa_new = INFLATION.do_RTPS(xf, xm + xp, gamma)
        case 2: #RTPP
            gamma = kwargs['gamma']
            xa_new = INFLATION.do_RTPP(xf, xm + xp, gamma)
        case _: #Anything Else
            xa_new = xm + xp # Do Nothing
    if inf_flag == 3:
        return xa_new, infs, infs_y, var_infs, var_infs_y, e_flag
    else:
        return xa_new, e_flag


def lpf_update(x : np.ndarray, hx : np.ndarray,
               Y : np.ndarray,
               H : np.ndarray, C_pf : np.ndarray,
               N_eff : float,
               min_res : float, maxiter : int,
               kddm_flag : int,
               e_flag : int, qcpass : np.ndarray,
               L : functools.partial,
               n_moments : int = 4, merge_flag : int = 0, qm_moments : int = 2,
               resample_neff : float = 0.98, min_weight : float = 1e-15,
               rng : np.random.Generator = None,
               cache_elems : int = 20_000_000):
    '''Performs a Local Particle Filter update (Poterjoy 2016, 2022), following
    the MATLAB LPF (Oct 2026) that the DART LPF is synchronized with.

    Observations are assimilated one at a time. Every state variable and every
    observation prior carries localized log-weights accumulated over the
    observations processed so far. These are tempered with regularization
    coefficients beta, chosen per variable so that the tempered weights keep an
    effective ensemble size of N_eff. For each observation, particles are drawn by
    systematic resampling from its tempered weights and merged with the current
    particles using the tempered localization beta*rho; the merge matches the
    first four moments of the weighted marginal (Cornish-Fisher), or the whole
    marginal (rank quantile mapping). Model- and observation-space particles are
    updated by the same steps. The remaining residual 1 - sum(beta) is
    assimilated in further passes (tempering) until it is used up or maxiter
    passes have run.

    Parameters
    ----------
    x : np.ndarray
        Array of size Nx x Ne containing the ensemble members
    hx : np.ndarray
        Array of size Ny x Ne containing the ensemble members projected into obs-space
    Y : np.ndarray
        Array of size Ny x 1 containing the observations at time T
    H : np.ndarray
        Array of size Ny x Nx representing the measurement operator
    C_pf : np.ndarray
        Localization matrix of size Ny x Nx in state-space
    N_eff : float
        Target effective ensemble size (number of particles, not a fraction)
    min_res : float
        Fraction of each observation's information that is not assimilated
        (residual left after tempering; 0 assimilates everything)
    maxiter : int
        Maximum number of tempering passes (1 = single pass)
    kddm_flag : int
        1 to map particles with kernel density distribution mapping after each pass
    e_flag : int
        Error flag
    qcpass : np.ndarray
        Array of length Ny repesenting quality control of observations (1 = rejected)
    L : functools.partial
        Callable likelihood function L(y, hx). The first argument is a vector of scalar observations, and the second argument is the model state projected into observation space.
        If L has an attribute neg_log (as the likelihoods from OBS_ERRORS.get_likelihood do), L.neg_log(y, hx)
        is used for the negative log-likelihood, which avoids underflow for particles far from the observation.
    n_moments : int, optional
        Moments fitted by the Cornish-Fisher merge: 2 (mean, variance), 3 (+ skewness)
        or 4 (+ kurtosis). Default 4.
    merge_flag : int, optional
        0 = moment-matching (Cornish-Fisher) merge, 1 = rank quantile-mapping merge. Default 0.
    qm_moments : int, optional
        With merge_flag = 1, adjustment after quantile mapping: 2 = exact posterior mean
        and variance, 1 = mean only, 0 = none. Default 2.
    resample_neff : float, optional
        An observation is resampled and merged only when the effective ensemble size of
        its tempered weights is below resample_neff*Ne; otherwise nearby variables only
        accumulate its weights. Default 0.98; 1.0 merges unless weights are uniform.
    min_weight : float, optional
        Weight floor: each observation's likelihood is tempered so that no normalized
        weight falls below min_weight/Ne. 0 turns it off. Default 1e-15.
    rng : np.random.Generator, optional
        Random number generator for resampling. Default: a new unseeded generator.
    cache_elems : int, optional
        The localized log-weight terms of all observations (Ny x (Nx+Ny) x Ne values)
        are kept between the regularization and update passes when they number at
        most cache_elems (default 2e7, 160 MB); otherwise they are recomputed.
        Results are the same either way.

    Returns
    --------
    xa : np.ndarray
        Array of size Nx x Ne representing analysis ensemble states
    e_flag : int
        Error flag after data assimilation step
    '''
    if e_flag != 0:
        return np.nan, e_flag

    if np.sum(qcpass) == len(Y):
        warnings.warn('No observations pass QAQC, error_flag set to 1.')
        e_flag = 1
        return np.nan, e_flag

    if np.any(np.isnan(x)):
        e_flag = 1
        return np.nan, e_flag

    if rng is None:
        rng = np.random.default_rng()

    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]
    hx = np.atleast_2d(hx)

    Nx, Ne = x.shape
    HCH = np.matmul(C_pf, H.T)

    keep = (qcpass == 0)
    Y = Y[keep, :]
    hx = hx[keep, :]
    HC = C_pf[keep, :]
    HCH = HCH[keep, :][:, keep]
    Ny = len(Y)

    # Stack model- and obs-space particles; rows ix are x, rows iy are hx
    ix = np.arange(Nx)
    iy = np.arange(Nx, Nx + Ny)
    z = np.vstack([x, hx]).astype(float)
    C = np.hstack([HC, HCH])  # localization between each obs and each row of z
    Nv = Nx + Ny

    epsilon = 1e-30  # added inside the logs of the localized weights
    res = np.ones(Nv) - min_res

    max_res = 1.0
    niter = 0
    while max_res > 0 and min_res < 1:
        niter += 1
        zo = z.copy()

        # Obs-space likelihood weights, tempered so that no normalized weight
        # falls below min_weight/Ne
        if hasattr(L, 'neg_log'):
            d = L.neg_log(Y, zo[iy, :])
        else:
            d = -np.log(np.maximum(L(Y, zo[iy, :]), np.finfo(float).tiny))
        d = d - np.min(d, axis=1, keepdims=True)
        wo = np.exp(-d)
        if min_weight > 0:
            # floor check for all obs at once; bisection only where it is needed
            ws = np.sum(wo, axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                ok = (ws > 0) & (np.min(wo, axis=1) / ws > min_weight / Ne)
            for i in np.where(~ok)[0]:
                wo[i] = np.exp(-d[i] * MISC.find_obs_infl(d[i], min_weight / Ne))
        wo = wo / np.sum(wo, axis=1, keepdims=True)
        if np.any(np.isnan(wo)):
            e_flag = 1
            return np.nan, e_flag

        # Localized log-weight terms log((Ne*w - 1)*rho + 1) for each obs; kept
        # from the regularization pass for the update pass when they fit in memory
        def logw_term(i):
            return np.log(np.outer(C[i], Ne * wo[i] - 1) + 1 + epsilon)
        cache = [] if Ny * Nv * Ne <= cache_elems else None

        # Regularization coefficients from the log-weights of all observations
        lw = np.zeros((Nv, Ne))
        for i in range(Ny):
            term = logw_term(i)
            if cache is not None:
                cache.append(term)
            lw -= term
            lw -= np.min(lw, axis=1, keepdims=True)
        beta, res = MISC.get_reg(lw, N_eff, res)

        # Update pass: tempered weights and posterior moments after each observation
        # depend only on the particles at the start of the pass (zo)
        lw = np.zeros((Nv, Ne))
        for i in range(Ny):
            lw -= cache[i] if cache is not None else logw_term(i)
            lw -= np.min(lw, axis=1, keepdims=True)
            omega = np.exp(-lw * beta[:, None])
            omega = omega / np.sum(omega, axis=1, keepdims=True)
            zm = np.sum(omega * zo, axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                zv = np.sum(omega * (zo - zm[:, None])**2, axis=1) / (1 - np.sum(omega**2, axis=1))
            zv = np.where(np.isfinite(zv), zv, 0.0)

            if np.any(np.isnan(zm[ix])):
                e_flag = 1
                return np.nan, e_flag

            # Skip resampling and merging if few particles would be removed
            w = omega[iy[i]]
            if 1 > resample_neff * Ne * np.sum(w**2):
                continue

            ind = MISC.systematic_sampling(w, rng)
            loc = C[i] * beta
            if merge_flag == 1:
                z = MISC.pf_merge_qm(z, zo[:, ind], loc, zm, zv, omega, zo, qm_moments)
            else:
                z = MISC.pf_merge_cf(z, zo[:, ind], loc, zm, zv, omega, zo, n_moments)

        # KDDM with the weights after the last observation
        if kddm_flag == 1 and Ny > 0:
            z = MISC.kddm_rows(z, zo, omega)

        max_res = np.max(res[iy]) if Ny > 0 else 0.0
        if niter == maxiter:
            break

    xa = z[ix, :]
    if np.any(~np.isfinite(xa)):
        e_flag = 1
        return np.nan, e_flag
    return xa, e_flag

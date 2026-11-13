import numpy as np
import warnings

def calc_SV(xa, xf):
      Nx, Ne = xa.shape
      sqrt_Ne = np.sqrt(1/(Ne - 1))
      xma = np.mean(xa, axis = 1)
      xmf = np.mean(xf, axis = 1)
      xpa, xpf = sqrt_Ne*(xa - xma[:, np.newaxis]), sqrt_Ne*(xf - xmf[:, np.newaxis])
      vals, vecs = np.linalg.eigh(np.matmul(xpf.T, xpf))
      val_sort = vals.argsort()[::-1]
      vals = vals[val_sort]
      vecs = vecs[:, val_sort]
      SVe = np.matmul(xpf, vecs)
      SVi = np.matmul(xpa, vecs)
      energy = np.sum(SVe*SVe, axis = 0)/np.sum(SVi*SVi, axis = 0)
      return SVi, SVe, energy, vals


def create_periodic(sigma, m, dx):
      if m % 2 == 0: #Even
            cx = m/2
            x = np.concatenate([np.arange(0, cx), np.arange(cx, 0, -1), np.arange(0, cx), np.arange(cx, 0, -1)])
      else: #Odd
            cx = np.floor(m/2)
            x = np.concatenate([np.arange(0, cx+1), np.arange(cx, 0, -1), np.arange(0, cx+1), np.arange(cx, 0, -1)])
      wlc = np.exp(-((dx*(x))**2)/(2*sigma*2))
      B = np.zeros((m, m))
      for i in range(m):
            B[i, :] = wlc[m - i:2*m - i]
      B = np.where(B < 0, 0, B)
      return B


def gaussian_L(x, y, r):
    return np.exp(-(y - x)**2 / (2 * r)).item()


# ---------------------------------------------------------------------------
# Local particle filter helpers (ported from the MATLAB LPF, Oct 2026)
# ---------------------------------------------------------------------------

def _neff_at(c, beta):
    '''Effective ensemble size of the weights exp(-beta*c); 0 if they all underflow.'''
    w = np.exp(-c * beta)
    sw = np.sum(w)
    if sw > 0:
        w = w / sw
        return 1.0 / np.sum(w * w)
    return 0.0


def find_beta(c, Neff):
    '''Tempering coefficient beta in [0, 1] for which the weights exp(-beta*c)
    have an effective ensemble size of Neff.

    Newton's method on Neff(beta) - Neff, safeguarded by a bisection bracket.
    Returns 1 if the untempered weights already meet the target, and 0 (with a
    warning) if the search misses the target.

    Parameters
    ----------
    c : np.ndarray
        Negative log-weights (length Ne)
    Neff : float
        Target effective ensemble size

    Returns
    -------
    beta : float
    '''
    Ne = len(c)
    tol = 1e-8

    Neff_init = _neff_at(c, 1.0)
    if Neff_init >= Neff and np.sum(np.exp(-c)) > 0:
        return 1.0

    # Bracket [ks, ke], initial guess interpolated between Neff(0) = Ne and Neff(1)
    ks, ke = 0.0, 1.0
    beta = 0.5
    if Ne > Neff_init:
        guess = (Ne - Neff) / (Ne - Neff_init)
        if ks < guess < ke:
            beta = guess

    for _ in range(20):
        w = np.exp(-c * beta)
        sw = np.sum(w)
        if sw == 0:
            beta = 0.5 * (ks + ke)  # extreme underflow
            continue
        w = w / sw
        w_sq = w * w
        S2 = np.sum(w_sq)
        Nf = 1.0 / S2
        f = Nf - Neff

        if abs(ke - ks) < tol:
            break

        if f > 0:
            ks = beta  # Neff above target: beta can grow
        else:
            ke = beta  # Neff below target: beta must shrink

        # Newton step with dNeff/dbeta = 2*Nf^2*(sum(w^2 c) - sum(w c)*S2)
        dN = 2.0 * Nf * Nf * (np.sum(w_sq * c) - np.sum(w * c) * S2)
        beta_new = beta - f / dN if dN < 0 else -1.0
        if beta_new <= ks or beta_new >= ke:
            beta = 0.5 * (ks + ke)
        else:
            beta = beta_new

    Nf = _neff_at(c, beta)
    if Nf <= Neff - 1 or np.isnan(Nf):
        warnings.warn(f'find_beta: Neff is {Nf} but target is {Neff}; beta set to 0')
        beta = 0.0
    return beta


def get_reg(lw, Neff, res):
    '''Regularization (tempering) coefficients for the localized weights.

    For each variable j, beta[j] gives the tempered weights exp(-beta[j]*lw[j, :])
    an effective ensemble size of Neff. beta is capped by the remaining residual,
    which is reduced by the amount used. Variables with no residual left get beta = 0.

    Parameters
    ----------
    lw : np.ndarray
        Localized negative log-weights from all observations (Nv x Ne)
    Neff : float
        Target effective ensemble size
    res : np.ndarray
        Remaining residual for each variable (Nv,); updated in place

    Returns
    -------
    beta : np.ndarray (Nv,)
    res : np.ndarray (Nv,)
    '''
    beta = np.zeros(lw.shape[0])
    act = np.where(res > 0)[0]
    if len(act) == 0:
        return beta, res
    b = find_beta_rows(lw[act], Neff)
    r = res[act]
    cap = r < b
    beta[act] = np.where(cap, r, b)
    res[act] = np.where(cap, 0.0, r - b)
    return beta, res


def find_beta_rows(c, Neff):
    '''find_beta applied to every row of c at once (same safeguarded Newton
    iteration, carried out for all rows together).

    Parameters
    ----------
    c : np.ndarray
        Negative log-weights (Nr x Ne)
    Neff : float
        Target effective ensemble size

    Returns
    -------
    beta : np.ndarray (Nr,)
    '''
    Nr, Ne = c.shape
    tol = 1e-8

    def neff_at(cc, bb):
        w = np.exp(-cc * bb[:, None])
        sw = np.sum(w, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            w = w / sw[:, None]
            nf = 1.0 / np.sum(w * w, axis=1)
        return np.where(sw > 0, nf, 0.0), sw

    beta = np.ones(Nr)
    Neff_init, sw1 = neff_at(c, np.ones(Nr))
    todo = ~((Neff_init >= Neff) & (sw1 > 0))
    idx = np.where(todo)[0]
    if len(idx) == 0:
        return beta

    cc = c[idx]
    ni = Neff_init[idx]
    ks = np.zeros(len(idx))
    ke = np.ones(len(idx))
    bt = np.full(len(idx), 0.5)
    with np.errstate(divide='ignore', invalid='ignore'):
        guess = (Ne - Neff) / (Ne - ni)
    ok = (Ne > ni) & (guess > ks) & (guess < ke)
    bt[ok] = guess[ok]

    live = np.ones(len(idx), dtype=bool)
    for _ in range(20):
        r = np.where(live)[0]
        if len(r) == 0:
            break
        cr = cc[r]
        w = np.exp(-cr * bt[r, None])
        sw = np.sum(w, axis=1)
        under = sw == 0
        if np.any(under):  # extreme underflow: bisect
            ru = r[under]
            bt[ru] = 0.5 * (ks[ru] + ke[ru])
            keep = ~under
            r, cr, w, sw = r[keep], cr[keep], w[keep], sw[keep]
            if len(r) == 0:
                continue
        w = w / sw[:, None]
        w_sq = w * w
        S2 = np.sum(w_sq, axis=1)
        Nf = 1.0 / S2
        f = Nf - Neff

        done = np.abs(ke[r] - ks[r]) < tol
        live[r[done]] = False
        go = ~done
        r, cr, w, w_sq, S2, Nf, f = r[go], cr[go], w[go], w_sq[go], S2[go], Nf[go], f[go]
        if len(r) == 0:
            continue

        pos = f > 0
        ks[r[pos]] = bt[r[pos]]
        ke[r[~pos]] = bt[r[~pos]]

        dN = 2.0 * Nf * Nf * (np.sum(w_sq * cr, axis=1) - np.sum(w * cr, axis=1) * S2)
        with np.errstate(divide='ignore', invalid='ignore'):
            bnew = np.where(dN < 0, bt[r] - f / dN, -1.0)
        bis = (bnew <= ks[r]) | (bnew >= ke[r])
        bt[r] = np.where(bis, 0.5 * (ks[r] + ke[r]), bnew)

    Nf, _ = neff_at(cc, bt)
    bad = (Nf <= Neff - 1) | np.isnan(Nf)
    if np.any(bad):
        warnings.warn(f'find_beta: {np.sum(bad)} rows missed the Neff target {Neff}; beta set to 0')
        bt[bad] = 0.0
    beta[idx] = bt
    return beta


def find_obs_infl(d, minwt):
    '''Tempering factor in [0, 1] so that the smallest normalized weight of
    exp(-k*d) is not below minwt (bisection). Returns 1 if already satisfied.

    Parameters
    ----------
    d : np.ndarray
        Negative log-likelihoods of the particles (length Ne)
    minwt : float
        Minimum allowed normalized weight
    '''
    def min_weight(k):
        w = np.exp(-d * k)
        return np.min(w / np.sum(w))

    w = np.exp(-d)
    ws = np.sum(w)
    if ws > 0 and np.min(w / ws) > minwt:
        return 1.0

    ks, ke = 0.0, 1.0
    fks, fke = min_weight(ks) - minwt, min_weight(ke) - minwt
    km = 0.5
    for _ in range(1000):
        km = 0.5 * (ks + ke)
        fkm = min_weight(km) - minwt
        if abs(fke - fks) < minwt:
            break
        if fkm * fks > 0:
            ks, fks = km, fkm
        else:
            ke, fke = km, fkm
    return km


def systematic_sampling(w, rng):
    '''Systematic resampling that keeps surviving particles in place.

    Draws Ne indices by systematic resampling (one uniform offset, then Ne
    equally spaced points on the cumulative weights). Each particle drawn at
    least once keeps its own slot; the extra copies are shuffled into the slots
    of particles that were not drawn, which avoids sequential index bias.

    Parameters
    ----------
    w : np.ndarray
        Normalized particle weights (length Ne)
    rng : np.random.Generator

    Returns
    -------
    ind : np.ndarray of int (length Ne)
    '''
    Ne = len(w)
    cw = np.cumsum(w)
    cw[-1] = 1.01  # every point lands in a bin
    u = (rng.random() + np.arange(Ne)) / Ne
    draw = np.searchsorted(cw, u, side='right')  # = number of cw <= u (0-based index)

    seen = np.zeros(Ne, dtype=bool)
    first = np.zeros(Ne, dtype=bool)
    for k in range(Ne):
        if not seen[draw[k]]:
            seen[draw[k]] = True
            first[k] = True
    ind = np.zeros(Ne, dtype=int)
    ind[seen] = np.where(seen)[0]
    extra = draw[~first]
    ind[~seen] = extra[rng.permutation(len(extra))]
    return ind


def wkurt(d, w=None):
    '''Excess kurtosis from weighted samples, with the fourth central moment and
    the squared variance both bias-corrected for the weights.

    Parameters
    ----------
    d : np.ndarray
        Deviations from the (weighted) mean (Nv x Ne)
    w : np.ndarray or None
        Normalized weights (Nv x Ne); None for equal weights

    Returns
    -------
    k : np.ndarray (Nv,)
        Excess kurtosis; NaN where the estimate is ill-conditioned
    '''
    d2 = d * d
    if w is None:
        Ne = d.shape[1]
        V2 = 1.0 / Ne
        V3 = V2 / Ne
        V4 = V3 / Ne
        m2 = np.sum(d2, axis=1) / Ne
        m4 = np.sum(d2 * d2, axis=1) / Ne
    else:
        w2 = w * w
        V2 = np.sum(w2, axis=1)
        V3 = np.sum(w2 * w, axis=1)
        V4 = np.sum(w2 * w2, axis=1)
        m2 = np.sum(w * d2, axis=1)
        m4 = np.sum(w * d2 * d2, axis=1)

    a1 = 1 - 4 * V2 + 6 * V3 - 3 * V4
    a2 = 6 * V2 - 6 * V3 - 9 * V2**2 + 9 * V4
    b1 = V2 - 2 * V3 + V4
    b2 = 1 - 3 * V2 + 2 * V3 + 3 * V2**2 - 3 * V4
    det = a1 * b2 - a2 * b1
    with np.errstate(divide='ignore', invalid='ignore'):
        mu4 = (b2 * m4 - a2 * m2**2) / det
        mu22 = (-b1 * m4 + a1 * m2**2) / det
        k = mu4 / mu22 - 3
    k = np.where((det > 1e-12) & (mu22 > 0), k, np.nan)
    return k


def pf_merge_cf(x, xs, loc, xmpf, var_a, w, xo, nmom=4):
    '''Merge resampled and current particles, then fit up to four moments of the
    weighted posterior with a Cornish-Fisher mapping.

    The resampled and current perturbations about the posterior mean are blended
    with the localization weights. Each variable is then standardized and mapped by

        z + dS/6 (z^2 - 1) + dK/24 (z^3 - 3z) - dS^2/36 (2z^3 - 5z)

    where dS and dK are the target-minus-particle skewness and excess kurtosis,
    each shrunk by its sampling reliability d^2/(d^2 + v) and clipped to keep the
    map close to monotone. The result is rescaled to the posterior mean and
    variance, so nmom = 2 is plain moment matching.

    Parameters
    ----------
    x : np.ndarray
        Current particles (Nv x Ne)
    xs : np.ndarray
        Resampled particles (Nv x Ne)
    loc : np.ndarray
        Localization weight for each variable (Nv,)
    xmpf : np.ndarray
        Posterior mean (Nv,)
    var_a : np.ndarray
        Posterior variance (Nv,)
    w : np.ndarray
        Normalized posterior weights of xo (Nv x Ne)
    xo : np.ndarray
        Particles the weights refer to (Nv x Ne)
    nmom : int
        Number of moments to fit: 2, 3 or 4

    Returns
    -------
    xa : np.ndarray (Nv x Ne)
    '''
    Ne = x.shape[1]
    xmpf = xmpf[:, None]
    loc = loc[:, None]
    xs_p = xs - xmpf
    xa = loc * xs_p + (1 - loc) * (x - xmpf)

    # Standardize the blended particles
    mu_p = np.mean(xa, axis=1, keepdims=True)
    dx_p = xa - mu_p
    sig_p = np.maximum(np.sqrt(np.sum(dx_p**2, axis=1, keepdims=True) / (Ne - 1)), 1e-12)
    z = dx_p / sig_p
    sig_t = np.sqrt(np.maximum(var_a, 1e-12))[:, None]

    if nmom >= 3:
        dx_t = xo - xmpf
        w2 = w * w
        s2 = np.sum(w2, axis=1, keepdims=True)
        s3 = np.sum(w2 * w, axis=1, keepdims=True)
        wd2 = w * dx_t * dx_t
        z2 = z * z
        max_z = np.max(np.abs(z), axis=1, keepdims=True)
        max_z[max_z < 1e-6] = 1.0

        # Skewness: target (weighted, bias-corrected) minus particles (G1)
        denom3 = 1 - 3 * s2 + 2 * s3
        with np.errstate(divide='ignore', invalid='ignore'):
            skew_t = (np.sum(wd2 * dx_t, axis=1, keepdims=True) / denom3) / sig_t**3
        skew_t = np.where((sig_t > 1e-5) & (denom3 > 1e-4), skew_t, 0.0)
        skew_p = np.sum(z2 * z, axis=1, keepdims=True) * Ne / ((Ne - 1) * (Ne - 2))
        neff = 1.0 / s2
        dS = skew_t - skew_p
        dS = dS * dS**2 / (dS**2 + 6.0 / neff + 6.0 / Ne)  # reliability shrinkage
        lim = 2.85 / max_z
        dS = np.clip(dS, -lim, lim)
        zn = z + (dS / 6) * (z2 - 1)

        if nmom >= 4:
            kurt_p = wkurt(dx_p)[:, None]
            kurt_t = wkurt(dx_t, w)[:, None]
            undef = ~np.isfinite(kurt_t) | ~np.isfinite(kurt_p) | ~(sig_t > 1e-5)
            kurt_t[undef] = 0.0
            kurt_p[undef] = 0.0
            dK = kurt_t - kurt_p
            dK = dK * dK**2 / (dK**2 + 24.0 / neff + 24.0 / Ne)
            lim = 4.0 / (max_z * max_z)
            dK = np.clip(dK, -lim, lim)
            z3 = z2 * z
            zn = zn + (dK / 24) * (z3 - 3 * z) - (dS * dS / 36) * (2 * z3 - 5 * z)

        # Re-standardize so the first two moments are exact
        zn = zn - np.mean(zn, axis=1, keepdims=True)
        sz = np.sqrt(np.sum(zn**2, axis=1, keepdims=True) / (Ne - 1))
        with np.errstate(divide='ignore', invalid='ignore'):
            z = zn / sz
        z[(sz <= 1e-12)[:, 0], :] = 0.0

    xa = xmpf + sig_t * z

    # Where the spread is zero (or the result is not finite), use the resampled particle
    bad = ~np.isfinite(xa) | np.broadcast_to(sig_p <= 1e-12, xa.shape)
    if np.any(bad):
        xa[bad] = xs[bad]
    return xa


def pf_merge_qm(x, xs, loc, xmpf, var_a, w, xo, qm_moments=2):
    '''Merge by rank quantile mapping (alternative to pf_merge_cf).

    The resampled and current particles are blended as in pf_merge_cf; the blend
    only supplies the ranks. The particle of rank k gets the quantile (k - 1/2)/Ne
    of the weighted target distribution of xo: knots at the sorted xo with
    mid-point cumulative weights, linear between knots, plus a tail on each side
    of width sqrt(5)*sqrt(2)*Neff^(-1/5)*sigma (Neff of the weights, sigma the
    target standard deviation).

    Parameters
    ----------
    x, xs, loc, xmpf, var_a, w, xo : see pf_merge_cf
    qm_moments : int
        Adjustment after mapping: 2 = exact posterior mean and variance,
        1 = mean only, 0 = none

    Returns
    -------
    xa : np.ndarray (Nv x Ne)
    '''
    Nv, Ne = x.shape
    b = loc[:, None] * (xs - xmpf[:, None]) + (1 - loc[:, None]) * (x - xmpf[:, None])
    ordb = np.argsort(b, axis=1, kind='stable')

    ordo = np.argsort(xo, axis=1, kind='stable')
    xk = np.take_along_axis(xo, ordo, axis=1)
    wk = np.take_along_axis(w, ordo, axis=1)
    Pk = np.cumsum(wk, axis=1) - wk / 2
    sig = np.sqrt(np.maximum(var_a, 1e-12))
    neff = 1.0 / np.sum(w * w, axis=1)
    r = np.sqrt(5) * np.sqrt(2) * neff**(-0.2) * sig
    X = np.hstack([xk[:, :1] - r[:, None], xk, xk[:, -1:] + r[:, None]])
    P = np.hstack([np.zeros((Nv, 1)), Pk, np.ones((Nv, 1))])

    q = (np.arange(1, Ne + 1) - 0.5) / Ne
    xq = np.empty((Nv, Ne))
    for j in range(Nv):
        # inverse CDF; P is nondecreasing, interp handles flat steps by taking the right knot
        idx = np.clip(np.searchsorted(P[j], q, side='right') - 1, 0, Ne)
        p1, p2 = P[j, idx], P[j, idx + 1]
        f = np.clip((q - p1) / np.maximum(p2 - p1, 1e-300), 0.0, 1.0)
        xq[j] = X[j, idx] + f * (X[j, idx + 1] - X[j, idx])

    xa = np.empty((Nv, Ne))
    np.put_along_axis(xa, ordb, xq, axis=1)

    mu = np.mean(xa, axis=1, keepdims=True)
    sd = np.sqrt(np.sum((xa - mu)**2, axis=1, keepdims=True) / (Ne - 1))
    if qm_moments == 2:
        xa = xmpf[:, None] + (xa - mu) * (sig[:, None] / np.maximum(sd, 1e-12))
    elif qm_moments == 1:
        xa = xa - mu + xmpf[:, None]

    bad = ~np.isfinite(xa) | np.broadcast_to(sd <= 1e-12, xa.shape)
    if np.any(bad):
        xa[bad] = xs[bad]
    return xa


def kddm(x, xo, w):
    '''Kernel density distribution mapping (KDDM) for one variable.

    Maps the particles x onto the weighted posterior distribution of xo by
    matching quantiles: each particle's quantile under a kernel estimate of the
    current particles is mapped through the inverse of a kernel estimate of the
    weighted posterior CDF. The kernel CDF uses the algebraic sigmoid
    z/sqrt(1+z^2) in place of erf, with bandwidth sqrt(2)*std(x)*Ne^(-1/5).

    Parameters
    ----------
    x : np.ndarray
        Particles to map (length Ne)
    xo : np.ndarray
        Particles defining the posterior (length Ne)
    w : np.ndarray
        Posterior weights of xo (length Ne)

    Returns
    -------
    xa : np.ndarray (length Ne)
    '''
    Ne = len(w)
    srt = np.argsort(xo, kind='stable')
    xd = xo[srt]
    w_sort = w[srt]

    prior_std = max(np.std(x, ddof=1), 1e-12)
    bw_scale = 1.0 / max(np.sqrt(2) * prior_std * Ne**(-0.2), 1e-15)

    Z = (x[:, None] - x[None, :]) * bw_scale
    qf = 0.5 + 0.5 * np.sum(Z / np.sqrt(1.0 + Z * Z), axis=1) / Ne

    Z = (xd[None, :] - xd[:, None]) * bw_scale  # Z[k, j] = xd[j] - xd[k]
    cdf = 0.5 + 0.5 * (w_sort @ (Z / np.sqrt(1.0 + Z * Z)))

    cdf_u, iu = np.unique(cdf, return_index=True)
    xd_u = xd[iu]
    if len(cdf_u) < 2:
        return np.full(Ne, xd_u[0])

    # linear interpolation with linear extrapolation (MATLAB interp1 'extrap')
    xa = np.interp(qf, cdf_u, xd_u)
    lo = qf < cdf_u[0]
    hi = qf > cdf_u[-1]
    if np.any(lo):
        s = (xd_u[1] - xd_u[0]) / (cdf_u[1] - cdf_u[0])
        xa[lo] = xd_u[0] + s * (qf[lo] - cdf_u[0])
    if np.any(hi):
        s = (xd_u[-1] - xd_u[-2]) / (cdf_u[-1] - cdf_u[-2])
        xa[hi] = xd_u[-1] + s * (qf[hi] - cdf_u[-1])
    return xa


def kddm_rows(x, xo, w, max_elems=4_000_000):
    '''kddm applied to every row of x at once (rows processed in chunks so the
    kernel arrays stay below max_elems elements).

    Parameters
    ----------
    x : np.ndarray
        Particles to map (Nv x Ne)
    xo : np.ndarray
        Particles defining the posterior (Nv x Ne)
    w : np.ndarray
        Posterior weights of xo (Nv x Ne)

    Returns
    -------
    xa : np.ndarray (Nv x Ne)
    '''
    Nv, Ne = x.shape
    xa = np.empty((Nv, Ne))
    step = max(1, max_elems // (Ne * Ne))
    for j0 in range(0, Nv, step):
        sl = slice(j0, min(Nv, j0 + step))
        xc, xoc, wc = x[sl], xo[sl], w[sl]
        srt = np.argsort(xoc, axis=1, kind='stable')
        xd = np.take_along_axis(xoc, srt, axis=1)
        ws = np.take_along_axis(wc, srt, axis=1)

        prior_std = np.maximum(np.std(xc, axis=1, ddof=1), 1e-12)
        bw = (1.0 / np.maximum(np.sqrt(2) * prior_std * Ne**(-0.2), 1e-15))[:, None, None]

        Z = (xc[:, :, None] - xc[:, None, :]) * bw
        qf = 0.5 + 0.5 * np.sum(Z / np.sqrt(1.0 + Z * Z), axis=2) / Ne

        Z = (xd[:, None, :] - xd[:, :, None]) * bw  # Z[r, k, j] = xd[j] - xd[k]
        cdf = 0.5 + 0.5 * np.einsum('rk,rkj->rj', ws, Z / np.sqrt(1.0 + Z * Z))

        for r in range(xc.shape[0]):
            xa[j0 + r] = _kddm_interp(qf[r], cdf[r], xd[r], Ne)
    return xa


def _kddm_interp(qf, cdf, xd, Ne):
    '''Inverse of the kernel CDF (as in kddm): unique CDF values, linear
    interpolation with linear extrapolation.'''
    cdf_u, iu = np.unique(cdf, return_index=True)
    xd_u = xd[iu]
    if len(cdf_u) < 2:
        return np.full(Ne, xd_u[0])
    xa = np.interp(qf, cdf_u, xd_u)
    lo = qf < cdf_u[0]
    hi = qf > cdf_u[-1]
    if np.any(lo):
        s = (xd_u[1] - xd_u[0]) / (cdf_u[1] - cdf_u[0])
        xa[lo] = xd_u[0] + s * (qf[lo] - cdf_u[0])
    if np.any(hi):
        s = (xd_u[-1] - xd_u[-2]) / (cdf_u[-1] - cdf_u[-2])
        xa[hi] = xd_u[-1] + s * (qf[hi] - cdf_u[-1])
    return xa

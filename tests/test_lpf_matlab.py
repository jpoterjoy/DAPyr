'''Checks of the LPF against the MATLAB LPF (lpf_update.m, pf_merge_cf2.m,
pf_merge_qm.m, kddm_fast.m; Oct 2026) that the DART LPF is synchronized with.

tests/states/lpf_matlab_ref.npz holds the inputs and the MATLAB outputs. The
MATLAB run used a deterministic systematic resampler (offset 0.5, extra copies
placed in order), reproduced here by DetRng, so the updates must agree to
rounding (the tempering coefficients come from an iterative solve, hence the
tolerance).'''
import sys
import unittest
sys.path.append("src")
import numpy as np
import DAPyr as dap
import DAPyr.MISC
import DAPyr.OBS_ERRORS


class DetRng:
      def random(self):
            return 0.5
      def permutation(self, n):
            return np.arange(n)


def maxrel(a, b):
      return np.max(np.abs(a - b)) / np.max(np.abs(b))


class TestLPFMatlab(unittest.TestCase):

      def setUp(self):
            self.d = np.load('./tests/states/lpf_matlab_ref.npz')
            return super().setUp()

      def _update(self, hx, y, frac_neff, kddm_flag, maxiter):
            d = self.d
            x = d['x']
            Ne = x.shape[1]
            L = dap.OBS_ERRORS.get_likelihood(0, {'mu': 0, 'sigma': np.sqrt(float(d['var_y']))})
            xa, e_flag = dap.DA.lpf_update(x.copy(), hx, y, d['H'], d['C'], frac_neff*Ne, 0.0, maxiter,
                                           kddm_flag, 0, d['qc'], L, rng = DetRng())
            self.assertEqual(e_flag, 0)
            return xa

      def test_linear_single_pass(self):
            d = self.d
            xa = self._update(d['H'] @ d['x'], d['y_lin'], 0.5, 0, 1)
            self.assertLess(maxrel(xa, d['ml_xa1']), 1e-6)

      def test_linear_tempered_kddm(self):
            d = self.d
            xa = self._update(d['H'] @ d['x'], d['y_lin'], 0.8, 1, 10)
            self.assertLess(maxrel(xa, d['ml_xa2']), 1e-4)

      def test_quadratic_single_pass(self):
            d = self.d
            xa = self._update(d['H'] @ d['x']**2, d['y_sq'], 0.3, 0, 1)
            self.assertLess(maxrel(xa, d['ml_xa3']), 1e-6)

      def test_quadratic_tempered_kddm(self):
            d = self.d
            xa = self._update(d['H'] @ d['x']**2, d['y_sq'], 0.5, 1, 5)
            self.assertLess(maxrel(xa, d['ml_xa4']), 1e-4)

      def test_merges_and_kddm(self):
            d = self.d
            args = [d['ml_' + k] for k in ['xc', 'xs', 'loc', 'xm', 'va', 'w', 'xo']]
            for n in (2, 3, 4):
                  self.assertLess(maxrel(dap.MISC.pf_merge_cf(*args, n), d['ml_m%d' % n]), 1e-12)
            self.assertLess(maxrel(dap.MISC.pf_merge_qm(*args), d['ml_q2']), 1e-12)
            xc, xo, w = d['ml_xc'], d['ml_xo'], d['ml_w']
            self.assertLess(maxrel(dap.MISC.kddm(xc[0], xo[0], w[0]), d['ml_k1']), 1e-12)
            self.assertLess(maxrel(dap.MISC.kddm(xc[4], xo[4], w[4]), d['ml_k2']), 1e-12)

      def test_bootstrap_limit(self):
            # Without localization (rho = 1) and without tempering, the LPF is the bootstrap
            # PF: the final merge replaces every variable by particles resampled with the full
            # importance weights, standardized to the importance-weighted mean and variance
            d = self.d
            x = d['x']
            Nx, Ne = x.shape
            H = d['H']
            Ny = H.shape[0]
            sig = 3.0
            C = np.ones((Ny, Nx))
            L = dap.OBS_ERRORS.get_likelihood(0, {'mu': 0, 'sigma': sig})
            xa, e_flag = dap.DA.lpf_update(x.copy(), H @ x, d['y_lin'], H, C, 1.0, 0.0, 1, 0, 0,
                                           np.zeros(Ny), L, n_moments = 2, min_weight = 0,
                                           rng = np.random.default_rng(1))
            self.assertEqual(e_flag, 0)
            lw = np.sum((d['y_lin'][:, None] - H @ x)**2, axis = 0) / (2 * sig**2)
            w = np.exp(-(lw - lw.min()))
            w = w / w.sum()
            xm = x @ w
            xv = ((x - xm[:, None])**2 @ w) / (1 - np.sum(w**2))
            self.assertTrue(np.allclose(np.mean(xa, axis = 1), xm))
            self.assertTrue(np.allclose(np.var(xa, axis = 1, ddof = 1), xv))


if __name__ == '__main__':
      unittest.main()

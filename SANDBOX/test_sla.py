# test_sla.py
import numpy.f2py as npc
from math import log
from math import exp
func = """subroutine spec_la(tau_leaf, sla)

      real(8),intent(in) :: tau_leaf  !years
      real(8),intent(out) :: sla   !m2 gC-1

      real(8) :: n_tau_leaf, tl0

      n_tau_leaf = (tau_leaf - 0.08333333)/(8.33333333 - 0.08333333)

      tl0 = (365.242D0 / 12.0D0 * 10.00D0) ** (2.00D0*n_tau_leaf)

      sla = (3D-2 * (365.2420D0 / tl0) ** (-0.460D0))

   end subroutine spec_la"""

npc.compile(func, modulename='sla', verbose=0, extension='.f90')
import sla


def nrubisco(leaf_t, n_in):
    from math import e

    tl = e**(-(leaf_t + 1.03)) + 0.08

    return tl * (n_in * 0.3)


def vcmax(leaf_turnover, leaf_c, nbio):
    xbio = nrubisco(leaf_turnover, nbio)
    N = (nbio * 1e3) / (leaf_c * 1000)

    a = 1.57
    b = 0.55

    vm = a + (b * log(N, 10))
    return 10**vm  # * 1e-6


def calc_ratios(pool):
    import numpy as np
    from random import shuffle

    pool_n2c = np.linspace(0.003, 0.04, 500)
    pool_p2c = np.linspace(1e-5, 0.005, 500)

    if pool == 'leaf' or pool == 'root':
        pass
    else:
        pool_n2c /= 100.0
        pool_p2c /= 100.0

    x = [[a, b] for a in pool_n2c for b in pool_p2c if (
        (a / b) >= 3.0) and ((a / b) <= 50.0)]
    assert len(x) > 0, "zero len"
    shuffle(x)
    return x

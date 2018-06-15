# coding: utf-8

""" Utilities for fitting orbits to stream data. """

# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import norm
from gala.units import galactic

# Project
from . import galactocentric_frame, vcirc, vlsr
from .coordinates import Ophiuchus
from .util import integrate_forward_backward

__all__ = ['ln_prior', 'ln_likelihood', 'ln_posterior']

# ----------------------------------------------------------------------------
# For inference:

def _unpack(p, freeze=None):
    if freeze is None:
        freeze = dict()

    # these are for the initial conditions
    phi2, d, mu1, mu2, vr = p[:5]
    count_ix = 5

    # time to integrate forward and backward
    if 't_forw' not in freeze:
        t_forw = p[count_ix]
        count_ix += 1
    else:
        t_forw = freeze['t_forw']

    if 't_back' not in freeze:
        t_back = p[count_ix]
        count_ix += 1
    else:
        t_back = freeze['t_back']

    # prior on instrinsic width of stream
    if 'phi2_sigma' not in freeze:
        phi2_sigma = p[count_ix]
        count_ix += 1
    else:
        phi2_sigma = freeze['phi2_sigma']

    # prior on instrinsic depth (distance) of stream
    if 'd_sigma' not in freeze:
        d_sigma = p[count_ix]
        count_ix += 1
    else:
        d_sigma = freeze['d_sigma']

    # prior on instrinsic LOS velocity dispersion of stream
    if 'vr_sigma' not in freeze:
        vr_sigma = p[count_ix]
        count_ix += 1
    else:
        vr_sigma = freeze['vr_sigma']

    return phi2, d, mu1, mu2, vr, t_forw, t_back, phi2_sigma, d_sigma, vr_sigma

def ln_prior(p, ophdata, potential, dt, freeze=None):
    """
    Evaluate the prior over stream orbit fit parameters.

    See docstring for `ln_likelihood()` for information on args and kwargs.

    """

    # log prior value
    lp = 0.

    # unpack the parameters and the frozen parameters
    phi2,d,mul,mub,vr,t_forw,t_back,phi2_sigma,d_sigma,vr_sigma = _unpack(p, freeze)

    # time to integrate forward and backward
    t_integ = np.abs(t_forw) + np.abs(t_back)
    if t_forw <= t_back:
        raise ValueError("Forward integration time less than or equal to "
                         "backwards integration time.")

    if t_forw < dt or t_back > -dt:
        return -np.inf

    # prior on instrinsic width of stream
    if 'phi2_sigma' not in freeze:
        if phi2_sigma <= 0.:
            return -np.inf
        lp += -np.log(phi2_sigma)

    # prior on instrinsic depth (distance) of stream
    if 'd_sigma' not in freeze:
        if d_sigma <= 0.:
            return -np.inf
        lp += -np.log(d_sigma)

    # prior on instrinsic LOS velocity dispersion of stream
    if 'vr_sigma' not in freeze:
        if vr_sigma <= 0.:
            return -np.inf
        lp += -np.log(vr_sigma)

    # strong prior on phi2
    if phi2 < -np.pi/2. or phi2 > np.pi/2:
        return -np.inf
    lp += norm.logpdf(phi2, loc=0., scale=phi2_sigma)

    # uniform prior on integration time
    ntimes = int(t_integ / dt) + 1
    if t_integ <= 2. or t_integ > 1000. or ntimes < 4:
        return -np.inf

    return lp

def ln_likelihood(p, ophdata, potential, dt, freeze=None):
    """
    Evaluate the stream orbit fit likelihood.

    Parameters
    ----------
    p : iterable
        The parameters of the model: distance, proper motions, radial velocity, the integration time,
        and (optionally) intrinsic angular width of the stream.
    potential : :class:`gala.potential.PotentialBase`
        The gravitational potential.
    dt : float
        Timestep for integrating the orbit.

    Returns
    -------
    ll : :class:`numpy.ndarray`
        An array of likelihoods for each data point.

    """
    chi2 = 0.

    # unpack the parameters and the frozen parameters
    phi2,d,mul,mub,vr,t_forw,t_back,phi2_sigma,d_sigma,vr_sigma = _unpack(p, freeze)

    w0 = ophdata._mcmc_sample_to_w0([phi2,d,mul,mub,vr])[:,0]

    # HACK: a prior on velocities
    vmag2 = np.sum(w0[3:]**2)
    chi2 += -vmag2 / (0.15**2)

    # integrate the orbit
    orbit = integrate_forward_backward(potential, w0, t_back=t_back, t_forw=t_forw)

    # rotate the model points to stream coordinates
    model_c,model_v = orbit.to_frame(coord.Galactic, vcirc=vcirc, vlsr=vlsr,
                                     galactocentric_frame=galactocentric_frame)
    model_oph = model_c.transform_to(Ophiuchus)

    # model stream points in ophiuchus coordinates
    model_phi1 = model_oph.phi1
    model_phi2 = model_oph.phi2.radian
    model_d = model_oph.distance.decompose(galactic).value
    model_mul,model_mub,model_vr = [x.decompose(galactic).value for x in model_v]

    # for independent variable, use cos(phi)
    data_x = np.cos(ophdata.coord_oph.phi1)
    model_x = np.cos(model_phi1)
    # data_x = ophdata.coord_oph.phi1.wrap_at(180*u.deg).radian
    # model_x = model_phi1.wrap_at(180*u.deg).radian
    ix = np.argsort(model_x)

    # shortening for readability -- the data
    c,v = ophdata.coord_oph, ophdata.veloc
    c_err,v_err = ophdata.coord_err, ophdata.veloc_err

    # define interpolating functions
    order = 3
    # bbox = [-np.pi, np.pi]
    bbox = [-1, 1]
    phi2_interp = InterpolatedUnivariateSpline(model_x[ix], model_phi2[ix], k=order, bbox=bbox) # change bbox to units of model_x
    d_interp = InterpolatedUnivariateSpline(model_x[ix], model_d[ix], k=order, bbox=bbox)
    mul_interp = InterpolatedUnivariateSpline(model_x[ix], model_mul[ix], k=order, bbox=bbox)
    mub_interp = InterpolatedUnivariateSpline(model_x[ix], model_mub[ix], k=order, bbox=bbox)
    vr_interp = InterpolatedUnivariateSpline(model_x[ix], model_vr[ix], k=order, bbox=bbox)

    chi2 += -(phi2_interp(data_x) - c.phi2.radian)**2 / phi2_sigma**2 - 2*np.log(phi2_sigma)

    err = c_err['distance'].decompose(galactic).value
    chi2 += -(d_interp(data_x) - c.distance.decompose(galactic).value)**2 / (err**2 + d_sigma**2) - np.log(err**2 + d_sigma**2)

    err = v_err['mul'].decompose(galactic).value
    chi2 += -(mul_interp(data_x) - v['mul'].decompose(galactic).value)**2 / (err**2) - 2*np.log(err)

    err = v_err['mub'].decompose(galactic).value
    chi2 += -(mub_interp(data_x) - v['mub'].decompose(galactic).value)**2 / (err**2) - 2*np.log(err)

    err = v_err['vr'].decompose(galactic).value
    chi2 += -(vr_interp(data_x) - v['vr'].decompose(galactic).value)**2 / (err**2 + vr_sigma**2) - np.log(err**2 + vr_sigma**2)

    # this is some kind of whack prior - don't integrate more than we have to
    chi2 += -(model_phi1.radian.min() - ophdata.coord_oph.phi1.radian.min())**2 / (phi2_sigma**2)
    chi2 += -(model_phi1.radian.max() - ophdata.coord_oph.phi1.radian.max())**2 / (phi2_sigma**2)

    return 0.5*chi2

def ln_posterior(p, *args, **kwargs):
    """
    Evaluate the stream orbit fit posterior probability.

    See docstring for `ln_likelihood()` for information on args and kwargs.

    Returns
    -------
    lp : float
        The log of the posterior probability.

    """

    lp = ln_prior(p, *args, **kwargs)
    if not np.isfinite(lp):
        return -np.inf

    ll = ln_likelihood(p, *args, **kwargs)
    if not np.all(np.isfinite(ll)):
        return -np.inf

    return lp + ll.sum()

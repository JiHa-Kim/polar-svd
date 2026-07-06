from collections import namedtuple
from fractions import Fraction as F
from math import frexp, fsum, inf, isfinite, isqrt, ldexp, nextafter as na, sqrt

import torch

SVDResult, SpectralEnvelope = namedtuple("SVDResult", "u s vh info"), namedtuple("SpectralEnvelope", "sigma_lo sigma_hi")
_FMAX = torch.finfo(torch.float64).max


def _hi(x, err=0):
    return inf if (x := F(x) + err) > _FMAX else f if F(f := float(x)) >= x else na(f, inf)


def _lo(x, err=0):
    return _FMAX if (x := max(F(0), F(x) - err)) > _FMAX else f if F(f := float(x)) <= x else na(f, 0.0)


def _root_hi(x):
    return F((s := isqrt(n := (x := F(x)).numerator * x.denominator << 128)) + (s * s < n), x.denominator << 64)


def _gamma(k, dtype):
    return inf if (x := k * 0.5 * float(torch.finfo(dtype).eps)) >= 1.0 else na(x / (1.0 - x), inf)


def _sum_sq(x):
    err, total = _gamma(x.numel() + 1, torch.float64), fsum(float((c := chunk.double()) @ c) for chunk in x.reshape(-1).split(2**26))
    return (0.0, inf) if err >= 1.0 or not isfinite(total) else (max(0.0, na(total / (1.0 + err), 0.0)), na(total / (1.0 - err), inf))


def _prod_norm_hi(a, b=None):
    a, b = (None, a.abs()) if b is None else (a.abs(), b.abs())
    if (err := _gamma(max(*b.shape) if a is None else max(*a.shape, *b.shape), torch.float64)) >= 1.0:
        return inf
    v = torch.ones(b.shape[1], dtype=b.dtype, device=b.device)
    if not isfinite(vmax := float((w := b.mT @ (b @ v) if a is None else b.mT @ (a.mT @ (a @ (b @ v)))).max())) or vmax <= 0.0:
        return inf
    v, b = (w / vmax).clamp_min(torch.finfo(b.dtype).tiny).double(), b.double()
    ratio = float(((b.mT @ (b @ v) if a is None else b.mT @ ((a := a.double()).mT @ (a @ (b @ v)))) / v).max())
    return inf if not isfinite(ratio) or ratio < 0.0 else _hi((F(ratio) if a is None else _root_hi(ratio)) / (1 - F(err)) ** 2)


def _theta(y):
    if not na(2.0**-1022, 0.0) <= y < 1.0:
        return 1.0 if y >= 1.0 else 0.0
    t = 0.7937005259840998 * (c := y ** (1.0 / 3.0)) * (1.0 + 1.0793683991589853 * c) / (1.0 + 0.650395792127202 * c)
    for _ in (0, 1):
        t *= ((v := (u := t * t) * t) * (u + 2.0 * t + 2.0) + y * (4.0 * u + 5.0 * t + 2.0)) / (v * (2.0 * u + 5.0 * t + 4.0) + y * (2.0 * u + 2.0 * t + 1.0))
    return t * (1.0 - 2.0**-45)


def _qdwh_params(ell, dtype):
    r_floor = (tiny := float(torch.finfo(dtype).tiny)) + sqrt(tiny * (1.0 + tiny))
    r = max(_theta(ell * ell), r_floor)
    beta, shift, eta = 1.0 / (q := 1.0 + 2.0 * r), r * r / q, 2.0 * r * (1.0 + r) ** 2 / (q * q)
    gap = na((1.0 + r) * (1.0 - r) ** 3 / (q * q * q), inf)
    ell_next = na(1.0 - gap / (1.0 + sqrt(max(0.0, 1.0 - gap))), 0.0) if gap < 0.5 else na(sqrt(max(0.0, na(r * (r + 2.0) ** 3 / (q * q * q), 0.0))), 0.0)
    return beta, shift, sqrt(shift), eta, gap, ell_next


def _qdwh_schedule(l0, dtype, stop=None):
    stop = (tol := 0.5 * float(torch.finfo(dtype).eps)) * (2.0 - tol) if stop is None else stop
    ell, gap, steps = 0.0 if l0 < sqrt(torch.finfo(dtype).tiny) else na(l0, 0.0), inf, []
    while gap > stop and max(0.0, (1.0 - ell) * (1.0 + ell)) > stop:
        beta, shift, sqrt_shift, eta, gap, ell = _qdwh_params(ell, dtype)
        steps.append((beta, shift, sqrt_shift, eta))
    return steps


def _qdwh_scale(env, m, n, dtype):
    if env.sigma_lo == env.sigma_hi:
        return env.sigma_hi, float(env.sigma_hi > 0.0)
    div_floor, div_round = _root_hi(m * n) * F(torch.finfo(dtype).tiny), _root_hi(n) * F(_gamma(1, dtype)) * F(min(env.sigma_hi, _FMAX))
    alpha = _hi((F(env.sigma_hi) + div_round) / (1 - div_floor)) if isfinite(env.sigma_hi) and div_floor < 1 else inf
    if not isfinite(alpha):
        if 0.0 < env.sigma_hi <= float(torch.finfo(dtype).max):
            return env.sigma_hi, 0.0
        raise RuntimeError("could not certify finite QDWH scale")
    l0 = _lo((F(env.sigma_lo) - div_round) / F(alpha), div_floor)
    pow2 = inf if (exp2 := (me := frexp(_hi(F(env.sigma_hi) / (1 - div_floor))))[1] - (me[0] == 0.5)) > 1023 else ldexp(1.0, exp2)
    if env.sigma_lo > 0.0 and pow2 <= float(torch.finfo(dtype).max) and (pow2_l0 := _lo(F(env.sigma_lo) / F(pow2), div_floor)) > l0:
        alpha, l0 = pow2, pow2_l0
    return alpha, min(l0, 1.0)


def _qdwh_work(lo, hi, n, dtype):
    steps = _qdwh_schedule(_qdwh_scale(SpectralEnvelope(lo, hi), n, n, dtype)[1], dtype) if lo >= 0.0 and hi > 0.0 and isfinite(hi) else None
    return inf if steps is None else len(steps) * (16.0 * n**3 / 3.0 + n * n)


def _auto_reduce(m, n, dtype):
    shifts = [step[1] for step in _qdwh_schedule(0.0, dtype)]
    tall, square = (sum(4.0 * d * n * n + d * n + (4.0 if s <= _gamma(d, dtype) else 1.0) * n**3 / 3.0 for s in shifts) for d in (m, n))
    return 19.0 * n**3 / 3.0 + square < tall


def _inverse_gram_norm24_hi(y, inv_abs):
    n, dot, sym = y.shape[0], _gamma(y.shape[0], y.dtype), _gamma(2, y.dtype)
    h = 0.5 * ((h := y.mT @ y) + h.mT)
    q1lo, q1hi = _sum_sq(h)
    if dot >= 1.0 or sym >= 1.0 or not isfinite(inv_abs) or not isfinite(q1hi) or not isfinite(q2s := _sum_sq(h @ h)[1]):
        return inf
    dot, floor = F(dot), F(n * n) * F(torch.finfo(y.dtype).tiny)
    delta = (dot + F(sym) * (1 + dot)) * F(inv_abs) + floor
    q2 = (_root_hi(q2s) + dot * F(q1hi) + floor) ** 2
    spread = F(n - 1, n) * max(F(0), q2 - F(q1lo) ** 2 / n)
    return _hi(_root_hi(_root_hi(F(q1hi) / n + _root_hi(spread)) + delta))


def _residual_plan(n, dtype, c_hi):
    blk, floor, best = min(n, 64), F(n * n) * F(torch.finfo(dtype).tiny), (inf, 1)
    if not isfinite(c_hi) or not isfinite(_gamma(panels := (n + blk - 1) // blk, dtype)):
        return inf, blk, 1
    panel_hi, scan = (1 + (gb := F(_gamma(blk, dtype)))) * F(c_hi), isqrt(panels) + 2
    for group in sorted({*range(1, scan), *((panels + v - 1) // v for v in range(1, scan))}):
        gg, gh = F(_gamma(group, dtype)), F(_gamma((panels + group - 1) // group, dtype))
        best = min(best, (gg * panel_hi + gh * (1 + (1 + gg) * panel_hi), group))
    return _hi(best[0] + gb * F(c_hi) + floor), blk, best[1]


def _triangular_inverse_lower(r, norm_err, sigma_hi, base_lo, sigma_ub):
    try:
        y = torch.linalg.solve_triangular(r, torch.eye(n := r.shape[0], dtype=r.dtype, device=r.device), upper=True)
    except RuntimeError:
        return 0.0
    ylo, yhi = _sum_sq(y)
    if not isfinite(inv_bound := min(yhi, _prod_norm_hi(y))) or inv_bound <= 0.0 or not isfinite(rsq := _sum_sq(r)[1]):
        return 0.0
    inv_hi = _root_hi(inv_bound)
    if _qdwh_work(max(base_lo, _lo(1 / inv_hi, norm_err)), sigma_hi, n, r.dtype) + n**3 >= _qdwh_work(base_lo, sigma_hi, n, r.dtype):
        return 0.0
    plans = [(*_residual_plan(n, r.dtype, min(_prod_norm_hi(lhs, rhs), _hi(_root_hi(F(rsq) * F(yhi))))), lhs, rhs) for lhs, rhs in ((y, r), (r, y))]
    acc_err, blk, group, left, right = min(plans, key=lambda p: p[0])
    if not acc_err < 1.0:
        return 0.0
    resid, partial = -torch.eye(n, dtype=r.dtype, device=r.device), torch.empty_like(r)
    for j in range(0, n, blk * group):
        partial.zero_()
        for k in range(j, min(n, j + blk * group), blk):
            partial.addmm_(left[:, k : k + blk], right[k : k + blk])
        resid.add_(partial)
    slack = 1 - _root_hi(resid_sq) - F(acc_err) if isfinite(resid_sq := _sum_sq(resid)[1]) else F(-1)
    lower = _lo(slack / inv_hi, norm_err)
    if _qdwh_work(min(sigma_ub, _lo(slack / _root_hi(F(ylo) / n), norm_err)), sigma_hi, n, r.dtype) + 2.0 * n**3 >= _qdwh_work(lower, sigma_hi, n, r.dtype):
        return lower
    return lower if not isfinite(g24 := _inverse_gram_norm24_hi(y, inv_bound)) else max(lower, _lo(slack / F(g24), norm_err))


def spectral_envelope(a, triangular=False):
    torch.backends.cuda.matmul.allow_tf32 = False
    amax = float(a.abs().max())
    if amax == 0.0 or a.numel() == 1:
        return SpectralEnvelope(amax, amax)
    pow2 = inf if (exp2 := (me := frexp(amax))[1] - (me[0] == 0.5)) > 1023 else ldexp(1.0, exp2)
    scale, exact = (pow2, True) if pow2 <= float(torch.finfo(a.dtype).max) else (amax, False)
    tlo, thi = _sum_sq(x := a / scale)
    mu = 0.5 * (tlo + thi) / (gx := (x.mT if x.shape[1] > x.shape[0] else x).contiguous()).shape[1]
    h, g1 = (gx.mT @ gx).to(torch.float64, copy=True), F(_gamma(1, torch.float64))
    h.diagonal().sub_(mu)
    hi2, lo2 = min(thi, abs_norm := _prod_norm_hi(gx)), F(0)
    if isfinite(abs_norm) and isfinite(ge := _gamma(gx.shape[0], x.dtype)):
        radius = _root_hi(_sum_sq(h)[1]) + g1 * _root_hi(_sum_sq(h.diagonal().abs() + 2.0 * abs(mu))[1]) / (1 - g1) + F(ge) * F(abs_norm)
        hi2, lo2 = min(hi2, _hi(F(mu) + radius)), max(F(0), F(mu) - radius)
    norm_err = _root_hi(x.numel()) * F(torch.finfo(x.dtype).tiny) + (0 if exact else F(_gamma(1, x.dtype)) * _root_hi(thi))
    sigma_lo, sigma_hi = _lo(lo2 / _root_hi(lo2), norm_err) if lo2 else 0.0, _hi(_root_hi(hi2), norm_err)
    if triangular and x.dtype == torch.float32 and x.shape[0] == x.shape[1] and (diag_min := float(x.diagonal().abs().min())) > 0.0:
        sigma_ub = _lo(diag_min, norm_err)
        if _qdwh_work(max(sigma_lo, sigma_ub), sigma_hi, x.shape[0], x.dtype) + x.shape[0] ** 3 < _qdwh_work(sigma_lo, sigma_hi, x.shape[0], x.dtype):
            sigma_lo = max(sigma_lo, _triangular_inverse_lower(x.contiguous(), norm_err, sigma_hi, sigma_lo, sigma_ub))
    return SpectralEnvelope(_lo(F(sigma_lo) * F(scale)) if sigma_lo > 0.0 else 0.0, min(_hi(F(sigma_hi) * F(scale)), _hi(F(amax) * _root_hi(a.numel()))))


def qdwh_polar(a, *, triangular=False):
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.ndim != 2 or 0 in a.shape or a.shape[0] < a.shape[1] or a.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(a).all()):
        raise ValueError("expected a tall float32/float64 matrix")
    m, n, dtype = *a.shape, a.dtype
    alpha, l0 = _qdwh_scale(spectral_envelope(a, triangular), m, n, dtype)
    steps = [] if alpha == 0.0 else _qdwh_schedule(l0, dtype)
    x = x0 = torch.zeros_like(a) if alpha == 0.0 else (a.double() / alpha if alpha > float(torch.finfo(dtype).max) else a / alpha).to(dtype).contiguous()
    routes, qr_work, gram_noise = {"qr": 0, "chol": 0}, None, _gamma(m, dtype)
    for beta, shift, sqrt_shift, eta in steps:
        if shift > gram_noise:
            qr_work = None
            gram = 0.5 * ((gram := x.mT @ x) + gram.mT)
            gram.diagonal().add_(shift)
            chol, fail = torch.linalg.cholesky_ex(gram)
            if not bool(fail.any()):
                x = beta * x + eta * torch.cholesky_solve(x.mT, chol).mT
                routes["chol"] += 1
                continue
        if qr_work is None:
            qr_work = torch.empty((m + n, n), dtype=dtype, device=x.device)
            qr_work[m:].zero_()
        qr_work[:m] = x
        qr_work[m:].diagonal().fill_(sqrt_shift)
        q, _ = torch.linalg.qr(qr_work)
        x = torch.addmm(x, q[:m], q[m:].mT, beta=beta, alpha=eta / sqrt_shift)
        routes["qr"] += 1
    core = x.mT @ x0
    skew = inf if not isfinite(csq := _sum_sq(core - core.mT)[1]) else _hi(_root_hi(csq) / 2)
    skew_budget = _hi(F(min(_gamma(x0.shape[0], x0.dtype), _FMAX)) * _root_hi(_sum_sq(x0)[1]))
    info = dict(alpha=alpha, l0=l0, iters=len(steps), routes=routes, core_skew=skew, skew_budget=skew_budget)
    return x, x0, core, info


def _finish(w, x0, core, info):
    if skewed := info["core_skew"] > info["skew_budget"]:
        vc, s_unit, vh = torch.linalg.svd(core, full_matrices=False)
    else:
        evals, evecs = torch.linalg.eigh(0.5 * (core + core.mT))
        vc, s_unit = torch.flip(evecs, [1]), torch.flip(evals, [0]).clamp_min(0)
        vh = vc.mT.contiguous()
    info["core"] = "svd" if skewed else "eigh"
    rank, trusted = int((s_unit > 0.0).sum()), info["l0"] > info["core_skew"]
    s = s_unit.double() * float(info["alpha"])
    s = s if x0.dtype == torch.float32 and bool((s > float(torch.finfo(x0.dtype).max)).any()) else s.to(x0.dtype)
    u = w @ vc[:, :rank] if trusted else (x0 @ vh[:rank].mT) / s_unit[:rank]
    if rank and not trusted:
        u, r = torch.linalg.qr(u)
        u = u * torch.where(r.diagonal() < 0.0, -1.0, 1.0).to(u.dtype)
    if rank < vh.shape[0]:
        q, _ = torch.linalg.qr(torch.cat((u, torch.eye(x0.shape[0], vh.shape[0], dtype=x0.dtype, device=x0.device)), dim=1))
        u = torch.cat((u, q[:, rank : vh.shape[0]]), dim=1)
    return u, s, vh


def svd(a, *, reduce_tall="auto"):
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.ndim != 2 or 0 in a.shape or a.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(a).all()):
        raise ValueError("expected a finite float32/float64 matrix")
    m, n = a.shape
    if m < n:
        return SVDResult((sub := svd(a.mT.contiguous(), reduce_tall=reduce_tall)).vh.mT.contiguous(), sub.s, sub.u.mT.contiguous(), sub.info)
    if m > n and (reduce_tall is True or (reduce_tall == "auto" and _auto_reduce(m, n, a.dtype))):
        hh, tau = torch.geqrf(a)
        r = torch.triu(hh[:n, :]).contiguous()
        if not bool(torch.isfinite(r).all()):
            return svd(a, reduce_tall=False)
        w, x0, core, info = qdwh_polar(r, triangular=True)
        ur, s, vh = _finish(w, x0, core, info)
        a_norm_hi = 0.0 if float(a.abs().max()) == 0.0 else inf if not isfinite(sq := _sum_sq(a)[1]) else _hi(_root_hi(sq))
        info.update({"polar_alpha": info["alpha"], "polar_l0": info["l0"], "alpha": a_norm_hi, "l0": 0.0, "reduced": True})
        return SVDResult(torch.ormqr(hh, tau, torch.cat((ur, hh.new_zeros((m - n, ur.shape[1]))))), s, vh, info)
    w, x0, core, info = qdwh_polar(a)
    return SVDResult(*_finish(w, x0, core, info), info | {"reduced": False})

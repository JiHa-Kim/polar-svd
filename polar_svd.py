"""Certified SVD via QR-reduced QDWH polar: scalar bounds are exact rationals rounded outward only at the float boundary, so every decision holds in the worst case."""

from collections import namedtuple
from fractions import Fraction as F
from math import frexp, fsum, inf, isfinite, isqrt, ldexp, nextafter, sqrt

import torch

SVDResult, SpectralEnvelope = namedtuple("SVDResult", "u s vh info"), namedtuple("SpectralEnvelope", "sigma_lo sigma_hi")
_up, _dn, _FMAX = (lambda x: nextafter(x, inf)), (lambda x: nextafter(x, 0.0)), torch.finfo(torch.float64).max  # directed rounding and the float ceiling


def _hi(x, err=0):  # smallest float >= the exact value x + err: certified upper bounds leave rational arithmetic only here or through _lo
    return inf if (x := F(x) + err) > _FMAX else f if F(f := float(x)) >= x else _up(f)


def _lo(x, err=0):  # largest float <= the exact value max(0, x - err)
    return _FMAX if (x := max(F(0), F(x) - err)) > _FMAX else f if F(f := float(x)) <= x else _dn(f)


def _root_hi(x):  # rational upper bound on sqrt(x >= 0) via the exact integer square root: at most 2^-64 relative slack, exact on perfect squares
    return F((s := isqrt(n := (x := F(x)).numerator * x.denominator << 128)) + (s * s < n), x.denominator << 64)


def _gamma(k, dtype):  # gamma_k = ku / (1 - ku) with u the unit roundoff: standard bound on relative error accumulated over k rounded operations
    return inf if (x := k * 0.5 * float(torch.finfo(dtype).eps)) >= 1.0 else _up(x / (1.0 - x))


def _sum_sq(x):  # certified enclosure [lo, hi] of sum(x**2), accumulated over exact fp64 chunk dot products
    e, s = _gamma(x.numel() + 1, torch.float64), fsum(float((c := w.double()) @ c) for w in x.reshape(-1).split(2**26))
    return (0.0, inf) if e >= 1.0 or not isfinite(s) else (max(0.0, _dn(s / (1.0 + e))), _up(s / (1.0 - e)))


def _prod_norm_hi(a, b=None):  # certified upper bound on ||a @ b||_2 — or on the gram norm ||a.T a||_2 = ||a||_2^2 when b is None
    if (err := _gamma(max(*a.shape) if b is None else max(*a.shape, *b.shape), torch.float64)) >= 1.0:
        return inf
    gram, b, a = b is None, (a if b is None else b).abs(), None if b is None else a.abs()
    v = torch.ones(b.shape[1], dtype=b.dtype, device=b.device)
    if not isfinite(vmax := float((w := b.mT @ (b @ v) if gram else b.mT @ (a.mT @ (a @ (b @ v)))).max())) or vmax <= 0.0:
        return inf
    v, b = (w / vmax).clamp_min(torch.finfo(b.dtype).tiny).double(), b.double()
    q = float(((b.mT @ (b @ v) if gram else b.mT @ ((a := a.double()).mT @ (a @ (b @ v)))) / v).max())
    return inf if not isfinite(q) or q < 0.0 else _hi((F(q) if gram else _root_hi(q)) / (1 - F(err)) ** 2)


def _theta(y):  # certified under-estimate of the largest r with F(r) = r^3 (r + 2) / (1 + 2 r) <= y; cube-root seed derived, rational coefficients calibrated
    if not _dn(2.0**-1022) <= y < 1.0:  # S12-certified domain [largest subnormal, 1): its floor is the boundary-seed y; endpoints are one-sided
        return 1.0 if y >= 1.0 else 0.0
    t = 0.7937005259840998 * (c := y ** (1.0 / 3.0)) * (1.0 + 1.0793683991589853 * c) / (1.0 + 0.650395792127202 * c)
    for _ in (0, 1):
        t *= ((v := (u := t * t) * t) * (u + 2.0 * t + 2.0) + y * (4.0 * u + 5.0 * t + 2.0)) / (v * (2.0 * u + 5.0 * t + 4.0) + y * (2.0 * u + 2.0 * t + 1.0))
    return t * (1.0 - 2.0**-45)


def _qdwh_params(ell, dtype):  # one r-form Halley step from certified lower bound ell: returns (beta, shift, sqrt_shift, eta, gap, ell_next)
    re = (tiny := float(torch.finfo(dtype).tiny)) + sqrt(tiny * (1.0 + tiny))  # smallest viable weight: shift(re) = tiny exactly
    r = max(_theta(ell * ell), re)
    beta, shift, eta = 1.0 / (q := 1.0 + 2.0 * r), r * r / q, 2.0 * r * (1.0 + r) ** 2 / (q * q)
    gap = _up((1.0 + r) * (1.0 - r) ** 3 / (q * q * q))
    ell_next = _dn(1.0 - gap / (1.0 + sqrt(max(0.0, 1.0 - gap)))) if gap < 0.5 else _dn(sqrt(max(0.0, _dn(r * (r + 2.0) ** 3 / (q * q * q)))))
    return beta, shift, sqrt(shift), eta, gap, ell_next


def _qdwh_schedule(l0, dtype, stop=None):  # the full certified step list, fixed upfront from l0 alone — no per-step device sync
    stop = (tol := 0.5 * float(torch.finfo(dtype).eps)) * (2.0 - tol) if stop is None else stop
    ell, gap, steps = 0.0 if l0 < sqrt(torch.finfo(dtype).tiny) else _dn(l0), inf, []
    while gap > stop and max(0.0, (1.0 - ell) * (1.0 + ell)) > stop:
        beta, shift, sqrt_shift, eta, gap, ell = _qdwh_params(ell, dtype)
        steps.append((beta, shift, sqrt_shift, eta))
    return steps


def _qdwh_scale(env, m, n, dtype):  # certified scale alpha and l0 <= sigma_min(a / alpha) from an envelope, preferring an exact power-of-two alpha
    if env.sigma_lo == env.sigma_hi:
        return env.sigma_hi, float(env.sigma_hi > 0.0)
    div_floor, div_round = _root_hi(m * n) * F(torch.finfo(dtype).tiny), _root_hi(n) * F(_gamma(1, dtype)) * F(min(env.sigma_hi, _FMAX))
    alpha = _hi((F(env.sigma_hi) + div_round) / (1 - div_floor)) if isfinite(env.sigma_hi) and div_floor < 1 else inf
    if not isfinite(alpha):
        if 0.0 < env.sigma_hi <= float(torch.finfo(dtype).max):
            return env.sigma_hi, 0.0
        raise RuntimeError("could not certify finite QDWH scale")
    l0 = _lo((F(env.sigma_lo) - div_round) / F(alpha), div_floor)
    dyadic = inf if (shift := (e := frexp(_hi(F(env.sigma_hi) / (1 - div_floor))))[1] - (e[0] == 0.5)) > 1023 else ldexp(1.0, shift)
    if env.sigma_lo > 0.0 and dyadic <= float(torch.finfo(dtype).max) and (cand := _lo(F(env.sigma_lo) / F(dyadic), div_floor)) > l0:
        alpha, l0 = dyadic, cand  # dividing by a power of two is exact, so it certifies a larger l0: prefer it
    return alpha, min(l0, 1.0)


def _qdwh_work(lo, hi, n, dtype):  # flop model of the schedule an envelope [lo, hi] buys; refinements must beat their own cost in this model
    ok = lo >= 0.0 and hi > 0.0 and isfinite(hi)
    return len(_qdwh_schedule(_qdwh_scale(SpectralEnvelope(lo, hi), n, n, dtype)[1], dtype)) * (16.0 * n**3 / 3.0 + n * n) if ok else inf


def _auto_reduce(m, n, dtype):  # reduce tall m x n to square iff geqrf + the square schedule costs fewer flops than iterating on the tall matrix
    def work(d, gn):
        return sum(4.0 * d * n * n + d * n + (4.0 if shift <= gn else 1.0) * n**3 / 3.0 for _, shift, _, _ in _qdwh_schedule(0.0, dtype))

    return 19.0 * n**3 / 3.0 + work(n, _gamma(n, dtype)) < work(m, _gamma(m, dtype))


def _inverse_gram_norm24_hi(y, inv_abs):  # certified upper bound on ||Y||_2 from Frobenius moments of H = Y^T Y (mean + spread bound on lambda_max)
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


def _residual_plan(n, dtype, c_hi):  # panel width and grouping that minimize the certified accumulation error of the blocked L @ R residual
    blk, floor, best = min(n, 64), F(n * n) * F(torch.finfo(dtype).tiny), (inf, 1)
    if not isfinite(c_hi) or not isfinite(_gamma(panels := (n + blk - 1) // blk, dtype)):  # gamma is monotone in group count: one gate covers the loop
        return inf, blk, 1
    t, s = (1 + (gb := F(_gamma(blk, dtype)))) * F(c_hi), isqrt(panels) + 2  # hoisted: error(group) = gb c + floor + gg t + gh (1 + (1 + gg) t)
    for group in sorted({*range(1, s), *((panels + v - 1) // v for v in range(1, s))}):  # monotone gamma: ceil(panels / ceil(panels / group)) dominates group
        gg, gh = F(_gamma(group, dtype)), F(_gamma((panels + group - 1) // group, dtype))
        best = min(best, (gg * t + gh * (1 + (1 + gg) * t), group))
    return _hi(best[0] + gb * F(c_hi) + floor), blk, best[1]


def _triangular_inverse_lower(r, norm_err, sigma_hi, base_lo, sigma_ub):  # certified sigma_min lower bound via Y ~ R^-1 with verified residual; 0.0 = declined
    try:
        y = torch.linalg.solve_triangular(r, torch.eye(n := r.shape[0], dtype=r.dtype, device=r.device), upper=True)
    except RuntimeError:
        return 0.0
    ylo, yhi = _sum_sq(y)
    if not isfinite(inv_bound := min(yhi, _prod_norm_hi(y))) or inv_bound <= 0.0 or not isfinite(rsq := _sum_sq(r)[1]):
        return 0.0  # an overflowed or vacuous norm certifies nothing: decline
    inv_hi = _root_hi(inv_bound)
    if _qdwh_work(max(base_lo, _lo(1 / inv_hi, norm_err)), sigma_hi, n, r.dtype) + n**3 >= _qdwh_work(base_lo, sigma_hi, n, r.dtype):
        return 0.0  # even a perfect residual could not shorten the schedule by more than the verification itself costs
    plans = [(*_residual_plan(n, r.dtype, min(_prod_norm_hi(lhs, rhs), _hi(_root_hi(F(rsq) * F(yhi))))), lhs, rhs) for lhs, rhs in ((y, r), (r, y))]
    if not (plan := min(plans, key=lambda p: p[0]))[0] < 1.0:
        return 0.0
    (berr, blk, group, left, right), e, tmp = plan, -torch.eye(n, dtype=r.dtype, device=r.device), torch.empty_like(r)
    for j in range(0, n, blk * group):  # E = L @ R - I accumulated in grouped panels per the plan, keeping the accumulation error certified
        tmp.zero_()
        for k in range(j, min(n, j + blk * group), blk):
            tmp.addmm_(left[:, k : k + blk], right[k : k + blk])
        e.add_(tmp)
    slack = 1 - _root_hi(esq) - F(berr) if isfinite(esq := _sum_sq(e)[1]) else F(-1)  # 1 - eta; a negative slack fails every bound below closed
    lower, best = _lo(slack / inv_hi, norm_err), min(sigma_ub, _lo(slack / _root_hi(F(ylo) / n), norm_err))
    if _qdwh_work(best, sigma_hi, n, r.dtype) + 2.0 * n**3 >= _qdwh_work(lower, sigma_hi, n, r.dtype):
        return lower  # the norm24 refinement below would cost more than the schedule steps it could save
    return lower if not isfinite(g24 := _inverse_gram_norm24_hi(y, inv_bound)) else max(lower, _lo(slack / F(g24), norm_err))


def spectral_envelope(a, triangular=False):  # certified enclosure [sigma_lo, sigma_hi] of the extreme singular values of a
    torch.backends.cuda.matmul.allow_tf32 = False
    amax = float(a.abs().max())
    if amax == 0.0 or a.numel() == 1:
        return SpectralEnvelope(amax, amax)
    scale = inf if (shift := (e := frexp(amax))[1] - (e[0] == 0.5)) > 1023 else ldexp(1.0, shift)  # smallest power of two >= amax: dividing by it is exact
    scale = scale if (exact := scale <= float(torch.finfo(a.dtype).max)) else amax  # not representable: divide by amax, charge the rounding to norm_err
    tlo, thi = _sum_sq(x := a / scale)  # finite by construction: every entry of x is at most 1
    mu = 0.5 * (tlo + thi) / (gx := (x.mT if x.shape[1] > x.shape[0] else x).contiguous()).shape[1]
    h, e = (gx.mT @ gx).to(torch.float64, copy=True), F(_gamma(1, torch.float64))
    h.diagonal().sub_(mu)
    hi2, lo2 = min(thi, abs_norm := _prod_norm_hi(gx)), F(0)
    if isfinite(abs_norm) and isfinite(ge := _gamma(gx.shape[0], x.dtype)):  # moment bound: every gram eigenvalue lies within radius of the center mu
        radius = _root_hi(_sum_sq(h)[1]) + e * _root_hi(_sum_sq(h.diagonal().abs() + 2.0 * abs(mu))[1]) / (1 - e) + F(ge) * F(abs_norm)
        hi2, lo2 = min(hi2, _hi(F(mu) + radius)), max(F(0), F(mu) - radius)
    norm_err = _root_hi(x.numel()) * F(torch.finfo(x.dtype).tiny) + (0 if exact else F(_gamma(1, x.dtype)) * _root_hi(thi))
    sigma_lo, sigma_hi = _lo(lo2 / _root_hi(lo2), norm_err) if lo2 else 0.0, _hi(_root_hi(hi2), norm_err)  # z / root_hi(z) <= sqrt(z), certified
    if triangular and x.dtype == torch.float32 and x.shape[0] == x.shape[1] and (dmin := float(x.diagonal().abs().min())) > 0.0:
        sigma_ub = _lo(dmin, norm_err)  # sigma_min <= min |R_ii| caps what any inverse-based refinement can certify
        if _qdwh_work(max(sigma_lo, sigma_ub), sigma_hi, x.shape[0], x.dtype) + x.shape[0] ** 3 < _qdwh_work(sigma_lo, sigma_hi, x.shape[0], x.dtype):
            sigma_lo = max(sigma_lo, _triangular_inverse_lower(x.contiguous(), norm_err, sigma_hi, sigma_lo, sigma_ub))
    return SpectralEnvelope(_lo(F(sigma_lo) * F(scale)) if sigma_lo > 0.0 else 0.0, min(_hi(F(sigma_hi) * F(scale)), _hi(F(amax) * _root_hi(a.numel()))))


def qdwh_polar(a, *, triangular=False):  # QDWH polar factor W of a, plus the scaled input x0 and core = W^T x0, with certificates in info
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.ndim != 2 or 0 in a.shape or a.shape[0] < a.shape[1] or a.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(a).all()):
        raise ValueError("expected a tall float32/float64 matrix")
    m, n, dtype = *a.shape, a.dtype
    alpha, l0 = _qdwh_scale(spectral_envelope(a, triangular), m, n, dtype)
    steps = [] if alpha == 0.0 else _qdwh_schedule(l0, dtype)
    x = x0 = torch.zeros_like(a) if alpha == 0.0 else (a.double() / alpha if alpha > float(torch.finfo(dtype).max) else a / alpha).to(dtype).contiguous()
    routes, qr_work, gram_noise = {"qr": 0, "chol": 0}, None, _gamma(m, dtype)
    for beta, shift, sqrt_shift, eta in steps:
        if shift > gram_noise:  # the Cholesky route needs the shift to certifiably dominate the gram's rounding noise
            qr_work = None
            gram = 0.5 * ((gram := x.mT @ x) + gram.mT)
            gram.diagonal().add_(shift)
            chol, fail = torch.linalg.cholesky_ex(gram)
            if not bool(fail.any()):
                x = beta * x + eta * torch.cholesky_solve(x.mT, chol).mT
                routes["chol"] += 1
                continue
        if qr_work is None:  # QR route: orthogonalize the stacked [x; sqrt_shift * I]
            qr_work = torch.empty((m + n, n), dtype=dtype, device=x.device)
            qr_work[m:].zero_()
        qr_work[:m] = x
        qr_work[m:].diagonal().fill_(sqrt_shift)
        q, _ = torch.linalg.qr(qr_work)
        x = torch.addmm(x, q[:m], q[m:].mT, beta=beta, alpha=eta / sqrt_shift)
        routes["qr"] += 1
    core = x.mT @ x0
    skew = inf if not isfinite(csq := _sum_sq(core - core.mT)[1]) else _hi(_root_hi(csq) / 2)  # inf skew fails the eigh trust gate closed
    skew_budget = _hi(F(min(_gamma(x0.shape[0], x0.dtype), _FMAX)) * _root_hi(_sum_sq(x0)[1]))  # legitimate skew of an exactly-orthogonal factor
    info = dict(alpha=alpha, l0=l0, iters=len(steps), routes=routes, core_skew=skew, skew_budget=skew_budget)
    return x, x0, core, info


def _finish(w, x0, core, info):  # (u, s, vh) from the polar core: eigh when the certified skew allows it, dense SVD of the core otherwise
    if skewed := info["core_skew"] > info["skew_budget"]:
        vc, s_unit, vh = torch.linalg.svd(core, full_matrices=False)
    else:
        e, v = torch.linalg.eigh(0.5 * (core + core.mT))
        vc, s_unit = torch.flip(v, [1]), torch.flip(e, [0]).clamp_min(0)
        vh = vc.mT.contiguous()
    info["core"] = "svd" if skewed else "eigh"
    rank, trusted = int((s_unit > 0.0).sum()), info["l0"] > info["core_skew"]
    s = s_unit.double() * float(info["alpha"])
    s = s if x0.dtype == torch.float32 and bool((s > float(torch.finfo(x0.dtype).max)).any()) else s.to(x0.dtype)  # keep fp64 s only if fp32 would overflow
    u = w @ vc[:, :rank] if trusted else (x0 @ vh[:rank].mT) / s_unit[:rank]  # untrusted W: rebuild U from x0, then re-orthogonalize
    if rank and not trusted:
        u, r = torch.linalg.qr(u)
        u = u * torch.where(r.diagonal() < 0.0, -1.0, 1.0).to(u.dtype)
    if rank < vh.shape[0]:  # complete a rank-deficient U to a full orthonormal basis
        q, _ = torch.linalg.qr(torch.cat((u, torch.eye(x0.shape[0], vh.shape[0], dtype=x0.dtype, device=x0.device)), dim=1))
        u = torch.cat((u, q[:, rank : vh.shape[0]]), dim=1)
    return u, s, vh


def svd(a, *, reduce_tall="auto"):  # certified SVD: optional QR reduction to square, QDWH polar factor, symmetric-core finish
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.ndim != 2 or 0 in a.shape or a.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(a).all()):
        raise ValueError("expected a finite float32/float64 matrix")
    m, n = a.shape
    if m < n:  # wide input: recurse on the transpose and swap the factors back
        return SVDResult((sub := svd(a.mT.contiguous(), reduce_tall=reduce_tall)).vh.mT.contiguous(), sub.s, sub.u.mT.contiguous(), sub.info)
    if m > n and (reduce_tall is True or (reduce_tall == "auto" and _auto_reduce(m, n, a.dtype))):
        hh, tau = torch.geqrf(a)
        r = torch.triu(hh[:n, :]).contiguous()
        if not bool(torch.isfinite(r).all()):  # geqrf overflowed: retry without the reduction
            return svd(a, reduce_tall=False)
        w, x0, core, info = qdwh_polar(r, triangular=True)
        ur, s, vh = _finish(w, x0, core, info)
        hi = 0.0 if float(a.abs().max()) == 0.0 else inf if not isfinite(sq := _sum_sq(a)[1]) else _hi(_root_hi(sq))
        info.update({"polar_alpha": info["alpha"], "polar_l0": info["l0"], "alpha": hi, "l0": 0.0, "reduced": True})
        return SVDResult(torch.ormqr(hh, tau, torch.cat((ur, hh.new_zeros((m - n, ur.shape[1]))))), s, vh, info)
    w, x0, core, info = qdwh_polar(a)
    return SVDResult(*_finish(w, x0, core, info), info | {"reduced": False})

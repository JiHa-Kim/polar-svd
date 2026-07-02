from collections import namedtuple
from math import exp, frexp, fsum, inf, isfinite, ldexp, log, log1p, nextafter as na, sqrt

import torch

SVDResult, SpectralEnvelope = namedtuple("SVDResult", "u s vh info"), namedtuple("SpectralEnvelope", "sigma_lo sigma_hi")


def _gamma(k, dtype):
    return inf if (x := k * 0.5 * float(torch.finfo(dtype).eps)) >= 1.0 else na(x / (1.0 - x), inf)


def _sum_sq(x):
    e, s = _gamma(x.numel() + 1, torch.float64), float((v := x.reshape(-1).double()) @ v)
    return (0.0, inf) if e >= 1.0 or not isfinite(s) else (max(0.0, na(s / (1.0 + e), 0.0)), na(s / (1.0 - e), inf))


def _prod_norm_hi(a, b, symmetric=False):
    if (err := _gamma(max(*a.shape, *b.shape), torch.float64)) >= 1.0:
        return inf
    b = b.abs()
    a = b.mT if symmetric else a.abs()  # symmetric contract: caller must pass a == b.mT (Gram |B|^T |B|); a is ignored on that path
    v = torch.ones(b.shape[1], dtype=b.dtype, device=b.device)
    if not isfinite(vmax := float((w := a @ (b @ v) if symmetric else b.mT @ (a.mT @ (a @ (b @ v)))).max())) or vmax <= 0.0:
        return inf
    v, b = (w / vmax).clamp_min(torch.finfo(b.dtype).tiny).double(), b.double()
    q = float(((b.mT @ (b @ v) if symmetric else b.mT @ ((a := a.double()).mT @ (a @ (b @ v)))) / v).max())
    return inf if not isfinite(q) or q < 0.0 else na((q if symmetric else sqrt(q)) / (1.0 - err) ** 2, inf)


def _inverse_gram_norm24_hi(y, inv_abs):
    n, dot, sym = y.shape[0], _gamma(y.shape[0], y.dtype), _gamma(2, y.dtype)
    if dot >= 1.0 or sym >= 1.0 or not isfinite(inv_abs):
        return inf
    floor = n * n * torch.finfo(y.dtype).tiny
    delta = na(fsum((dot * inv_abs, sym * (1.0 + dot) * inv_abs, floor)), inf)
    h = 0.5 * ((h := y.mT @ y) + h.mT)
    q1lo, q1hi = _sum_sq(h)
    q2root = na(sqrt(_sum_sq(h @ h)[1]) + na(dot * q1hi, inf) + floor, inf)
    q2hi = na(q2root * q2root, inf)
    q1sq_n_lo = na(na(na(q1lo * q1lo, 0.0) / n, 0.0), 0.0)
    spread_hi = na(na((n - 1.0) / n, inf) * na(max(0.0, na(q2hi - q1sq_n_lo, inf)), inf), inf)
    z_hi = na(na(q1hi / n, inf) + na(sqrt(spread_hi), inf), inf)
    return na(sqrt(na(sqrt(z_hi) + delta, inf)), inf)


def _qdwh_work(lo, hi, n, dtype):
    if not (lo >= 0.0 and hi > 0.0 and isfinite(hi)):
        return inf
    return len(_qdwh_schedule(_qdwh_scale(SpectralEnvelope(lo, hi), n, n, dtype)[1], dtype)) * (16.0 * n**3 / 3.0 + n * n)


def _residual_plan(n, dtype, c_hi):
    blk, floor = min(n, 64), n * n * torch.finfo(dtype).tiny
    gb, panels, best = _gamma(blk, dtype), (n + blk - 1) // blk, (inf, 1)
    for group in range(1, panels + 1):
        gg, gh = _gamma(group, dtype), _gamma((panels + group - 1) // group, dtype)
        err = na(fsum((gb * c_hi, gg * (1.0 + gb) * c_hi, gh * (1.0 + (1.0 + gb) * (1.0 + gg) * c_hi), floor)), inf)
        best = min(best, (err, group))
    return best[0], blk, best[1]


def _triangular_inverse_lower(r, norm_err, sigma_hi, base_lo, sigma_ub):
    try:
        y = torch.linalg.solve_triangular(r, torch.eye(n := r.shape[0], dtype=r.dtype, device=r.device), upper=True)
    except RuntimeError:
        return 0.0
    ysq = _sum_sq(y)
    inv_hi = na(sqrt(inv_bound := min(ysq[1], _prod_norm_hi(y.mT, y, True))), inf)
    if _qdwh_work(max(base_lo, max(0.0, na(1.0 / inv_hi - norm_err, 0.0))), sigma_hi, n, r.dtype) + n**3 >= _qdwh_work(base_lo, sigma_hi, n, r.dtype):
        return 0.0
    fro_c = na(sqrt(na(_sum_sq(r)[1] * ysq[1], inf)), inf)
    plans = [(*_residual_plan(n, r.dtype, min(_prod_norm_hi(a, b), fro_c)), a, b) for a, b in ((y, r), (r, y))]
    if not (plan := min(plans, key=lambda p: p[0]))[0] < 1.0:
        return 0.0
    berr, blk, group, left, right = plan
    e, tmp = -torch.eye(n, dtype=r.dtype, device=r.device), torch.empty_like(r)
    for j in range(0, n, blk * group):
        tmp.zero_()
        for k in range(j, min(n, j + blk * group), blk):
            tmp.addmm_(left[:, k : k + blk], right[k : k + blk])
        e.add_(tmp)
    eta = na(sqrt(_sum_sq(e)[1]) + berr, inf)
    lower = max(0.0, na((1.0 - eta) / inv_hi - norm_err, 0.0))
    best = min(sigma_ub, max(0.0, na((1.0 - eta) / sqrt(ysq[0] / n) - norm_err, 0.0)))
    if _qdwh_work(best, sigma_hi, n, r.dtype) + 2.0 * n**3 >= _qdwh_work(lower, sigma_hi, n, r.dtype):
        return lower
    return max(lower, max(0.0, na((1.0 - eta) / _inverse_gram_norm24_hi(y, inv_bound) - norm_err, 0.0)))


def spectral_envelope(a, triangular=False):
    torch.backends.cuda.matmul.allow_tf32 = False
    amax = float(a.abs().max())
    if amax == 0.0 or a.numel() == 1:
        return SpectralEnvelope(amax, amax)
    fmax = float(torch.finfo(a.dtype).max)
    scale = inf if (shift := (e := frexp(amax))[1] - (e[0] == 0.5)) > 1023 else ldexp(1.0, shift)
    scale = scale if (exact := scale <= fmax) else amax
    x = a / scale
    tlo, thi = _sum_sq(x)
    gx = (x.mT if x.shape[1] > x.shape[0] else x).contiguous()
    mu = 0.5 * (tlo + thi) / gx.shape[1]
    gram_err = na(_gamma(gx.shape[0], x.dtype) * (abs_norm := _prod_norm_hi(gx.mT, gx, True)), inf)
    h, e = (gx.mT @ gx).to(torch.float64, copy=True), _gamma(1, torch.float64)
    h.diagonal().sub_(mu)
    radius = na(sqrt(_sum_sq(h)[1]) + e * sqrt(_sum_sq(h.diagonal().abs() + 2.0 * abs(mu))[1] / ((1.0 - e) * (1.0 - e))), inf)
    moment_ok = isfinite(mu) and isfinite(radius) and isfinite(gram_err)
    hi2 = min(thi, abs_norm, na(fsum((mu, radius, gram_err)), inf) if moment_ok else inf)
    lo2 = max(0.0, na(fsum((mu, -radius, -gram_err)), 0.0)) if moment_ok else 0.0
    norm_err = na(na(sqrt(x.numel()) * torch.finfo(x.dtype).tiny, inf) + (0.0 if exact else na(_gamma(1, x.dtype) * sqrt(thi), inf)), inf)
    sigma_lo = max(0.0, na(sqrt(max(0.0, lo2)) - norm_err, 0.0))
    sigma_hi = na(sqrt(hi2) + norm_err, inf)
    if triangular and x.dtype == torch.float32 and x.shape[0] == x.shape[1] and (dmin := float(x.diagonal().abs().min())) > 0.0:
        sigma_ub = max(0.0, na(dmin - norm_err, 0.0))
        if _qdwh_work(max(sigma_lo, sigma_ub), sigma_hi, x.shape[0], x.dtype) + x.shape[0] ** 3 < _qdwh_work(sigma_lo, sigma_hi, x.shape[0], x.dtype):
            sigma_lo = max(sigma_lo, _triangular_inverse_lower(x.contiguous(), norm_err, sigma_hi, sigma_lo, sigma_ub))
    frob_hi = na(amax * na(sqrt(a.numel()), inf), inf)
    if not isfinite(frob_hi):
        p, q, m, s = *amax.as_integer_ratio(), *fmax.as_integer_ratio()
        frob_hi = fmax if p * p * a.numel() * s * s <= m * m * q * q else inf
    return SpectralEnvelope(na(sigma_lo * scale, 0.0) if sigma_lo > 0.0 else 0.0, min(na(sigma_hi * scale, inf), frob_hi))


def _qdwh_params(ell, dtype):
    tiny = float(torch.finfo(dtype).tiny)
    z = zmin = 3.0 * (tiny + 2.0 * sqrt(tiny * (1.0 + tiny))) / (4.0 + 3.0 * tiny)
    if ell > 0.0:
        log16, target = log(16.0), 2.0 * log(min(float(ell), 1.0))
        if target > log16 + 3.0 * log(zmin) - 3.0 * log(3.0 - zmin) - log1p(zmin):
            lo, hi = log(zmin), 0.0
            while na(lo, hi) < hi:
                u = exp(mid := 0.5 * (lo + hi))
                lo, hi = (mid, hi) if log16 + 3.0 * log(u) - 3.0 * log(3.0 - u) - log1p(u) <= target else (lo, mid)
            z = max(zmin, na(exp(lo), 0.0))
    op, om, pp, v = 1.0 + z, 3.0 - z, 3.0 + z, 1.0 - z
    beta, shift, eta = om / (3.0 * op), 4.0 * z * z / (3.0 * om * op), 4.0 * z * pp * pp / (9.0 * om * op * op)
    gap = na(pp * v * v * v / (om * op * op * op), inf)
    ell_next = na(1.0 - gap / (1.0 + sqrt(max(0.0, 1.0 - gap))), 0.0) if gap < 0.5 else na(sqrt(max(0.0, na(16.0 * z / (om * op * op * op), 0.0))), 0.0)
    return beta, shift, sqrt(shift), eta, gap, ell_next


def _qdwh_schedule(l0, dtype, stop=None):
    stop = (tol := 0.5 * float(torch.finfo(dtype).eps)) * (2.0 - tol) if stop is None else stop
    ell, gap, steps = 0.0 if l0 < sqrt(torch.finfo(dtype).tiny) else na(l0, 0.0), inf, []
    while gap > stop and max(0.0, (1.0 - ell) * (1.0 + ell)) > stop:
        beta, shift, sqrt_shift, eta, gap, ell = _qdwh_params(ell, dtype)
        steps.append((beta, shift, sqrt_shift, eta))
    return steps


def _auto_reduce(m, n, dtype):
    def work(d, gn):
        return sum(4.0 * d * n * n + d * n + (4.0 if shift <= gn else 1.0) * n**3 / 3.0 for _, shift, _, _ in _qdwh_schedule(0.0, dtype))

    return 19.0 * n**3 / 3.0 + work(n, _gamma(n, dtype)) < work(m, _gamma(m, dtype))


def _qdwh_scale(env, m, n, dtype):
    if env.sigma_lo == env.sigma_hi:
        return env.sigma_hi, float(env.sigma_hi > 0.0)
    div_floor = na(sqrt(m * n) * torch.finfo(dtype).tiny, inf)
    div_round = na(sqrt(n) * na(_gamma(1, dtype) * env.sigma_hi, inf), inf)
    den = na(1.0 - div_floor, 0.0)
    alpha = na(na(env.sigma_hi + div_round, inf) / den, inf)
    if not isfinite(alpha) or alpha <= 0.0:
        if 0.0 < env.sigma_hi <= float(torch.finfo(dtype).max):
            return env.sigma_hi, 0.0
        raise RuntimeError("could not certify finite QDWH scale")
    l0 = na(na(env.sigma_lo / alpha, 0.0) - na(div_round / alpha + div_floor, inf), 0.0)
    dyadic = inf if (shift := (e := frexp(na(env.sigma_hi / den, inf)))[1] - (e[0] == 0.5)) > 1023 else ldexp(1.0, shift)
    if env.sigma_lo > 0.0 and dyadic <= float(torch.finfo(dtype).max) and (cand := na(na(env.sigma_lo / dyadic, 0.0) - div_floor, 0.0)) > l0:
        alpha, l0 = dyadic, cand
    return alpha, min(max(0.0, l0), 1.0)


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
    skew = na(0.5 * sqrt(_sum_sq(core - core.mT)[1]), inf)
    skew_budget = na(_gamma(x0.shape[0], x0.dtype) * sqrt(_sum_sq(x0)[1]), inf)
    info = dict(alpha=alpha, l0=l0, iters=len(steps), routes=routes, core_skew=skew, skew_budget=skew_budget)
    return x, x0, core, info


def _finish(w, x0, core, info):
    if skewed := info["core_skew"] > info["skew_budget"]:
        vc, s_unit, vh = torch.linalg.svd(core, full_matrices=False)
    else:
        e, v = torch.linalg.eigh(0.5 * (core + core.mT))
        vc, s_unit = torch.flip(v, [1]), torch.flip(e, [0]).clamp_min(0)
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
    if a.shape[0] < a.shape[1]:
        sub = svd(a.mT.contiguous(), reduce_tall=reduce_tall)
        return SVDResult(sub.vh.mT.contiguous(), sub.s, sub.u.mT.contiguous(), sub.info)
    if a.shape[0] > a.shape[1] and (reduce_tall is True or (reduce_tall == "auto" and _auto_reduce(*a.shape, a.dtype))):
        hh, tau = torch.geqrf(a)
        r = torch.triu(hh[: a.shape[1], :]).contiguous()
        if not bool(torch.isfinite(r).all()):
            return svd(a, reduce_tall=False)
        w, x0, core, info = qdwh_polar(r, triangular=True)
        ur, s, vh = _finish(w, x0, core, info)
        hi = 0.0 if float(a.abs().max()) == 0.0 else na(sqrt(_sum_sq(a)[1]), inf)
        info.update({"polar_alpha": info["alpha"], "polar_l0": info["l0"], "alpha": hi, "l0": 0.0, "reduced": True})
        return SVDResult(torch.ormqr(hh, tau, torch.cat((ur, hh.new_zeros((hh.shape[0] - hh.shape[1], ur.shape[1]))))), s, vh, info)
    w, x0, core, info = qdwh_polar(a)
    return SVDResult(*_finish(w, x0, core, info), info | {"reduced": False})

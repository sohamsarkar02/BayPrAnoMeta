import torch
import math

KAPPA0 = 0.01
NU0_OFFSET = 2
LAMBDA0_SCALE = 1.0

def niw_posterior(z, Lambda0=None, validate=False):
    """Return NIW predictive parameters; ``Sigma_n`` is Student-t scale, not Lambda_n."""
    K, d = z.shape
    device, dtype = z.device, z.dtype

    mu0 = torch.zeros(d, device=device, dtype=dtype)
    if Lambda0 is None:
        Lambda0 = LAMBDA0_SCALE * torch.eye(d, device=device, dtype=dtype)
    else:
        if Lambda0.shape != (d, d):
            raise ValueError(f"Lambda0 must have shape {(d, d)}, got {tuple(Lambda0.shape)}")
        Lambda0 = Lambda0.to(device=device, dtype=dtype)

    z_bar = z.mean(dim=0)
    S = (z - z_bar).T @ (z - z_bar)

    kappa_n = KAPPA0 + K
    nu_n = d + NU0_OFFSET + K

    mu_n = (KAPPA0 * mu0 + K * z_bar) / kappa_n
    diff = (z_bar - mu0).unsqueeze(1)

    Lambda_n = Lambda0 + S + (KAPPA0 * K / kappa_n) * (diff @ diff.T)

    dof = nu_n - d + 1
    Sigma_n = ((kappa_n + 1) / (kappa_n * dof)) * Lambda_n

    if validate:
        # Debug-only numerical checks.  They are intentionally opt-in because
        # repeated eigendecompositions would slow full episodic experiments.
        assert torch.isfinite(Lambda_n).all() and torch.isfinite(Sigma_n).all()
        assert dof > 0
        assert torch.allclose(Sigma_n, Sigma_n.T, atol=1e-6, rtol=1e-5)
        torch.linalg.cholesky(Sigma_n + 1e-6 * torch.eye(d, device=device, dtype=dtype))

    return mu_n, Sigma_n, dof

def log_student_t(x, mu, Sigma, dof):
    d = x.shape[1]
    device, dtype = x.device, x.dtype

    if not torch.is_tensor(dof):
        dof = torch.tensor(dof, device=device, dtype=dtype)
    else:
        dof = dof.to(device=device, dtype=dtype)

    d_t = torch.tensor(float(d), device=device, dtype=dtype)

    xm = x - mu.unsqueeze(0)

    jitter = 1e-6 * torch.eye(d, device=device, dtype=dtype)
    L = torch.linalg.cholesky(Sigma + jitter)
    invSigma = torch.cholesky_inverse(L)
    logdet = 2 * torch.sum(torch.log(torch.diag(L)))

    delta = torch.sum((xm @ invSigma) * xm, dim=1)

    return (
        torch.lgamma((dof + d_t) / 2)
        - torch.lgamma(dof / 2)
        - 0.5 * d_t * torch.log(dof * torch.tensor(math.pi, device=device))
        - 0.5 * logdet
        - ((dof + d_t) / 2) * torch.log1p(delta / dof)
    )


def matched_gaussian_covariance(Sigma_n, dof, validate=False):
    """Match the covariance of Student-t(mu, Sigma_n, dof), requiring dof > 2."""
    dof = torch.as_tensor(dof, device=Sigma_n.device, dtype=Sigma_n.dtype)
    if not bool((dof > 2).item()):
        raise ValueError(f"Matched Gaussian covariance requires dof > 2, got {dof.item()}")
    Sigma_G = (dof / (dof - 2.0)) * Sigma_n
    if validate:
        assert torch.isfinite(Sigma_G).all()
        torch.linalg.cholesky(
            Sigma_G + 1e-6 * torch.eye(Sigma_G.shape[0], device=Sigma_G.device, dtype=Sigma_G.dtype)
        )
    return Sigma_G


def log_gaussian_matched(x, mu, Sigma_n, dof, validate=False):
    """Differentiable Gaussian log density with covariance dof/(dof-2)*Sigma_n."""
    d = x.shape[1]
    Sigma_G = matched_gaussian_covariance(Sigma_n, dof, validate=validate)
    jitter = 1e-6 * torch.eye(d, device=x.device, dtype=x.dtype)
    L = torch.linalg.cholesky(Sigma_G + jitter)
    centered = (x - mu.unsqueeze(0)).T
    whitened = torch.linalg.solve_triangular(L, centered, upper=False)
    mahalanobis = torch.sum(whitened.square(), dim=0)
    logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))
    constant = torch.as_tensor(d * math.log(2.0 * math.pi), device=x.device, dtype=x.dtype)
    logp = -0.5 * (constant + logdet + mahalanobis)
    if validate:
        assert torch.isfinite(logp).all()
    return logp

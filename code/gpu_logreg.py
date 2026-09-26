import os, sys, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def fit_logreg_gpu(X, y, C=1.0, max_iter=1000, tol=1e-6, dtype="float64", verbose=True,
                   W0=None, b0=None, warm=False):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    td = torch.float64 if dtype == "float64" else torch.float32
    n, d = X.shape
    K = int(y.max()) + 1
    Xt = torch.as_tensor(X, dtype=td, device=dev)
    yt = torch.as_tensor(y, dtype=torch.long, device=dev)

    if warm and td is torch.float64 and W0 is None:
        W0, b0 = fit_logreg_gpu(X, y, C=C, max_iter=max_iter, tol=tol,
                                dtype="float32", verbose=False, warm=False)
    if W0 is None:
        W = torch.zeros((K, d), dtype=td, device=dev, requires_grad=True)
        b = torch.zeros(K, dtype=td, device=dev, requires_grad=True)
    else:
        W = torch.as_tensor(W0, dtype=td, device=dev).clone().requires_grad_(True)
        b = torch.as_tensor(b0, dtype=td, device=dev).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([W, b], max_iter=max_iter, tolerance_grad=tol,
                            tolerance_change=0.0, history_size=10,
                            line_search_fn="strong_wolfe")
    lossfn = torch.nn.CrossEntropyLoss(reduction="sum")
    t0 = time.time()
    state = {"n": 0}

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = 0.5 * (W * W).sum() + C * lossfn(Xt @ W.T + b, yt)
        loss.backward()
        state["n"] += 1
        if verbose and state["n"] % 50 == 0:
            print(f"      lbfgs {state['n']}  loss={loss.item():.6e}  "
                  f"{time.time()-t0:.0f}s", end="\r")
        return loss

    opt.step(closure)
    if verbose:
        print(f"      lbfgs {state['n']} 회 평가, {time.time()-t0:.0f}s")
    return (W.detach().cpu().numpy(), b.detach().cpu().numpy())


class GPUProbe:

    def __init__(self, W, b):
        self.W, self.b = W, b

    def decision_function(self, Q):
        return Q @ self.W.T + self.b

    def score(self, X, y):
        return float((self.decision_function(X).argmax(1) == y).mean())


def fit(X, y, C=1.0, max_iter=1000, verbose=True):
    return GPUProbe(*fit_logreg_gpu(X, y, C=C, max_iter=max_iter, verbose=verbose))


class SklearnCompatGPU:

    def __init__(self, max_iter=1000, C=1.0, **kw):
        self.max_iter, self.C = max_iter, C

    def fit(self, X, y):
        self.coef_, self.intercept_ = fit_logreg_gpu(
            X, y, C=self.C, max_iter=self.max_iter, verbose=False)
        self.classes_ = np.arange(self.coef_.shape[0])
        return self

    def decision_function(self, Q):
        return Q @ self.coef_.T + self.intercept_

    def predict(self, Q):
        return self.decision_function(Q).argmax(1)

    def score(self, X, y):
        return float((self.predict(X) == y).mean())


def install():
    import sklearn.linear_model as sklm
    if getattr(sklm, "_gpu_patched", False):
        return
    sklm._orig_LogisticRegression = sklm.LogisticRegression
    sklm.LogisticRegression = SklearnCompatGPU
    sklm._gpu_patched = True
    print("  [gpu_logreg] sklearn.linear_model.LogisticRegression -> GPU 판으로 교체됨")


def logit_scores(probe, Q, odin_T):
    lg = probe.decision_function(Q)
    mx = lg.max(1, keepdims=True)
    p = np.exp(lg - mx); p /= p.sum(1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(lg - mx).sum(1))
    lt = lg / odin_T
    mt = lt.max(1, keepdims=True)
    pt = np.exp(lt - mt); pt /= pt.sum(1, keepdims=True)
    return dict(energy=-lse, msp=-p.max(1), maxlogit=-lg.max(1),
                entropy=-(p * np.log(p + 1e-12)).sum(1), odin=-pt.max(1)), lg


def verify(bench):
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import spearmanr
    import derma_probe as DP
    import in200_probe as IP

    rng = np.random.default_rng(0)
    groups, refs = IP.build("dino", rng, bench)
    Q = np.vstack([groups[g] for g in groups])
    G = np.concatenate([np.full(len(groups[g]), g) for g in groups])
    X = np.vstack(refs)
    y = np.concatenate([np.full(len(R), i) for i, R in enumerate(refs)])
    print(f"  {bench}: X={X.shape} K={len(refs)} Q={len(Q)}")

    t = time.time()
    sk = LogisticRegression(max_iter=1000, C=1.0).fit(X, y)
    t_sk = time.time() - t
    print(f"  [sklearn] {t_sk:.0f}s  train acc={sk.score(X, y):.4f}")

    t = time.time()
    gp = fit(X, y, C=1.0, max_iter=1000)
    t_gp = time.time() - t
    print(f"  [gpu]     {t_gp:.0f}s  train acc={gp.score(X, y):.4f}   "
          f"** {t_sk/max(t_gp,1e-9):.1f}x 빠름 **")

    s_sk, lg_sk = logit_scores(sk, Q, DP.CFG["odin_T"])
    s_gp, lg_gp = logit_scores(gp, Q, DP.CFG["odin_T"])
    print(f"\n  계수: max|W_sk - W_gp| = {np.abs(sk.coef_ - gp.W).max():.3e}   "
          f"logit max|diff| = {np.abs(lg_sk - lg_gp).max():.3e}")

    def within(x):
        o = np.asarray(x, float).copy()
        for g in np.unique(G):
            o[G == g] -= o[G == g].mean()
        return o

    print(f"\n  {'score':>10} {'rho(sk,gpu)':>12} {'max|diff|':>12}  판정")
    worst_rho = 1.0
    for k in s_sk:
        r = float(spearmanr(s_sk[k], s_gp[k]).statistic)
        d = float(np.abs(within(s_sk[k]) - within(s_gp[k])).max())
        worst_rho = min(worst_rho, r)
        print(f"  {k:>10} {r:12.6f} {d:12.3e}  {'O' if r > 0.9999 else '** X **'}")
    print(f"\n  [verify] 최소 순위상관 {worst_rho:.6f}  ->  "
          f"{'통과 (표의 모든 숫자가 같다)' if worst_rho > 0.9999 else '** 실패 **'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", default="in200")
    a = ap.parse_args()
    if a.verify:
        verify(a.bench)

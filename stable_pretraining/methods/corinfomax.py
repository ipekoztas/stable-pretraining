"""CorInfoMax: Correlative Information Maximization (work in progress).

Self-supervised learning that maximizes the log-determinant of
exponential-moving-average estimates of each branch's projection
covariance (the "big-bang" term), while an MSE term aligns the two
L2-normalized views:

    L = -(logdet(R1 + eps*I) + logdet(R2 + eps*I)) / D
        + alpha * MSE(z1, z2)

Planned API (not implemented yet):
    - ``stable_pretraining.losses.CorInfoMaxLoss``
    - ``stable_pretraining.forward.corinfomax``
    - ``stable_pretraining.methods.CorInfoMax``

References:
    Ozsoy, Hamdan, Arik, Yuret, Erdogan. "Self-Supervised Learning with an
    Information Maximization Criterion." NeurIPS 2022.
    https://arxiv.org/abs/2209.07999

    Reference implementation: https://github.com/serdarozsoy/corinfomax-ssl
"""

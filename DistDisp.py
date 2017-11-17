def distdisp(squigly, moments, mu_0):
    import numpy as np
    from div0 import div0
    N = len(squigly)
    k_m = mu_0 / (4 * np.pi)
    dist = np.sqrt(np.sum(np.multiply(squigly, squigly), axis=1)[:, np.newaxis, :])  # Distance between particles
    disp = np.multiply(squigly, div0(1, dist))  # Normed displacement vectors
    m_i = np.repeat(np.swapaxes(moments[:, :, np.newaxis], 0, 2),N ,axis=0)  # (N, 3, N)-tensor with copies of i'moment in i'th layer
    m_j = np.repeat(moments[:, :, np.newaxis], N, axis=2)  # (N, 3, N)-tensor with copy of moments in each layer
    return dist, disp, k_m, m_i, m_j

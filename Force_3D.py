def calc_force(dist, disp, k_m, m_i, m_j, mu_0):
    import numpy as np
    from div0 import div0
    A = np.sum(np.multiply(m_j, disp), axis=1)[:, np.newaxis, :]  # Dot product of m_j and r_hat
    B = np.sum(np.multiply(m_i, disp), axis=1)[:, np.newaxis, :]  # Dot product of m_i and r_hat
    C = np.sum(np.multiply(m_i, m_j), axis=1)[:, np.newaxis, :]*disp
    Force =  div0(k_m, dist**4) * (A * m_i + B * m_j + C - 5 * A * B * disp)
    Force = np.sum(Force, axis=0).transpose()
    return Force
# from DistDisp import distdisp
# from Squiglys import squiglys
# pos, vel, m = np.array([[-2,0,0], [2,0,0]]),  np.array([[0,0,0], [0,0,0]]),  np.array([[1,0, 0], [1,0,0]])
# mu_0 = 4*np.pi
# Sq = squiglys(pos)
# dist, disp, k_m, m_i, m_j = distdisp(Sq, m, mu_0)
# F = calc_force(dist, disp, k_m, m_i, m_j, mu_0)

def torque(dist, disp, m_i, m_j, mu_0, N):
    # Squigly is a (N, 3, N)-tensor containg all displacement vectors as row vectors.
    # moments is a (N, 3)-matrix containing all dipole moments as row vectors
    import numpy as np
    from div0 import div0
    A = np.sum(np.multiply(disp, m_j), axis=1)[:,np.newaxis,:]                  #Dot product of moments and diplacements
    B = np.cross(m_i,disp,axis=1)                                               #Cross product of each corresponding row in the 2 tensors
    C = np.cross(m_i,m_j,axis=1)
    k_m = mu_0 / (4 * np.pi)
    torques = div0(k_m, dist**3) * (np.multiply(A,B) - C)                            #The torque on every particle from every other particle
    torques[np.isnan(torques)] = 0                                              #Removes undefined elements, resulting from division by 0
    result = np.sum(torques, axis = 0)                                          #(3,N)-matrix with summed torque on every particle
    return result.transpose() #Summed torque on every particle

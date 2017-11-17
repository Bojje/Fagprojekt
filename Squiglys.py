def squiglys(pos):
    import numpy as np
    N = len(pos)
    r_j = np.repeat(pos[:,:,np.newaxis],N,axis=2)
    r_i = np.swapaxes(pos[:, :, np.newaxis], 0, 2)
    return r_i - r_j

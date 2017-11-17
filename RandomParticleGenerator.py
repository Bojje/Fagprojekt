def RandomParticleGenerator(vec_num):
    import numpy as np
    ran_mat = lambda low, high, size: np.random.randint(low, high, size=size)
    names = ["pos", "vel", "m"]
    for i in names:
        globals()[i] = np.append(ran_mat(-4,4,2*vec_num),np.zeros(vec_num)).reshape(3,vec_num).transpose()
    #pos = np.append(ran_mat(-4,4,2*vec_num),np.zeros(vec_num)).reshape(3,vec_num).transpose()
    #m = np.append(ran_mat(-4,4,2*vec_num),np.zeros(vec_num)).reshape(3,vec_num).transpose()
    return pos, vel, m


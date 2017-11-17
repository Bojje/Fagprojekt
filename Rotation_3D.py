def ang_vel_step3D(moments, ang_vel, torque, timestep, inertia):
    import numpy as np
    return ang_vel + torque / inertia * timestep

def rot_step3D(moments, ang_vel, timestep):
    import numpy as np
    from div0 import div0
    ang_disp = ang_vel * timestep                                       #Angular displacement
    ang_disp_mag = np.sqrt(np.sum(ang_disp**2,axis=1)[:,np.newaxis])    #Magnitude of displacement
    # normed_disp = ang_disp/ang_disp_mag                               #Normed displacement vector
    # normed_disp[np.isnan(normed_disp)] = 0                            #Removal of undefined values (division by 0)
    R = np.sin(ang_disp_mag/2) * div0(ang_disp, ang_disp_mag)
    w = np.cos(ang_disp_mag/2)
    A = np.cross(R, moments, axis=1)
    result = 2 * np.cross(R, A + w * moments, axis=1)
    result[np.isnan(result)] = 0
    return moments + result

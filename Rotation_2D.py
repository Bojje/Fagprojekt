def rot_step2D(moments, ang, ang_vel, torque, timestep):  # Rotation step
    ang = ang_vel * timestep + torque / (2 * inertia) * timestep**2

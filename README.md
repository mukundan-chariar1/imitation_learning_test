# Imitation learning using brax and torch

## Notes

Attributes of state for generalized
    root_com: (num_links,) center of mass position of link root kinematic tree
    cinr: (num_links,) inertia in com frame
    cd: (num_links,) link velocities in com frame
    cdof: (qd_size,) dofs in com frame
    cdofd: (qd_size,) cdof velocity
    mass_mx: (qd_size, qd_size) mass matrix
    mass_mx_inv: (qd_size, qd_size) inverse mass matrix
    contact: calculated contacts
    con_jac: constraint jacobian
    con_diag: constraint A diagonal
    con_aref: constraint reference acceleration
    qf_smooth: (qd_size,) smooth dynamics force
    qf_constraint: (qd_size,) force from constraints (collision etc)
    qdd: (qd_size,) joint acceleration vector
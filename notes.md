# brax.base.System

Describes a physical environment: its links, joints and geometries.

## Attributes:

- gravity: (3,) linear universal force applied during forward dynamics
- viscosity: (1,) viscosity of the medium applied to all links
- density: (1,) density of the medium applied to all links
- link: (num_link,) the links in the system
- dof: (qd_size,) every degree of freedom for the system
- actuator: actuators that can be applied to links
- init_q: (q_size,) initial q position for the system
- elasticity: bounce/restitution encountered when hitting another geometry
- vel_damping: (1,) linear vel damping applied to each body.
- ang_damping: (1,) angular vel damping applied to each body.
- baumgarte_erp: how aggressively interpenetrating bodies should push away\
            from one another
- spring_mass_scale: a float that scales mass as `mass^(1 - x)`
- spring_inertia_scale: a float that scales inertia diag as `inertia^(1 - x)`
- joint_scale_ang: scale for position-based joint rotation update
- joint_scale_pos: scale for position-based joint position update
- collide_scale: fraction of position based collide update to apply
- enable_fluid: (1,) enables or disables fluid forces based on the
    default viscosity and density parameters provided in the XML
- link_names: (num_link,) link names
- link_types: (num_link,) string specifying the joint type of each link
            valid types are:

    - 'f': free, full 6 dof (position + rotation), no parent link
    - '1': revolute,  1 dof, like a hinge
    - '2': universal, 2 dof, like a drive shaft joint
    - '3': spherical, 3 dof, like a ball joint
- link_parents: (num_link,) int list specifying the index of each link's
                parent link, or -1 if the link has no parent
- matrix_inv_iterations: maximum number of iterations of the matrix inverse
- solver_iterations: maximum number of iterations of the constraint solver
- solver_maxls: maximum number of line searches of the constraint solver
- mj_model: mujoco.MjModel that was used to build this brax System
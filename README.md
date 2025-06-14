# Imitation learning using brax and torch

## Docker Setup

### Prerequisites
- Docker installed on your system
- NVIDIA GPU with appropriate drivers (for GPU acceleration)
- NVIDIA Container Toolkit installed (for GPU access in Docker)

### Docker Commands

#### Building the Docker Image
Build the image with all required dependencies:
```bash
docker build -t irl-project .
```

#### Running the Container
Run the container with GPU support and current directory mounted:
```bash
docker run --gpus all -it --rm -v "${PWD}:/app" irl-project
```

This command:
- `--gpus all`: Enables GPU access
- `-it`: Provides interactive terminal
- `--rm`: Automatically removes the container when exited
- `-v "${PWD}:/app"`: Mounts current directory to /app in the container

#### Common Operations

**Attaching to a Running Container**
If you need to connect to an already running container:
```bash
# First, find the container ID
docker ps
# Then attach to it
docker exec -it CONTAINER_ID bash
```

**Exiting a Container**
Simply type `exit` in the terminal to leave the container. Since we use `--rm`, the container will be automatically removed when you exit.

**Running Specific Commands**
To run a specific Python script in the container:
```bash
docker run --gpus all -it --rm -v "${PWD}:/app" irl-project python your_script.py
```

**Persisting Data**
All changes made in the `/app` directory are automatically saved to your host machine's current directory.

**Troubleshooting GPU Issues**
If you encounter GPU-related issues, verify that NVIDIA drivers are working:
```bash
# Inside the container
nvidia-smi
```

If this fails, try forcing CPU mode by modifying the entrypoint script.

## Notes

Attributes of state for generalized
- root_com: (num_links,) center of mass position of link root kinematic tree
- cinr: (num_links,) inertia in com frame
- cd: (num_links,) link velocities in com frame
- cdof: (qd_size,) dofs in com frame
- cdofd: (qd_size,) cdof velocity
- mass_mx: (qd_size, qd_size) mass matrix
- mass_mx_inv: (qd_size, qd_size) inverse mass matrix
- contact: calculated contacts
- con_jac: constraint jacobian
- con_diag: constraint A diagonal
- con_aref: constraint reference acceleration
- qf_smooth: (qd_size,) smooth dynamics force
- qf_constraint: (qd_size,) force from constraints (collision etc)
- qdd: (qd_size,) joint acceleration vector

Attributes of state in general
- q: (q_size,) joint position vector
- qd: (qd_size,) joint velocity vector
- x: (num_links,) link position in world frame
- xd: (num_links,) link velocity in world frame
- contact: calculated contacts

link names for base humanoid:
- 'torso', 
- 'lwaist', 
- 'pelvis', 
- 'right_thigh', 
- 'right_shin', 
- 'left_thigh', 
- 'left_shin', 
- 'right_upper_arm', 
- 'right_lower_arm', 
- 'left_upper_arm', 
- 'left_lower_arm'


Describes a physical environment: its links, joints and geometries.

- Attributes:
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
                * 'f': free, full 6 dof (position + rotation), no parent link
                * '1': revolute,  1 dof, like a hinge
                * '2': universal, 2 dof, like a drive shaft joint
                * '3': spherical, 3 dof, like a ball joint
    - link_parents: (num_link,) int list specifying the index of each link's
                  parent link, or -1 if the link has no parent
    - matrix_inv_iterations: maximum number of iterations of the matrix inverse
    - solver_iterations: maximum number of iterations of the constraint solver
    - solver_maxls: maximum number of line searches of the constraint solver
    - mj_model: mujoco.MjModel that was used to build this brax System


['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']

 """PPO training.

  Args:
    environment: the environment to train
    num_timesteps: the total number of environment steps to use during training
    episode_length: the length of an environment episode
    wrap_env: If True, wrap the environment for training. Otherwise use the
      environment as is.
    action_repeat: the number of timesteps to repeat an action
    num_envs: the number of parallel environments to use for rollouts
      NOTE: `num_envs` must be divisible by the total number of chips since each
        chip gets `num_envs // total_number_of_chips` environments to roll out
      NOTE: `batch_size * num_minibatches` must be divisible by `num_envs` since
        data generated by `num_envs` parallel envs gets used for gradient
        updates over `num_minibatches` of data, where each minibatch has a
        leading dimension of `batch_size`
    max_devices_per_host: maximum number of chips to use per host process
    num_eval_envs: the number of envs to use for evluation. Each env will run 1
      episode, and all envs run in parallel during eval.
    learning_rate: learning rate for ppo loss
    entropy_cost: entropy reward for ppo loss, higher values increase entropy of
      the policy
    discounting: discounting rate
    seed: random seed
    unroll_length: the number of timesteps to unroll in each environment. The
      PPO loss is computed over `unroll_length` timesteps
    batch_size: the batch size for each minibatch SGD step
    num_minibatches: the number of times to run the SGD step, each with a
      different minibatch with leading dimension of `batch_size`
    num_updates_per_batch: the number of times to run the gradient update over
      all minibatches before doing a new environment rollout
    num_evals: the number of evals to run during the entire training run.
      Increasing the number of evals increases total training time
    num_resets_per_eval: the number of environment resets to run between each
      eval. The environment resets occur on the host
    normalize_observations: whether to normalize observations
    reward_scaling: float scaling for reward
    clipping_epsilon: clipping epsilon for PPO loss
    gae_lambda: General advantage estimation lambda
    deterministic_eval: whether to run the eval with a deterministic policy
    network_factory: function that generates networks for policy and value
      functions
    progress_fn: a user-defined callback function for reporting/plotting metrics
    normalize_advantage: whether to normalize advantage estimate
    eval_env: an optional environment for eval only, defaults to `environment`
    policy_params_fn: a user-defined callback function that can be used for
      saving policy checkpoints
    randomization_fn: a user-defined callback function that generates randomized
      environments
    restore_checkpoint_path: the path used to restore previous model params
    max_grad_norm: gradient clipping norm value. If None, no clipping is done

  Returns:
    Tuple of (make_policy function, network params, metrics)
  """
solo indexes according to names: [9, 12, 11, 10, 13, 0]
left indexes according to names: [2, 15, 16, 1, 4, 3, 17, 18, 14]
right indexes according to names: [6, 20, 21, 5, 8, 7, 22, 23, 19]

['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']

# Joints
pelvis: 0 change according to height
l hip (1) = r hip (5)= +ve
l knee (2) = r knee (6) = +ve
l ankle (3) = r ankle (7) = +ve
l toe (4) = r toe (8) = +ve

torso (9) = +ve
spine (10) = +ve
chest (11) = +ve
neck (12) = +ve
head (13) = +ve

l thorax (14) = -ve r thorax (19)= -ve
l shoulder (15) = -ve r shoulder (20)= -ve
l elbow (16) = -ve r elbow (21)= -ve
l wrist (17) = -ve r wrist (22)= -ve
l hand (18) = -ve r hand (23)= -ve

# Geoms

pelvis: (1) sphere
l hip (2) = r hip (6)= capsule (sphere+cylinder)
l knee (3) = r knee (7) = capsule (sphere+cylinder)
l ankle (4) = r ankle (8) = box
l toe (5) = r toe (9) = box

torso (10) = capsule (sphere+cylinder)
spine (11) = capsule (sphere+cylinder)
chest (12) = capsule (sphere+cylinder)
neck (13) = capsule (sphere+cylinder)
head (14) = sphere

l thorax (15) = -ve r thorax (20)= capsule (sphere+cylinder)
l shoulder (16) = -ve r shoulder (21)= capsule (sphere+cylinder)
l elbow (17) = -ve r elbow (22)= capsule (sphere+cylinder)
l wrist (18) = -ve r wrist (23)= capsule (sphere+cylinder)
l hand (19) = -ve r hand (24)= sphere




custom in mjcf={'ang_damping': array(-0.05), 'baumgarte_erp': array(0.1), 'collide_scale': array(1.), 'constraint_ang_damping': array([30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30.,
       30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30.]), 'constraint_limit_stiffness': array([2500., 2500., 2500., 2500., 2500., 2500., 2500., 2500., 2500.,
       2500., 2500., 2500., 2500., 2500., 2500., 2500., 2500., 2500.,
       2500., 2500., 2500., 2500., 2500., 2500., 2500.]), 'constraint_stiffness': array([27000., 27000., 27000., 27000., 27000., 27000., 27000., 27000.,
       27000., 27000., 27000., 27000., 27000., 27000., 27000., 27000.,
       27000., 27000., 27000., 27000., 27000., 27000., 27000., 27000.,
       27000.]), 'constraint_vel_damping': array([80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80.,
       80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80.]), 'elasticity': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0.]), 'joint_scale_ang': array(0.1), 'joint_scale_pos': array(0.5), 'matrix_inv_iterations': array(20.), 'solver_maxls': array(15.), 'spring_inertia_scale': array(1.), 'spring_mass_scale': array(0.), 'vel_damping': array(0.)}



# coordinate systems

## Z (vertical) blue
## Y (horizontal) green
## X (horizontal) blue

## translation joints

L_Hip_z_transl
L_Knee_z_transl
L_Ankle_z_transl
L_Toe_x_transl
R_Hip_z_transl
R_Knee_z_transl
R_Ankle_z_transl
R_Toe_x_transl
Torso_z_transl
Spine_z_transl
Chest_z_transl
Neck_z_transl
Head_z_transl
L_Thorax_y_transl
L_Shoulder_y_transl
L_Elbow_y_transl
L_Wrist_y_transl
L_Hand_y_transl
R_Thorax_y_transl
R_Shoulder_y_transl
R_Elbow_y_transl
R_Wrist_y_transl
R_Hand_y_transl

<position joint="L_Hip_z_transl" />
<position joint="L_Knee_z_transl" />
<position joint="L_Ankle_z_transl" />
<position joint="L_Toe_x_transl" />
<position joint="R_Hip_z_transl" />
<position joint="R_Knee_z_transl" />
<position joint="R_Ankle_z_transl" />
<position joint="R_Toe_x_transl" />
<position joint="Torso_z_transl" />
<position joint="Spine_z_transl" />
<position joint="Chest_z_transl" />
<position joint="Neck_z_transl" />
<position joint="Head_z_transl" />
<position joint="L_Thorax_y_transl" />
<position joint="L_Shoulder_y_transl" />
<position joint="L_Elbow_y_transl" />
<position joint="L_Wrist_y_transl" />
<position joint="L_Hand_y_transl" />
<position joint="R_Thorax_y_transl" />
<position joint="R_Shoulder_y_transl" />
<position joint="R_Elbow_y_transl" />
<position joint="R_Wrist_y_transl" />
<position joint="R_Hand_y_transl" />


# qpos notes

- 0: pelvis x
- 1: pelvis y
- 2: pelvis z
- 3: pelvis rot w
- 4: pelvis rot x
- 5: pelvis rot y
- 6: pelvis rot z
- 7: l hip z
- 8: l hip rot x
- 9: l hip rot y
- 10: l hip rot z
- 11: l knee z
- 12: l knee rot x
- 13: l knee rot y
- 14: l knee rot z
- 15: l ankle z
- 16: l ankle rot x
- 17: l ankle rot
- 18: l ankle rot
- 19: l toe x
- 20: l toe rot x
- 21: l toe rot y
- 22: l toe rot z
- 23: r hip z
- 24: r hip rot x
- 25: r hip rot x
- 26: r hip rot x
- 27: r knee z
- 28: r knee rot x
- 29: r knee rot y
- 30: r knee rot z
- 31: r ankle z
- 32: r ankle rot x
- 33: r ankle rot y
- 34: r ankle rot z
- 35: r toe x
- 36: r toe rot x
- 37: r toe rot y
- 38: r toe rot z
- 39: torso z
- 40: torso rot x
- 41: torso rot y
- 42: torso rot z
- 43: spine z
- 44: spine rot x
- 45: spine rot y
- 46: spine rot z
- 47: chest z
- 48: chest rot x
- 49: chest rot y
- 50: chest rot z
- 51: neck z
- 52: neck rot x
- 53: neck rot y
- 54: neck rot z
- 55: head z
- 56: neck rot x
- 57: neck rot y
- 58: neck rot z
- 59: l thorax y
- 60: l thorax rot x
- 61: l thorax rot y
- 62: l thorax rot z
- 63: l shoulder y
- 64: l shoulder rot x
- 65: l shoulder rot y
- 66: l shoulder rot z
- 67: l elbow y
- 68: l elbow rot x
- 69: l elbow rot y
- 70: l elbow rot z
- 71: l wrist y
- 72: l wrist rot x
- 73: l wrist rot y
- 74: l wrist rot z
- 75: l hand y
- 76: l hand rot x
- 77: l hand rot y
- 78: l hand rot z
- 79: r thorax y
- 80: r thorax rot x
- 81: r thorax rot y
- 82: r thorax rot z
- 83: r shoulder y
- 84: r shoulder rot x
- 85: r shoulder rot y
- 86: r shoulder rot z
- 87: r elbow y
- 88: r elbow rot x
- 89: r elbow rot y
- 90: r elbow rot z
- 91: r wrist y
- 92: r wrist rot x
- 93: r wrist rot y
- 94: r wrist rot z
- 95: r hand y
- 96: r hand rot x
- 97: r hand rot y
- 98: r hand rot z

# act notes

- 0: l hip x
- 1: l hip y
- 2: l hip z
- 3: l knee x
- 4: l knee y
- 5: l knee z
- 6: l ankle x
- 7: l ankle y
- 8: l ankle z
- 9: l toe x
- 10: l toe y
- 11: l toe z
- 12: r hip x
- 13: r hip y
- 14: r hip z
- 15: r knee x
- 16: r knee y
- 17: r knee z
- 18: r ankle x
- 19: r ankle y
- 20: r ankle z
- 21: r toe x
- 22: r toe y
- 23: r toe z
- 24: torso x
- 25: torso y
- 26: torso z
- 27: spine x
- 28: spine y
- 29: spine z
- 30: chest x
- 31: chest y
- 32: chest z
- 33: neck x
- 34: neck y
- 35: neck z
- 36: head x
- 37: head y
- 38: head z
- 39: l thorax x
- 40: l thorax y
- 41: l thorax z
- 42: l shoulder x
- 43: l shoulder y
- 44: l shoulder z
- 45: l elbow x
- 46: l elbow y
- 47: l elbow z
- 48: l wrist x
- 49: l wrist y
- 50: l wrist z
- 51: l hand x
- 52: l hand y
- 53: l hand z
- 54: r thorax x
- 55: r thorax y
- 56: r thorax z
- 57: r shoulder x
- 58: r shoulder y
- 59: r shoulder z
- 60: r elbow x
- 61: r elbow y
- 62: r elbow z
- 63: r wrist x
- 64: r wrist y
- 65: r wrist z
- 66: r hand x
- 67: r hand y
- 68: r hand z
- 69: l hip z transl  
- 70: l knee z transl  
- 71: l ankle z transl  
- 72: l toe x transl  
- 73: r hip z transl  
- 74: r knee z transl  
- 75: r ankle z transl  
- 76: r toe x transl  
- 77: torso z transl  
- 78: spine z transl  
- 79: chest z transl  
- 80: neck z transl  
- 81: head z transl  
- 82: l thorax y transl  
- 83: l shoulder y transl  
- 84: l elbow y transl  
- 85: l wrist y transl  
- 86: l hand y transl  
- 87: r thorax y transl  
- 88: r shoulder y transl  
- 89: r elbow y transl  
- 90: r wrist y transl  
- 91: r hand y transl  

# limit idxs

- 6: l hip z transl  
- 10: l knee z transl  
- 14: l ankle z transl  
- 18: l toe x transl  
- 22: r hip z transl  
- 26: r knee z transl  
- 30: r ankle z transl  
- 34: r toe x transl  
- 38: torso z transl  
- 42: spine z transl  
- 46: chest z transl  
- 50: neck z transl  
- 54: head z transl  
- 58: l thorax y transl  
- 62: l shoulder y transl  
- 66: l elbow y transl  
- 70: l wrist y transl  
- 74: l hand y transl  
- 78: r thorax y transl  
- 82: r shoulder y transl  
- 86: r elbow y transl  
- 90: r wrist y transl  
- 94: r hand y transl  

## pytree nodes in heirarchical structure

- 0 Floor
- 1    'Pelvis', 
- 2      'L_Hip', 
- 3          'L_Knee', 
- 4              'L_Ankle', 
- 5                  'L_Toe', 
- 6      'R_Hip', 
- 7          'R_Knee', 
- 8              'R_Ankle', 
- 9                  'R_Toe', 
- 10     'Torso', 
- 11         'Spine', 
- 12             'Chest', 
- 13                 'Neck', 
- 14                     'Head', 
- 15         'L_Thorax', 
- 16             'L_Shoulder', 
- 17                 'L_Elbow', 
- 18                     'L_Wrist', 
- 19                         'L_Hand', 
- 20         'R_Thorax', 
- 21             'R_Shoulder', 
- 22                 'R_Elbow', 
- 23                     'R_Wrist', 
- 24                         'R_Hand'


'nq', 'nv', 'nu', 'na', 'nbody', 'njnt', 'ngeom', 'nsite', 'ncam', 'nmesh', 'nmeshvert', 'nmeshface', 'nmat', 'npair', 'nexclude', 'neq', 'ngravcomp', 'nnumeric', 'nuserdata', 'ntuple', 'nsensor', 'nkey', 'nM', 'opt', 'stat', 'qpos0', 'qpos_spring', 'body_parentid', 'body_rootid', 'body_weldid', 'body_jntnum', 'body_jntadr', 'body_dofnum', 'body_dofadr', 'body_geomnum', 'body_geomadr', 'body_pos', 'body_quat', 'body_ipos', 'body_iquat', 'body_mass', 'body_subtreemass', 'body_inertia', 'body_gravcomp', 'body_invweight0', 'jnt_type', 'jnt_qposadr', 'jnt_dofadr', 'jnt_bodyid', 'jnt_limited', 'jnt_actfrclimited', 'jnt_actgravcomp', 'jnt_solref', 'jnt_solimp', 'jnt_pos', 'jnt_axis', 'jnt_stiffness', 'jnt_range', 'jnt_actfrcrange', 'jnt_margin', 'dof_bodyid', 'dof_jntid', 'dof_parentid', 'dof_Madr', 'dof_solref', 'dof_solimp', 'dof_frictionloss', 'dof_armature', 'dof_damping', 'dof_invweight0', 'dof_M0', 'geom_type', 'geom_contype', 'geom_conaffinity', 'geom_condim', 'geom_bodyid', 'geom_dataid', 'geom_group', 'geom_matid', 'geom_priority', 'geom_solmix', 'geom_solref', 'geom_solimp', 'geom_size', 'geom_rbound', 'geom_pos', 'geom_quat', 'geom_friction', 'geom_margin', 'geom_gap', 'geom_rgba', 'site_bodyid', 'site_pos', 'site_quat', 'cam_mode', 'cam_bodyid', 'cam_targetbodyid', 'cam_pos', 'cam_quat', 'cam_poscom0', 'cam_pos0', 'cam_mat0', 'mesh_vertadr', 'mesh_faceadr', 'mesh_graphadr', 'mesh_vert', 'mesh_face', 'mesh_graph', 'mat_rgba', 'pair_dim', 'pair_geom1', 'pair_geom2', 'pair_solref', 'pair_solreffriction', 'pair_solimp', 'pair_margin', 'pair_gap', 'pair_friction', 'exclude_signature', 'eq_type', 'eq_obj1id', 'eq_obj2id', 'eq_active0', 'eq_solref', 'eq_solimp', 'eq_data', 'actuator_trntype', 'actuator_dyntype', 'actuator_gaintype', 'actuator_biastype', 'actuator_trnid', 'actuator_actadr', 'actuator_actnum', 'actuator_ctrllimited', 'actuator_forcelimited', 'actuator_actlimited', 'actuator_dynprm', 'actuator_gainprm', 'actuator_biasprm', 'actuator_ctrlrange', 'actuator_forcerange', 'actuator_actrange', 'actuator_gear', 'numeric_adr', 'numeric_data', 'tuple_adr', 'tuple_size', 'tuple_objtype', 'tuple_objid', 'tuple_objprm', 'name_bodyadr', 'name_jntadr', 'name_geomadr', 'name_siteadr', 'name_camadr', 'name_meshadr', 'name_pairadr', 'name_eqadr', 'name_actuatoradr', 'name_sensoradr', 'name_numericadr', 'name_tupleadr', 'name_keyadr', 'names', 'gravity', 'viscosity', 'density', 'link', 'dof', 'actuator', 'init_q', 'elasticity', 'vel_damping', 'ang_damping', 'baumgarte_erp', 'spring_mass_scale', 'spring_inertia_scale', 'joint_scale_ang', 'joint_scale_pos', 'collide_scale', 'enable_fluid', 'link_names', 'link_types', 'link_parents', 'matrix_inv_iterations', 'solver_iterations', 'solver_maxls', 'mj_model'
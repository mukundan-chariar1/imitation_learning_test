"""Static model of the scene that remains unchanged with each physics step.

  Attributes:
    nq: number of generalized coordinates = dim(qpos)
    nv: number of degrees of freedom = dim(qvel)
    nu: number of actuators/controls = dim(ctrl)
    na: number of activation states = dim(act)
    nbody: number of bodies
    nbvh: number of total bounding volumes in all bodies
    nbvhstatic: number of static bounding volumes (aabb stored in mjModel)
    nbvhdynamic: number of dynamic bounding volumes (aabb stored in mjData)
    njnt: number of joints
    ngeom: number of geoms
    nsite: number of sites
    ncam: number of cameras
    nlight: number of lights
    nflex: number of flexes
    nflexvert: number of vertices in all flexes
    nflexedge: number of edges in all flexes
    nflexelem: number of elements in all flexes
    nflexelemdata: number of element vertex ids in all flexes
    nflexshelldata: number of shell fragment vertex ids in all flexes
    nflexevpair: number of element-vertex pairs in all flexes
    nflextexcoord: number of vertices with texture coordinates
    nmesh: number of meshes
    nmeshvert: number of vertices in all meshes
    nmeshnormal: number of normals in all meshes
    nmeshtexcoord: number of texcoords in all meshes
    nmeshface: number of triangular faces in all meshes
    nmeshgraph: number of ints in mesh auxiliary data
    nhfield: number of heightfields
    nhfielddata: number of data points in all heightfields
    ntex: number of textures
    ntexdata: number of bytes in texture rgb data
    nmat: number of materials
    npair: number of predefined geom pairs
    nexclude: number of excluded geom pairs
    neq: number of equality constraints
    ntendon: number of tendons
    nwrap: number of wrap objects in all tendon paths
    nsensor: number of sensors
    nnumeric: number of numeric custom fields
    ntuple: number of tuple custom fields
    nkey: number of keyframes
    nmocap: number of mocap bodies
    nM: number of non-zeros in sparse inertia matrix
    nD: number of non-zeros in sparse dof-dof matrix
    nB: number of non-zeros in sparse body-dof matrix
    nC: number of non-zeros in sparse reduced dof-dof matrix
    nD: number of non-zeros in sparse dof-dof matrix
    nJmom: number of non-zeros in sparse actuator_moment matrix
    ntree: number of kinematic trees under world body
    ngravcomp: number of bodies with nonzero gravcomp
    nuserdata: size of userdata array
    nsensordata: number of mjtNums in sensor data vector
    narena: number of bytes in the mjData arena (inclusive of stack)
    opt: physics options
    stat: model statistics
    qpos0: qpos values at default pose                        (nq,)
    qpos_spring: reference pose for springs                   (nq,)
    body_parentid: id of body's parent                        (nbody,)
    body_rootid: id of root above body                        (nbody,)
    body_weldid: id of body that this body is welded to       (nbody,)
    body_jntnum: number of joints for this body               (nbody,)
    body_jntadr: start addr of joints; -1: no joints          (nbody,)
    body_dofnum: number of motion degrees of freedom          (nbody,)
    body_dofadr: start addr of dofs; -1: no dofs              (nbody,)
    body_treeid: id of body's kinematic tree; -1: static      (nbody,)
    body_geomnum: number of geoms                             (nbody,)
    body_geomadr: start addr of geoms; -1: no geoms           (nbody,)
    body_simple: 1: diag M; 2: diag M, sliders only           (nbody,)
    body_pos: position offset rel. to parent body             (nbody, 3)
    body_quat: orientation offset rel. to parent body         (nbody, 4)
    body_ipos: local position of center of mass               (nbody, 3)
    body_iquat: local orientation of inertia ellipsoid        (nbody, 4)
    body_mass: mass                                           (nbody,)
    body_subtreemass: mass of subtree starting at this body   (nbody,)
    body_inertia: diagonal inertia in ipos/iquat frame        (nbody, 3)
    body_gravcomp: antigravity force, units of body weight    (nbody,)
    body_margin: MAX over all geom margins                    (nbody,)
    body_contype: OR over all geom contypes                   (nbody,)
    body_conaffinity: OR over all geom conaffinities          (nbody,)
    body_bvhadr: address of bvh root                          (nbody,)
    body_bvhnum: number of bounding volumes                   (nbody,)
    bvh_child: left and right children in tree                (nbvh, 2)
    bvh_nodeid: geom or elem id of node; -1: non-leaf         (nbvh,)
    bvh_aabb: local bounding box (center, size)               (nbvhstatic, 6)
    body_invweight0: mean inv inert in qpos0 (trn, rot)       (nbody, 2)
    jnt_type: type of joint (mjtJoint)                        (njnt,)
    jnt_qposadr: start addr in 'qpos' for joint's data        (njnt,)
    jnt_dofadr: start addr in 'qvel' for joint's data         (njnt,)
    jnt_bodyid: id of joint's body                            (njnt,)
    jnt_group: group for visibility                           (njnt,)
    jnt_limited: does joint have limits                       (njnt,)
    jnt_actfrclimited: does joint have actuator force limits  (njnt,)
    jnt_actgravcomp: is gravcomp force applied via actuators  (njnt,)
    jnt_solref: constraint solver reference: limit            (njnt, mjNREF)
    jnt_solimp: constraint solver impedance: limit            (njnt, mjNIMP)
    jnt_pos: local anchor position                            (njnt, 3)
    jnt_axis: local joint axis                                (njnt, 3)
    jnt_stiffness: stiffness coefficient                      (njnt,)
    jnt_range: joint limits                                   (njnt, 2)
    jnt_actfrcrange: range of total actuator force            (njnt, 2)
    jnt_margin: min distance for limit detection              (njnt,)
    dof_bodyid: id of dof's body                              (nv,)
    dof_jntid: id of dof's joint                              (nv,)
    dof_parentid: id of dof's parent; -1: none                (nv,)
    dof_treeid: id of dof's kinematic tree                    (nv,)
    dof_Madr: dof address in M-diagonal                       (nv,)
    dof_simplenum: number of consecutive simple dofs          (nv,)
    dof_solref: constraint solver reference:frictionloss      (nv, mjNREF)
    dof_solimp: constraint solver impedance:frictionloss      (nv, mjNIMP)
    dof_frictionloss: dof friction loss                       (nv,)
    dof_hasfrictionloss: dof has >0 frictionloss (MJX)        (nv,)
    dof_armature: dof armature inertia/mass                   (nv,)
    dof_damping: damping coefficient                          (nv,)
    dof_invweight0: diag. inverse inertia in qpos0            (nv,)
    dof_M0: diag. inertia in qpos0                            (nv,)
    geom_type: geometric type (mjtGeom)                       (ngeom,)
    geom_contype: geom contact type                           (ngeom,)
    geom_conaffinity: geom contact affinity                   (ngeom,)
    geom_condim: contact dimensionality (1, 3, 4, 6)          (ngeom,)
    geom_bodyid: id of geom's body                            (ngeom,)
    geom_dataid: id of geom's mesh/hfield; -1: none           (ngeom,)
    geom_group: group for visibility                          (ngeom,)
    geom_matid: material id for rendering                     (ngeom,)
    geom_priority: geom contact priority                      (ngeom,)
    geom_solmix: mixing coef for solref/imp in geom pair      (ngeom,)
    geom_solref: constraint solver reference: contact         (ngeom, mjNREF)
    geom_solimp: constraint solver impedance: contact         (ngeom, mjNIMP)
    geom_size: geom-specific size parameters                  (ngeom, 3)
    geom_aabb: bounding box, (center, size)                   (ngeom, 6)
    geom_rbound: radius of bounding sphere                    (ngeom,)
    geom_rbound_hfield: static rbound for hfield grid bounds  (ngeom,)
    geom_pos: local position offset rel. to body              (ngeom, 3)
    geom_quat: local orientation offset rel. to body          (ngeom, 4)
    geom_friction: friction for (slide, spin, roll)           (ngeom, 3)
    geom_margin: include in solver if dist<margin-gap         (ngeom,)
    geom_gap: include in solver if dist<margin-gap            (ngeom,)
    geom_rgba: rgba when material is omitted                  (ngeom, 4)
    site_bodyid: id of site's body                            (nsite,)
    site_pos: local position offset rel. to body              (nsite, 3)
    site_quat: local orientation offset rel. to body          (nsite, 4)
    cam_mode:  camera tracking mode (mjtCamLight)             (ncam,)
    cam_bodyid:  id of camera's body                          (ncam,)
    cam_targetbodyid:  id of targeted body; -1: none          (ncam,)
    cam_pos:  position rel. to body frame                     (ncam, 3)
    cam_quat:  orientation rel. to body frame                 (ncam, 4)
    cam_poscom0:  global position rel. to sub-com in qpos0    (ncam, 3)
    cam_pos0: global position rel. to body in qpos0           (ncam, 3)
    cam_mat0: global orientation in qpos0                     (ncam, 3, 3)
    cam_fovy: y field-of-view                                 (ncam,)
    cam_resolution: resolution: pixels                        (ncam, 2)
    cam_sensorsize: sensor size: length                       (ncam, 2)
    cam_intrinsic: [focal length; principal point]            (ncam, 4)
    light_mode: light tracking mode (mjtCamLight)             (nlight,)
    light_bodyid: id of light's body                          (nlight,)
    light_targetbodyid: id of targeted body; -1: none         (nlight,)
    light_directional: directional light                      (nlight,)
    light_pos: position rel. to body frame                    (nlight, 3)
    light_dir: direction rel. to body frame                   (nlight, 3)
    light_poscom0: global position rel. to sub-com in qpos0   (nlight, 3)
    light_pos0: global position rel. to body in qpos0         (nlight, 3)
    light_dir0: global direction in qpos0                     (nlight, 3)
    flex_contype: flex contact type                           (nflex,)
    flex_conaffinity: flex contact affinity                   (nflex,)
    flex_condim: contact dimensionality (1, 3, 4, 6)          (nflex,)
    flex_priority: flex contact priority                      (nflex,)
    flex_solmix: mix coef for solref/imp in contact pair      (nflex,)
    flex_solref: constraint solver reference: contact         (nflex, mjNREF)
    flex_solimp: constraint solver impedance: contact         (nflex, mjNIMP)
    flex_friction: friction for (slide, spin, roll)           (nflex,)
    flex_margin: detect contact if dist<margin                (nflex,)
    flex_gap: include in solver if dist<margin-gap            (nflex,)
    flex_internal: internal flex collision enabled            (nflex,)
    flex_selfcollide: self collision mode (mjtFlexSelf)       (nflex,)
    flex_activelayers: number of active element layers, 3D only  (nflex,)
    flex_dim: 1: lines, 2: triangles, 3: tetrahedra           (nflex,)
    flex_vertadr: first vertex address                        (nflex,)
    flex_vertnum: number of vertices                          (nflex,)
    flex_edgeadr: first edge address                          (nflex,)
    flex_edgenum: number of edges                             (nflex,)
    flex_elemadr: first element address                       (nflex,)
    flex_elemnum: number of elements                          (nflex,)
    flex_elemdataadr: first element vertex id address         (nflex,)
    flex_evpairadr: first evpair address                      (nflex,)
    flex_evpairnum: number of evpairs                         (nflex,)
    flex_vertbodyid: vertex body ids                          (nflex,)
    flex_edge: edge vertex ids (2 per edge)                   (nflexedge, 2)
    flex_elem: element vertex ids (dim+1 per elem)            (nflexelemdata,)
    flex_elemlayer: element distance from surface, 3D only    (nflexelem,)
    flex_evpair: (element, vertex) collision pairs            (nflexevpair, 2)
    flex_vert: vertex positions in local body frames          (nflexvert, 3)
    flexedge_length0: edge lengths in qpos0                   (nflexedge,)
    flexedge_invweight0: edge inv. weight in qpos0            (nflexedge,)
    flex_radius: radius around primitive element              (nflex,)
    flex_edgestiffness: edge stiffness                        (nflex,)
    flex_edgedamping: edge damping                            (nflex,)
    flex_edgeequality: is edge equality constraint defined    (nflex,)
    flex_rigid: are all verices in the same body              (nflex,)
    flexedge_rigid: are both edge vertices in same body       (nflexedge,)
    flex_centered: are all vertex coordinates (0,0,0)         (nflex,)
    flex_bvhadr: address of bvh root; -1: no bvh              (nflex,)
    flex_bvhnum: number of bounding volumes                   (nflex,)
    mesh_vertadr: first vertex address                        (nmesh,)
    mesh_vertnum: number of vertices                          (nmesh,)
    mesh_faceadr: first face address                          (nmesh,)
    mesh_bvhadr: address of bvh root                          (nmesh,)
    mesh_bvhnum: number of bvh                                (nmesh,)
    mesh_graphadr: graph data address; -1: no graph           (nmesh,)
    mesh_vert: vertex positions for all meshes                (nmeshvert, 3)
    mesh_face: vertex face data                               (nmeshface, 3)
    mesh_graph: convex graph data                             (nmeshgraph,)
    mesh_pos: translation applied to asset vertices           (nmesh, 3)
    mesh_quat: rotation applied to asset vertices             (nmesh, 4)
    mesh_convex: pre-compiled convex mesh info for MJX        (nmesh,)
    mesh_texcoordadr: texcoord data address; -1: no texcoord  (nmesh,)
    mesh_texcoordnum: number of texcoord                      (nmesh,)
    mesh_texcoord: vertex texcoords for all meshes            (nmeshtexcoord, 2)
    hfield_size: (x, y, z_top, z_bottom)                      (nhfield,)
    hfield_nrow: number of rows in grid                       (nhfield,)
    hfield_ncol: number of columns in grid                    (nhfield,)
    hfield_adr: address in hfield_data                        (nhfield,)
    hfield_data: elevation data                               (nhfielddata,)
    tex_type: texture type (mjtTexture)                       (ntex,)
    tex_height: number of rows in texture image               (ntex,)
    tex_width: number of columns in texture image             (ntex,)
    tex_nchannel: number of channels in texture image         (ntex,)
    tex_adr: start address in tex_data                        (ntex,)
    tex_data: pixel values                                    (ntexdata,)
    mat_rgba: rgba                                            (nmat, 4)
    mat_texid: indices of textures; -1: none                  (nmat, mjNTEXROLE)
    pair_dim: contact dimensionality                          (npair,)
    pair_geom1: id of geom1                                   (npair,)
    pair_geom2: id of geom2                                   (npair,)
    pair_signature: body1 << 16 + body2                       (npair,)
    pair_solref: solver reference: contact normal             (npair, mjNREF)
    pair_solreffriction: solver reference: contact friction   (npair, mjNREF)
    pair_solimp: solver impedance: contact                    (npair, mjNIMP)
    pair_margin: include in solver if dist<margin-gap         (npair,)
    pair_gap: include in solver if dist<margin-gap            (npair,)
    pair_friction: tangent1, 2, spin, roll1, 2                (npair, 5)
    exclude_signature: (body1+1) << 16 + body2+1              (nexclude,)
    eq_type: constraint type (mjtEq)                          (neq,)
    eq_obj1id: id of object 1                                 (neq,)
    eq_obj2id: id of object 2                                 (neq,)
    eq_objtype: type of both objects (mjtObj)                 (neq,)
    eq_active0: initial enable/disable constraint state       (neq,)
    eq_solref: constraint solver reference                    (neq, mjNREF)
    eq_solimp: constraint solver impedance                    (neq, mjNIMP)
    eq_data: numeric data for constraint                      (neq, mjNEQDATA)
    tendon_adr: address of first object in tendon's path      (ntendon,)
    tendon_num: number of objects in tendon's path            (ntendon,)
    tendon_limited: does tendon have length limits            (ntendon,)
    tendon_solref_lim: constraint solver reference: limit     (ntendon, mjNREF)
    tendon_solimp_lim: constraint solver impedance: limit     (ntendon, mjNIMP)
    tendon_solref_fri: constraint solver reference: friction  (ntendon, mjNREF)
    tendon_solimp_fri: constraint solver impedance: friction  (ntendon, mjNIMP)
    tendon_range: tendon length limits                        (ntendon, 2)
    tendon_margin: min distance for limit detection           (ntendon,)
    tendon_stiffness: stiffness coefficient                   (ntendon,)
    tendon_damping: damping coefficient                       (ntendon,)
    tendon_frictionloss: loss due to friction                 (ntendon,)
    tendon_lengthspring: spring resting length range          (ntendon, 2)
    tendon_length0: tendon length in qpos0                    (ntendon,)
    tendon_invweight0: inv. weight in qpos0                   (ntendon,)
    tendon_hasfrictionloss: tendon has >0 frictionloss (MJX)  (ntendon,)
    wrap_type: wrap object type (mjtWrap)                     (nwrap,)
    wrap_objid: object id: geom, site, joint                  (nwrap,)
    wrap_prm: divisor, joint coef, or site id                 (nwrap,)
    actuator_trntype: transmission type (mjtTrn)              (nu,)
    actuator_dyntype: dynamics type (mjtDyn)                  (nu,)
    actuator_gaintype: gain type (mjtGain)                    (nu,)
    actuator_biastype: bias type (mjtBias)                    (nu,)
    actuator_trnid: transmission id: joint, tendon, site      (nu, 2)
    actuator_actadr: first activation address; -1: stateless  (nu,)
    actuator_actnum: number of activation variables           (nu,)
    actuator_group: group for visibility                      (nu,)
    actuator_ctrllimited: is control limited                  (nu,)
    actuator_forcelimited: is force limited                   (nu,)
    actuator_actlimited: is activation limited                (nu,)
    actuator_dynprm: dynamics parameters                      (nu, mjNDYN)
    actuator_gainprm: gain parameters                         (nu, mjNGAIN)
    actuator_biasprm: bias parameters                         (nu, mjNBIAS)
    actuator_actearly: step activation before force           (nu,)
    actuator_ctrlrange: range of controls                     (nu, 2)
    actuator_forcerange: range of forces                      (nu, 2)
    actuator_actrange: range of activations                   (nu, 2)
    actuator_gear: scale length and transmitted force         (nu, 6)
    actuator_cranklength: crank length for slider-crank       (nu,)
    actuator_acc0: acceleration from unit force in qpos0      (nu,)
    actuator_lengthrange: feasible actuator length range      (nu, 2)
    sensor_type: sensor type (mjtSensor)                      (nsensor,)
    sensor_datatype: numeric data type (mjtDataType)          (nsensor,)
    sensor_needstage: required compute stage (mjtStage)       (nsensor,)
    sensor_objtype: type of sensorized object (mjtObj)        (nsensor,)
    sensor_objid: id of sensorized object                     (nsensor,)
    sensor_reftype: type of reference frame (mjtObj)          (nsensor,)
    sensor_refid: id of reference frame; -1: global frame     (nsensor,)
    sensor_dim: number of scalar outputs                      (nsensor,)
    sensor_adr: address in sensor array                       (nsensor,)
    sensor_cutoff: cutoff for real and positive; 0: ignore    (nsensor,)
    numeric_adr: address of field in numeric_data             (nnumeric,)
    numeric_data: array of all numeric fields                 (nnumericdata,)
    tuple_adr: address of text in text_data                   (ntuple,)
    tuple_size: number of objects in tuple                    (ntuple,)
    tuple_objtype: array of object types in all tuples        (ntupledata,)
    tuple_objid: array of object ids in all tuples            (ntupledata,)
    tuple_objprm: array of object params in all tuples        (ntupledata,)
    name_bodyadr: body name pointers                          (nbody,)
    name_jntadr: joint name pointers                          (njnt,)
    name_geomadr: geom name pointers                          (ngeom,)
    name_siteadr: site name pointers                          (nsite,)
    name_camadr: camera name pointers                         (ncam,)
    name_meshadr: mesh name pointers                          (nmesh,)
    name_pairadr: geom pair name pointers                     (npair,)
    name_eqadr: equality constraint name pointers             (neq,)
    name_tendonadr: tendon name pointers                      (ntendon,)
    name_actuatoradr: actuator name pointers                  (nu,)
    name_sensoradr: sensor name pointers                      (nsensor,)
    name_numericadr: numeric name pointers                    (nnumeric,)
    name_tupleadr: tuple name pointers                        (ntuple,)
    name_keyadr: keyframe name pointers                       (nkey,)
    names: names of all objects, 0-terminated                 (nnames,)
  """


  smpl_joints

  np.array([[ 1.2258e-03,  4.1264e-01,  1.3691e-01],
        [-1.6229e-04,  2.8760e-01, -1.4817e-02],
        [-1.7516e-01,  2.2512e-01, -1.9718e-02],
        [-4.2890e-01,  2.1179e-01, -4.1119e-02],
        [-6.8420e-01,  2.1956e-01, -4.6679e-02],
        [ 1.7244e-01,  2.2595e-01, -1.4918e-02],
        [ 4.3205e-01,  2.1318e-01, -4.2374e-02],
        [ 6.8128e-01,  2.2216e-01, -4.3545e-02],
        [-1.7951e-03, -2.2333e-01,  2.8219e-02],
        [-6.9466e-02, -3.1386e-01,  2.3899e-02],
        [-1.0776e-01, -6.9642e-01,  1.5049e-02],
        [-9.1982e-02, -1.0948e+00, -2.7263e-02],
        [ 6.7725e-02, -3.1474e-01,  2.1404e-02],
        [ 1.0200e-01, -6.8994e-01,  1.6908e-02],
        [ 8.8406e-02, -1.0879e+00, -2.6785e-02],
        [-3.2222e-02,  4.4744e-01,  9.7653e-02],
        [ 3.4353e-02,  4.4793e-01,  9.9210e-02],
        [-7.2048e-02,  4.1849e-01,  5.4575e-03],
        [ 7.4144e-02,  4.2113e-01,  7.5882e-03],
        [ 8.1933e-02, -1.1446e+00,  1.6693e-01],
        [ 1.5153e-01, -1.1517e+00,  1.1907e-01],
        [ 9.7877e-02, -1.1218e+00, -8.7602e-02],
        [-7.7405e-02, -1.1478e+00,  1.6646e-01],
        [-1.5278e-01, -1.1545e+00,  1.2201e-01],
        [-1.0036e-01, -1.1377e+00, -8.5108e-02]], dtype=float)

## from to shapes

"""
all

size="100.0 100.0 0.2" # Floor (0)
size="0.0942" # Pelvis (1)
fromto="-0.00135 0.0073 -0.07575 -0.00535 0.029 -0.30315" size="0.06105" # L_Hip (2)
fromto="0.0001 -0.00295 -0.07965 -0.0344 -0.01175 -0.31855" size="0.0541" # L_Knee (3)
size="0.08575 0.0483 0.0208" # L_Ankle (4)
size="0.01985 0.04785 0.0208" # L_Toe (5)
fromto="-0.00135 -0.0073 -0.07575 -0.00535 -0.029 -0.30315" size="0.06105" # R_Hip (6)
fromto="0.0001 0.00295 -0.07965 -0.0344 0.01175 -0.31855" size="0.0541" # R_Knee (7)
size="0.08575 0.0483 0.0208" # R_Ankle (8)
size="0.01985 0.04785 0.0208" # R_Toe (9)
fromto="0.0005 0.0025 0.0608 0.0006 0.003 0.0743" size="0.0769" # Torso (10)
fromto="0.0114 0.0007 0.0238 0.014 0.0008 0.0291" size="0.0755" # Spine (11)
fromto="-0.0173 -0.0009 0.0682 -0.0212 -0.001 0.0833" size="0.1002" # Chest (12)
fromto="0.0103 0.001 0.013 0.0411 0.0041 0.052" size="0.0436" # Neck (13)
size="0.1011" # Head (14)
fromto="-0.0018 0.0187 0.0063 -0.0072 0.0748 0.0252" size="0.0516" # L_Thorax (15)
fromto="-0.0049 0.0513 -0.00265 -0.01955 0.20535 -0.01045" size="0.0524" # L_Shoulder (16)
fromto="-0.00065 0.05045 0.0017 -0.00265 0.2018 0.0067" size="0.04065" # L_Elbow (17)
fromto="-0.00255 0.01685 -0.0014 -0.01015 0.06745 -0.0057" size="0.0322" # L_Wrist (18)
size="0.03385" # L_Hand (19)
fromto="-0.0018 -0.0187 0.0063 -0.0072 -0.0748 0.0252" size="0.0516" # R_Thorax (20)
fromto="-0.0049 -0.0513 -0.00265 -0.01955 -0.20535 -0.01045" size="0.0524" # R_Shoulder (21)
fromto="-0.00065 -0.05045 0.0017 -0.00265 -0.2018 0.0067" size="0.04065" # R_Elbow (22)
fromto="-0.00255 -0.01685 -0.0014 -0.01015 -0.06745 -0.0057" size="0.0322" # R_Wrist (23)
size="0.03385" # R_Hand (24)


"""
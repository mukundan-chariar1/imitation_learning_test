from lib.utils import utils
from configs import constants as _C

import os
import os.path as osp

import torch
import torch.nn as nn
from pytorch3d import transforms

import smplx
from human_body_prior.tools.model_loader import load_vposer

from pdb import set_trace as st

def append_wrists(body_pose):

    if len(body_pose.shape)==3: wrist_pose = torch.zeros([body_pose.shape[0], 2, 6],
                              dtype=body_pose.dtype,
                              device=body_pose.device)
    else: wrist_pose = torch.zeros([body_pose.shape[0], 6],
                              dtype=body_pose.dtype,
                              device=body_pose.device)
    body_pose = torch.cat([body_pose, wrist_pose], dim=1)
    return body_pose


class SMPL(nn.Module):
    def __init__(self, model, use_vposer=False, device='cuda'):
        super(SMPL, self).__init__()
        
        self.device = device
        self.model = model
        self.use_vposer = use_vposer

        num_joints = 23 if self.model.name().lower() == 'smpl' else 21

        if use_vposer:
            vposer_ckpt = osp.expandvars(_C.SMPL.VPOSER_CKPT)
            self.vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
            self.vposer = self.vposer.to(device=device)
            self.vposer.eval()
            pose_embedding = torch.zeros(
                [model.batch_size, 32], dtype=torch.float32)
        else:
            self.vposer = None
            pose_embedding = transforms.matrix_to_rotation_6d(
                torch.eye(3).unsqueeze(0).unsqueeze(0).expand(
                    model.batch_size, num_joints, -1, -1
            ))
        #st()
        self.register_buffer('pose_embedding', pose_embedding)
        self.to(device)

    @property
    def name(self, ):
        return self.model.name().lower()

    @torch.no_grad()
    def reset_params(self, **params):
        for param_name, param in self.model.named_parameters():
            if param_name in params:
                param[:] = params[param_name].clone().detach().requires_grad_(True)

    def reset_pose_embedding(self, new_embedding):   
        #st()
        if new_embedding.shape != self.pose_embedding.shape:
            if new_embedding.shape[-1] == 6:
                # Convert 6D representation to 32-D latent vector
                new_embedding=self.vposer.encode(transforms.matrix_to_axis_angle(transforms.rotation_6d_to_matrix(new_embedding[:, :-2, :])).reshape(-1, 63)).mean.detach()
            elif new_embedding.shape[-1] == 32:
                vposer_ckpt = osp.expandvars(_C.SMPL.VPOSER_CKPT)
                vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
                new_embedding=append_wrists(transforms.matrix_to_rotation_6d(vposer.decode(new_embedding).squeeze().reshape(-1, 21, 3, 3))).detach()
                del vposer_ckpt, vposer, _
                # Convert 32-D latent vector to 6D representation
                # self.vposer.decode(new_embedding)  
        self.pose_embedding.index_copy_(
            0, torch.LongTensor(range(len(self.pose_embedding))
            ).to(self.device), new_embedding.to(self.device)
        )

    def get_angle_axis(self, ):
        if self.use_vposer:
            raise f"Pose embedding shape {self.pose_embedding.shape} not transformable to axis angle!"
        return transforms.matrix_to_axis_angle(
            transforms.rotation_6d_to_matrix(self.pose_embedding
        )).reshape(self.model.batch_size, -1)

    def get_body_pose(self):
        _append_wrists = self.name == 'smpl' and self.use_vposer
        
        _append_wrists = self.name == 'smpl'
        if self.use_vposer:
            body_pose = self.vposer.decode(
                self.pose_embedding, output_type='aa').view(
                    self.pose_embedding.shape[0], -1)
            if _append_wrists: body_pose = append_wrists(body_pose)
        else:
            body_pose = self.get_angle_axis()
        return body_pose

    def forward(self, return_verts=True, return_full_pose=True, **kwargs):
        body_model_output = self.model(
            return_verts=return_verts,
            body_pose=self.get_body_pose(),
            return_full_pose=return_full_pose)

        return body_model_output


def build_smpl_model(device, **kwargs):
    joint_mapper = utils.JointMapper(utils.smpl_to_openpose(
        kwargs.get('model_type'), use_hands=False, use_face=False, use_face_contour=False))
    model_params = dict(
        model_path=_C.SMPL.FLDR,
        joint_mapper=joint_mapper,
        create_global_orient=True,
        create_body_pose=not kwargs.get('use_vposer'),
        create_transl=True,
        create_betas=True,
        create_left_hand_pose=False,
        create_right_hand_pose=False,
        create_expression=False,
        create_jaw_pose=False,
        create_leye_pose=False,
        create_reye_pose=False,
        dtype=torch.float32,
        **kwargs
    )

    model = smplx.create(gender='neutral', **model_params).to(device)
    smpl = SMPL(model=model, use_vposer=kwargs.get('use_vposer'), device=device)
    return smpl

def build_vposer_model(device, **kwargs):
    joint_mapper = utils.JointMapper(utils.smpl_to_openpose(
        kwargs.get('model_type'), use_hands=False, use_face=False, use_face_contour=False))
    model_params = dict(
        model_path=_C.SMPL.FLDR,
        joint_mapper=joint_mapper,
        create_global_orient=True,
        create_body_pose=not kwargs.get('vposer_old'),
        create_transl=True,
        create_betas=True,
        create_left_hand_pose=False,
        create_right_hand_pose=False,
        create_expression=False,
        create_jaw_pose=False,
        create_leye_pose=False,
        create_reye_pose=False,
        dtype=torch.float32,
        **kwargs
    )

    model = smplx.create(gender='neutral', **model_params).to(device)
    smpl = SMPL(model=model, use_vposer=kwargs.get('vposer_old'), device=device)
    return smpl


def build_body_model(args, device=None, batch_size=None, **kwargs):

    joint_mapper = utils.JointMapper(utils.smpl_to_openpose(
        args.get('model_type'), use_hands=False, use_face=False, use_face_contour=False))
    
    model_params = dict(
        model_path=_C.SMPL.FLDR,
        joint_mapper=joint_mapper,
        create_global_orient=True,
        create_body_pose=False,
        create_betas=True,
        create_left_hand_pose=False,
        create_right_hand_pose=False,
        create_expression=False,
        create_jaw_pose=False,
        create_leye_pose=False,
        create_reye_pose=False,
        create_transl=False,
        dtype=torch.float32,
        batch_size=batch_size,
        **args
    )

    model = smplx.create(gender='neutral', **model_params).to(device)
    return model

def load_previous_stage(file_dir, rep_idx, model=None, camera2=None, **kwargs):
    
    fname = osp.join(file_dir, 'kinematics', f'rep_{rep_idx:02d}_output.pt')
    results = torch.load(fname)

    if model is not None:
        device = model.device
        batch_size = model.model.batch_size

        if model.pose_embedding is not None:
            
            model.pose_embedding.index_copy_(
                0, torch.LongTensor(range(len(model.pose_embedding))
            ).to(device), results['pose_embedding'][:batch_size].to(device))

        global_orient = results['full_pose'][:batch_size, :3].to(device).requires_grad_(True)
        betas = results['betas'][:batch_size].to(device).requires_grad_(True)
        transl = results['transl'][:batch_size].to(device).requires_grad_(True)
        model.reset_params(global_orient=global_orient, betas=betas, transl=transl)

    if camera2 is not None and 'side_R' in results.keys():
        cam_params = {'r6d': results['side_R'], 'T': results['side_T']}
        camera2.requires_grad_(False)
        camera2.reset_params(cam_params)
        camera2.requires_grad_(True)
import torch

class SMPL:
    FLDR = 'dataset/body_models'
    DIR_REGRESSOR = 'datasets/body_models/J_regressor_OP25.npy'
    VPOSER_CKPT = 'dataset/body_models/vposer_v01'
    PARENTS = torch.tensor([
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])

    JOINTS = [
        'Pelvis', 'Left Hip', 'Right Hip', 'Spine 1 (Lower)', 'Left Knee',
        'Right Knee', 'Spine 2 (Middle)', 'Left Ankle', 'Right Ankle',
        'Spine 3 (Upper)', 'Left Foot', 'Right Foot', 'Neck',
        'Left Shoulder (Inner)', 'Right Shoulder (Inner)', 'Head',
        'Left Shoulder (Outer)', 'Right Shoulder (Outer)', 'Left Elbow',
        'Right Elbow', 'Left Wrist', 'Right Wrist', 'Left Hand', 'Right Hand']

    KEYPOINTS = [
        'Nose', 'Chest', 'Right Shoulder', 'Right Elbow', 'Right Wrist',
        'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Pelvis',
        'Right Hip', 'Right Knee', 'Right Ankle',
        'Left Hip', 'Left Knee', 'Left Ankle', 
        'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear',
        'Left BToe', 'Left SToe', 'Left Heel',
        'Right BToe', 'Right SToe', 'Right Heel',]

    FOOT_VERTS = {
        'Left': [3292, 3368, 3387],
        'Right': [6691, 6768, 6786],
    }

#     REDUCED_JOINTS_IDXS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #, 18, 19] (used later in attempts, uncomment and comment next line if necessary)
    REDUCED_JOINTS_IDXS = [0, 1, 2, 4, 5, 6]
    REDUCED_KEYPOINTS_IDXS = [0, 1, 9, 10, 11, 12, 13, 14]
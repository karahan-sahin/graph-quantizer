# 
# Description: Quantizer for Continuous Sign Language Recognition
# Author: Karahan Şahin

# Remove warnings from MediaPipe
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F

import mediapipe as mp

from lib.utils.pose import *

class Quantizer(nn.Module):
    """
    Quantizer for Continuous Sign Language Recognition

    Args:
        quantizer (nn.Module): base vq-vae based quantizer model
        num_frames (int): number of frames to consider
        stride (int): stride of the sliding window

    ### A. Usage:

    Define the quantizer model

    ```python
    quantizer = Quantizer(quantizer, num_frames=25, stride=1)
    quantized, indices = quantizer.quantize(x)
    ```

    If you want to transform the indices into one-hot, set `transform=True`

    ```python
    quantized, indices = quantizer.quantize(x, transform=True)
    ```

    Also process the video to get the pose estimation
        
    ```python
    x = Quantizer.process_video(video_path)
    ```
    """

    def __init__(self, 
                 base_model, 
                 num_frames=25, 
                 stride=1,
                 num_codebooks=1,
                 codebook_size=512):
        super(Quantizer, self).__init__()
        
        self.quantizer = torch.load(
            base_model,
            map_location=torch.device('cpu' if not torch.cuda.is_available() else 'cuda:0')
        )

        self.quantizer.eval()

        self.num_frames = num_frames
        self.stride = stride

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

    @torch.no_grad()
    def encode(self, x):
        x_hat = self.quantizer.encoder(x)
        quantized, indices, _ = self.quantizer.vq_vae(x_hat)
        return quantized, indices

    def quantize(self, x, transform=False):
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        # NOTE: x.shape = (B, C, D, H, W)
        # Slice input channels
        x = x[:, :self.quantizer.encoder.input_size[1], :, :, :]
        quantized, indices = self.encode(x)
        if transform:
            indices = indices.view(-1, self.num_codebooks)
            # turn indices into one-hot
            indices = F.one_hot(indices, num_classes= self.num_codebooks * self.codebook_size).float()

        return quantized, indices

    # @staticmethod
    def process_video(self, video_path: str):

        pose, _ = get_pose_estimation(video_path)
        pose_array = get_pose_array(pose)
        # Replace missing values with zeros
        pose_array = pose_array.replace(np.nan, 0)
        matrices = get_matrices(pose_array)

        # Generate overlapping windows
        windows = []
        for i in range(0, len(matrices) - self.num_frames, self.stride):
            window = matrices[i:i+self.num_frames]
            windows.append(window)
        
        # Convert to tensor
        data = torch.tensor(windows).float()
        
        return data
    








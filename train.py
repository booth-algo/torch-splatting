import torch
import numpy as np
import gaussian_splatting.utils as utils
from gaussian_splatting.trainer import Trainer
import gaussian_splatting.utils.loss_utils as loss_utils
from gaussian_splatting.utils.data_utils import read_all
from gaussian_splatting.utils.camera_utils import to_viewpoint_camera
from gaussian_splatting.utils.point_utils import get_point_clouds
from gaussian_splatting.gauss_model import GaussModel
from gaussian_splatting.gauss_render import GaussRenderer

import contextlib

from torch.profiler import profile, ProfilerActivity

import torch
from torch.fx import symbolic_trace
import torch.nn as nn
from chop import MaseGraph
import chop.passes as passes
from torch.nn import Conv1d
import torch.nn.functional as F
import time
import os
from pathlib import Path
import time

USE_GPU_PYTORCH = True
USE_PROFILE = False

class GSSTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = kwargs.get('data')
        self.gaussRender = GaussRenderer(**kwargs.get('render_kwargs', {}))
        self.lambda_dssim = 0.2
        self.lambda_depth = 0.0
        self.psnr_values = []
        self.min_psnr = float('inf')
        self.max_psnr = float('-inf')

        model_output = self.model()
        self.xyz_gradient_accum = torch.zeros_like(model_output['xyz'])
        self.denom = torch.zeros(model_output['xyz'].shape[0], 1, device=model_output['xyz'].device)

    def get_viewspace_points(self, camera):
        model_output = self.model()
        world_points = model_output['xyz']
        return camera.world_to_view(world_points)

    def get_visibility_filter(self, camera):
        viewspace_points = self.get_viewspace_points(camera)
        return viewspace_points[:, 2] > 0

    def accumulate_gradient_statistics(self):
        # This method will be called after backward() in the training loop
        model_output = self.model()
        xyz = model_output['xyz']
        
        if xyz.grad is not None:
            visibility_filter = self.get_visibility_filter(self.current_camera)
            grad_norm = torch.norm(xyz.grad[visibility_filter, :2], dim=-1, keepdim=True)
            self.xyz_gradient_accum[visibility_filter] += grad_norm
            self.denom[visibility_filter] += 1

    def on_train_step(self):
        ind = np.random.choice(len(self.data['camera']))
        camera_params = self.data['camera'][ind]
        self.current_camera = to_viewpoint_camera(camera_params)
        rgb = self.data['rgb'][ind]
        depth = self.data['depth'][ind]
        mask = (self.data['alpha'][ind] > 0.5)

        pc_output = self.model()
        out = self.gaussRender(pc_output=pc_output, camera=self.current_camera)

        l1_loss = loss_utils.l1_loss(out['render'], rgb)
        depth_loss = loss_utils.l1_loss(out['depth'][..., 0][mask], depth[mask])
        ssim = loss_utils.ssim(out['render'], rgb)
        ssim_loss = 1.0 - ssim

        total_loss = (1-self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss + depth_loss * self.lambda_depth

        total_loss.backward()
        self.accumulate_gradient_statistics()

        psnr = utils.img2psnr(out['render'], rgb)

        allocated, cached = self.log_gpu_usage()

        # Update PSNR tracking
        self.psnr_values.append(psnr)
        self.min_psnr = min(self.min_psnr, psnr)
        self.max_psnr = max(self.max_psnr, psnr)

        log_dict = {
            'total': total_loss,
            'l1': l1_loss,
            'ssim': ssim,
            'depth_loss': depth_loss,
            'psnr': psnr,
            'min_psnr': self.min_psnr,
            'max_psnr': self.max_psnr,
            'gpu_memory_allocated': allocated, 
            'gpu_memory_cached': cached
        }

        self.final_log_dict = log_dict

        return total_loss, log_dict
    
    def log_psnr_stats(self):
        if self.psnr_values:
            avg_psnr = sum(self.psnr_values) / len(self.psnr_values)
            print(f"PSNR Stats - Avg: {avg_psnr:.2f}, Min: {self.min_psnr:.2f}, Max: {self.max_psnr:.2f}")
            
    def on_evaluate_step(self, **kwargs):
        import matplotlib.pyplot as plt
        ind = np.random.choice(len(self.data['camera']))
        # camera = self.data['camera'][ind]
        # if USE_GPU_PYTORCH:
        #     camera = to_viewpoint_camera(camera)

        camera = to_viewpoint_camera(self.data['camera'][ind])

        rgb = self.data['rgb'][ind].detach().cpu().numpy()

        pc_output = self.model()
        # out = self.gaussRender(pc=self.model, camera=camera)
        out = self.gaussRender(pc_output=pc_output, camera=camera)

        rgb_pd = out['render'].detach().cpu().numpy()
        depth_pd = out['depth'].detach().cpu().numpy()[..., 0]
        depth = self.data['depth'][ind].detach().cpu().numpy()

        if depth.shape != depth_pd.shape:
            depth = np.resize(depth, depth_pd.shape)

        depth = np.concatenate([depth, depth_pd], axis=1)
        depth = (1 - depth / depth.max())
        depth = plt.get_cmap('jet')(depth)[..., :3]

        if rgb.shape != rgb_pd.shape:
            rgb = np.resize(rgb, rgb_pd.shape)

        image = np.concatenate([rgb, rgb_pd], axis=1)
        image = np.concatenate([image, depth], axis=0)
        utils.imwrite(str(self.results_folder / f'image-{self.step}.png'), image)

    def log_gpu_usage(self):
        allocated = torch.cuda.memory_allocated(0) / (1024 **2)
        cached = torch.cuda.memory_reserved(0) / (1024 **2)
        return allocated, cached
    

def get_test_folder(base_folder='result', prefix='test'):
    """ Finds next available test folder in results folder """

    base_path = Path(base_folder)
    base_path.mkdir(parents=True, exist_ok=True)
    
    test_folders = [f.name for f in base_path.iterdir() if f.is_dir() and f.name.startswith(prefix)]
    
    test_numbers = [int(f[len(prefix):]) for f in test_folders if f[len(prefix):].isdigit()]
    
    next_test_number = max(test_numbers) + 1 if test_numbers else 0

    return f"{prefix}{next_test_number}"    


if __name__ == "__main__":
    device = 'cuda'
    folder = './training-data/B075X65R3X'
    data = read_all(folder, resize_factor=0.5)
    data = {k: v.to(device) for k, v in data.items()}
    data['depth_range'] = torch.Tensor([[1,3]]*len(data['rgb'])).to(device)

    points = get_point_clouds(data['camera'], data['depth'], data['alpha'], data['rgb'])
    random_samp = 2**13
    raw_points = points.random_sample(random_samp)
    # raw_points.write_ply(open('points.ply', 'wb'))

    full_width = 4
    frac_width = 2

    quant_config = {
        "by": "type",
        "default": {
            "config": {
                "name": None,
            }
        },
        "linear": {
            "config": {
                "name": "block_fp",
                # data
                "data_in_width": full_width,
                "data_in_frac_width": frac_width,
                # weight
                "weight_width": full_width,
                "weight_frac_width": frac_width,
                # bias
                "bias_width": full_width,
                "bias_frac_width": frac_width,
            }
        },
    }


    GaussModel = GaussModel(sh_degree=4, debug=False)
    GaussModel.create_from_pcd(pcd=raw_points)

    render_kwargs = {
        'white_bkgd': True,
    }
    
    # traced_model = symbolic_trace(GaussModel)
    # traced_model.graph.print_tabular()

    newGaussModel = MaseGraph(GaussModel)

    newGaussModel, _ = passes.init_metadata_analysis_pass(newGaussModel)
    newGaussModel, _ = passes.add_common_metadata_analysis_pass(newGaussModel)
    newGaussModel, _ = passes.quantize_transform_pass(newGaussModel, quant_config)

    results_folder = get_test_folder()

    trainer = GSSTrainer(
        model=newGaussModel.model, 
        # model=GaussModel,
        data=data,
        train_batch_size=1, 
        train_num_steps=100,
        i_image =100,
        train_lr=1e-3, 
        amp=False,
        fp16=False,
        results_folder=f'result/{results_folder}',
        render_kwargs=render_kwargs,

        # Densification settings
        densify_from_iter=5,
        densify_until_iter=500,
        densification_interval=2,
        opacity_reset_interval=500,
        densify_grad_threshold=0.01,
        min_opacity=0.005,
        scene_extent=1.0,
        size_threshold=20,
    )

    start_time = time.time()

    trainer.on_evaluate_step()
    trainer.train()

    end_time = time.time()
    total_time = end_time - start_time

    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\nTraining took {hours:.0f} hours, {minutes:.0f} minutes, {seconds:.0f} seconds")
    print("\nFinal Training Log: ", trainer.final_log_dict)
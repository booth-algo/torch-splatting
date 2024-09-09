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
    
    def on_train_step(self):
        ind = np.random.choice(len(self.data['camera']))
        camera = self.data['camera'][ind]
        rgb = self.data['rgb'][ind]
        depth = self.data['depth'][ind]
        mask = (self.data['alpha'][ind] > 0.5)
        if USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)

        if USE_PROFILE:
            prof = profile(activities=[ProfilerActivity.CUDA], with_stack=True)
        else:
            prof = contextlib.nullcontext()

        with prof:
            pc_output = self.model()
            out = self.gaussRender(pc_output=pc_output, camera=camera)
            # out = self.gaussRender(pc=self.model, camera=camera)

        if USE_PROFILE:
            print(prof.key_averages(group_by_stack_n=True).table(sort_by='self_cuda_time_total', row_limit=20))


        l1_loss = loss_utils.l1_loss(out['render'], rgb)
        depth_loss = loss_utils.l1_loss(out['depth'][..., 0][mask], depth[mask])
        ssim_loss = 1.0-loss_utils.ssim(out['render'], rgb)

        total_loss = (1-self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss + depth_loss * self.lambda_depth
        psnr = utils.img2psnr(out['render'], rgb)

        # Update PSNR tracking
        self.psnr_values.append(psnr)
        self.min_psnr = min(self.min_psnr, psnr)
        self.max_psnr = max(self.max_psnr, psnr)

        log_dict = {'total': total_loss,'l1':l1_loss, 'ssim': ssim_loss, 'depth': depth_loss, 'psnr': psnr, 'min_psnr': self.min_psnr, 'max_psnr': self.max_psnr}

        return total_loss, log_dict
    
    def log_psnr_stats(self):
        if self.psnr_values:
            avg_psnr = sum(self.psnr_values) / len(self.psnr_values)
            print(f"PSNR Stats - Avg: {avg_psnr:.2f}, Min: {self.min_psnr:.2f}, Max: {self.max_psnr:.2f}")
            
    def on_evaluate_step(self, **kwargs):
        import matplotlib.pyplot as plt
        ind = np.random.choice(len(self.data['camera']))
        camera = self.data['camera'][ind]
        if USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)

        rgb = self.data['rgb'][ind].detach().cpu().numpy()

        pc_output = self.model()
        # out = self.gaussRender(pc=self.model, camera=camera)
        out = self.gaussRender(pc_output=pc_output, camera=camera)

        rgb_pd = out['render'].detach().cpu().numpy()
        depth_pd = out['depth'].detach().cpu().numpy()[..., 0]
        depth = self.data['depth'][ind].detach().cpu().numpy()
        depth = np.concatenate([depth, depth_pd], axis=1)
        depth = (1 - depth / depth.max())
        depth = plt.get_cmap('jet')(depth)[..., :3]
        image = np.concatenate([rgb, rgb_pd], axis=1)
        image = np.concatenate([image, depth], axis=0)
        utils.imwrite(str(self.results_folder / f'image-{self.step}.png'), image)


if __name__ == "__main__":
    device = 'cuda'
    folder = './B075X65R3X'
    data = read_all(folder, resize_factor=0.5)
    data = {k: v.to(device) for k, v in data.items()}
    data['depth_range'] = torch.Tensor([[1,3]]*len(data['rgb'])).to(device)


    points = get_point_clouds(data['camera'], data['depth'], data['alpha'], data['rgb'])
    random_samp = 2**13
    raw_points = points.random_sample(random_samp)
    # raw_points.write_ply(open('points.ply', 'wb'))

    quant_config = {
        "by": "type",
        "default": {
            "config": {
                "name": None,
            }
        },
        "linear": {
            "config": {
                "name": "integer",
                # data
                "width": 2,
                "frac_width": 1
            }
        },
    }


    GaussModel = GaussModel(sh_degree=4, debug=False)
    GaussModel.create_from_pcd(pcd=raw_points)
    
    render_kwargs = {
        'white_bkgd': True,
    }
    
    traced_model = symbolic_trace(GaussModel)
    # print(traced_model.graph)
    traced_model.graph.print_tabular()

    newGaussModel = MaseGraph(GaussModel)

    newGaussModel, _ = passes.init_metadata_analysis_pass(newGaussModel)
    newGaussModel, _ = passes.add_common_metadata_analysis_pass(newGaussModel)

    newGaussModel, _ = passes.quantize_transform_pass(newGaussModel, quant_config)

    trainer = GSSTrainer(
        model=newGaussModel.model, 
        # model=GaussModel,
        data=data,
        train_batch_size=1, 
        train_num_steps=1000,
        i_image =100,
        train_lr=1e-3, 
        amp=False,
        fp16=False,
        results_folder='result/test',
        render_kwargs=render_kwargs,
    )

    trainer.on_evaluate_step()
    trainer.train()
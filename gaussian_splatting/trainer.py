import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.optim import Adam
from pathlib import Path
import os
from tqdm import tqdm
from gaussian_splatting.gauss_render import strip_symmetric, inverse_sigmoid, build_scaling_rotation, build_rotation

def exists(x):
    return x is not None

class Trainer(object):
    """
    Basic Trainer
    
    model (NeRFRender): nerf renderer
    sampler (RaySampler): ray sampler
    batch_size (int): Batch size per GPU
    batch_split (bool): Split batch with gradient accumulation. 
    training_lr (float): training leraning rate
    training_num_steps (int): training steps
    learning_rate (float): Learning rate.
    result_folder (str): Output directory.
    amp: if use amp.
    fp16: if use fp16.

    >>> renderer = create_renderer(...)
    >>> ray_sampler = create_ray_sampler(...)
    >>> trainer = Trainer(model=model, ray_sampler=ray_sampler, ....)
    >>> trainer.run()
    """
    def __init__(
        self,
        model,
        *, 
        sampler=None,
        results_folder='./result', 
        train_lr=1e-2,
        train_batch_size=4096,
        train_num_steps=25000,
        gradient_accumulate_every=1,
        adam_betas=(0.9,0.99),
        i_print=100,
        i_image=1000,
        i_save=50000,
        split_batches=False,
        amp=False,
        fp16=False,
        with_tracking=False,
        densify_from_iter=1000,
        densify_until_iter=5000,
        densification_interval=100,
        opacity_reset_interval=500,
        densify_grad_threshold=0.01,
        min_opacity=0.005,
        scene_extent=1.0,
        size_threshold=20,
        **kwargs,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision = 'fp16' if fp16 else 'no',
            project_dir=results_folder if with_tracking else None,
            log_with="all",
        )

        self.accelerator.native_amp = amp

        self.model = model
        self.sampler = sampler

        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densification_interval = densification_interval
        self.opacity_reset_interval = opacity_reset_interval
        self.densify_grad_threshold = densify_grad_threshold
        self.min_opacity = min_opacity
        self.scene_extent = scene_extent
        self.size_threshold = size_threshold # max_screen_size

        self.train_num_steps = train_num_steps
        self.i_save = i_save
        self.i_print = i_print
        self.i_image = i_image
        self.train_batch_size = train_batch_size

        self.results_folder = results_folder
        self.gradient_accumulate_every = gradient_accumulate_every
        self.with_tracking = with_tracking
        self.step = 0

        # self.opt = Adam(self.model.parameters(), lr=train_lr, betas=adam_betas)

        self.opt = Adam([
            {'params': self.model._xyz, 'name': 'xyz'},
            {'params': self.model._features_dc, 'name': 'f_dc'},
            {'params': self.model._features_rest, 'name': 'f_rest'},
            {'params': self.model._scaling, 'name': 'scaling'},
            {'params': self.model._rotation, 'name': 'rotation'},
            {'params': self.model._opacity, 'name': 'opacity'},
        ], lr = train_lr, betas = adam_betas)
        
        if self.accelerator.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # accelerator tracking
        if self.with_tracking:
            run = os.path.split(__file__)[-1].split(".")[0]
            self.accelerator.init_trackers(run, config={
                'train_lr':train_lr,
                'train_batch_size':train_batch_size,
                'gradient_accumulate_every':gradient_accumulate_every,
                'train_num_steps':train_num_steps,
            })
    
    def log_point_count(self):
        ''' Helper function to log and observe change in the number of points in the model '''
        model_output = self.model()
        num_points = model_output['xyz'].shape[0]
        print(f"Step {self.step}: Number of points: {num_points}")

    def perform_densification_and_pruning(self):
        model_output = self.model()
        xyz = model_output['xyz']
        opacity = model_output['opacity']
        scaling = model_output['scaling']

        if self.step < self.densify_from_iter:
            return

        if self.step > self.densify_from_iter and self.step % self.densification_interval == 0:

            initial_point_count = model_output['xyz'].shape[0]

            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0

            self.densify_and_clone(grads, self.densify_grad_threshold, self.scene_extent)
            self.densify_and_split(grads, xyz, opacity, scaling)

            if self.step % self.opacity_reset_interval == 0:
                self.reset_opacity()

        # Update optimizer parameters
        self.opt.param_groups[0]['params'] = list(self.model.parameters())

        # Re-prepare model after densification 
        #! not sure if this affects momentum of Adam optimizer
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
            
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        
        with tqdm(
            initial = self.step, 
            total = self.train_num_steps, 
            disable = not accelerator.is_main_process, 
            dynamic_ncols = True,
            ncols = None
        ) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.
                log_dict = {}

                for _ in range(self.gradient_accumulate_every):

                    with self.accelerator.autocast():
                        loss, step_log_dict = self.on_train_step()
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss

                    # Merge log dicts
                    for k, v in step_log_dict.items():
                        if k in log_dict:
                            log_dict[k] += v / self.gradient_accumulate_every
                        else:
                            log_dict[k] = v / self.gradient_accumulate_every

                self.opt.step()
                self.opt.zero_grad()

                # Densification and pruning
                if self.step < self.train_num_steps:
                    self.perform_densification_and_pruning()

                # all reduce to get the total loss
                total_loss = accelerator.reduce(total_loss)
                total_loss = total_loss.item()
                log_str = f'loss: {total_loss:.3f}'
                
                for k in log_dict.keys():
                    log_str += " {}: {:.3f}".format(k, log_dict[k])
                
                pbar.set_description(log_str)

                self.opt.step()
                self.opt.zero_grad()

                self.step += 1
                if accelerator.is_main_process:
                    
                    if (self.step % self.i_image == 0):
                        self.on_evaluate_step()

                    if self.step !=0 and (self.step % self.i_save == 0):
                        milestone = self.step // self.i_save
                        self.save(milestone)
                        self.log_psnr_stats()
                
                pbar.update(1)

        if accelerator.is_main_process:
            self.log_psnr_stats()

        if self.with_tracking:
            accelerator.end_training()

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
    
    def densify_and_clone(self, grads, grad_threshold, scene_extent):

        model_output = self.model()

        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, 
            torch.max(self.model._scaling, dim=1).values <= model_output['percent_dense'] * scene_extent
        )
        
        new_xyz = model_output['xyz'][selected_pts_mask]
        new_features_dc = model_output['features_dc'][selected_pts_mask]
        new_features_rest = model_output['features_rest'][selected_pts_mask]
        new_opacities = model_output['opacity'][selected_pts_mask]
        new_scaling = model_output['scaling'][selected_pts_mask]
        new_rotation = model_output['rotation'][selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_split(self, grads, xyz, opacity, scaling, N=2):
        n_init_points = xyz.shape[0]
        # Extract points that satisfy the gradient condition
        grad_threshold = self.densify_grad_threshold
        
        # Calculate the norm of the gradients
        grad_norms = torch.norm(grads, dim=1)
        
        # Create a padded version of grad_norms if necessary
        if grad_norms.shape[0] < n_init_points:
            padded_grad_norms = torch.zeros(n_init_points, device=grad_norms.device)
            padded_grad_norms[:grad_norms.shape[0]] = grad_norms
        else:
            padded_grad_norms = grad_norms

        # Create a mask for points that meet the gradient threshold
        selected_pts_mask = grad_norms >= grad_threshold
        
        # Apply the percent_dense condition
        percent_dense = self.model().get('percent_dense', 0.01)  # default to 0.01 if not present
        scene_extent = self.scene_extent
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(scaling, dim=1).values > percent_dense * scene_extent)

        if not selected_pts_mask.any():
            return

        stds = scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=xyz.device)
        samples = torch.normal(mean=means, std=stds)
        
        # Get rotation from the model output
        rotation = self.model().get('rotation')
        rots = build_rotation(rotation[selected_pts_mask]).repeat(N, 1, 1)
        
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = torch.log(scaling[selected_pts_mask].repeat(N, 1) / (0.8*N)) # Inverse scaling activation
        new_rotation = rotation[selected_pts_mask].repeat(N, 1)
        
        # Get features from the model output
        features = self.model().get('features')
        new_features_dc = features[:, :, 0][selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = features[:, :, 1:][selected_pts_mask].repeat(N, 1, 1)
        
        new_opacity = opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=xyz.device, dtype=bool)))
        self.prune_points(prune_filter)

        
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity() < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling().max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity(), torch.ones_like(self.get_opacity())*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.opt.param_groups:
            if group["name"] in tensors_dict:
                extension_tensor = tensors_dict[group["name"]]
                param = group["params"][0]  # Assume the first parameter is the one we want to extend
                
                if extension_tensor.numel() == 0:
                    continue

                if param in self.opt.state:
                    state = self.opt.state[param]
                    for key in state:
                        if isinstance(state[key], torch.Tensor):
                            # Ensure state tensor is on the same device as param
                            state[key] = state[key].to(param.device)
                            if state[key].dim() == 0:  # If it's a scalar
                                state[key] = state[key].unsqueeze(0)  # Make it a 1D tensor
                            zeros = torch.zeros(extension_tensor.shape[0], *state[key].shape[1:], device=param.device)
                            state[key] = torch.cat((state[key], zeros), dim=0)

                new_param = nn.Parameter(torch.cat((param.data, extension_tensor), dim=0).requires_grad_(True))
                
                group["params"][0] = new_param
                if param in self.opt.state:
                    self.opt.state[new_param] = self.opt.state.pop(param)
                
                optimizable_tensors[group["name"]] = new_param

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation):

        if new_xyz.numel() == 0:
            return
        
        device = self.model._xyz.device  # Get the device of an existing model parameter
        
        d = {
            "xyz": new_xyz.to(device),
            "f_dc": new_features_dc.to(device),
            "f_rest": new_features_rest.to(device),
            "opacity": new_opacity.to(device),
            "scaling": new_scaling.to(device),
            "rotation": new_rotation.to(device)
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        model_output = self.model()
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((model_output['xyz'].shape[0], 1), device="cuda")
        self.denom = torch.zeros((model_output['xyz'].shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((model_output['xyz'].shape[0]), device="cuda")
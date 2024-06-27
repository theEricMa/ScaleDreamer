import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.mesh import Mesh
from threestudio.utils.misc import broadcast, get_rank, C
from threestudio.utils.typing import *

from .utils import contract_to_unisphere_custom
from threestudio.utils.ops import get_activation
from threestudio.models.networks import get_encoding, get_mlp

from .utils import get_trilinear_feature

@threestudio.register("3DConv-net")
class Voxel_3d_Sdf(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3
        space_generator_config: dict = field(
            default_factory=lambda: {
                "z_dim": 512,
                "w_dim": 512,
                "num_layers": 2,
                "img_resolution": 128, # the resolution of tri-plane, bigger or smaller?
                "img_channels": 32, # for tri-plane
                "channel_multiplier": 1,
            }
        )

        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )

        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )

        backbone: str = "3dconv_net"
        truncation_psi: float = 1.0
        activation: str = "none"

        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']
        finite_difference_normal_eps: Union[
            float, str
        ] = 0.01  # in [float, "progressive"]
        shape_init: Optional[str] = None
        shape_init_params: Optional[Any] = None
        shape_init_mesh_up: str = "+z"
        shape_init_mesh_front: str = "+x"
        force_shape_init: bool = False
        sdf_bias: Union[float, str] = 0.0
        sdf_bias_params: Optional[Any] = None

        # no need to removal outlier for SDF
        isosurface_remove_outliers: bool = False

    def configure(self) -> None:
        super().configure()

        # set up the space generator

        if self.cfg.backbone == "3dconv_net":
            from ...extern.stylegan_3dconv_modules import Generator
            self.space_generator = Generator(**self.cfg.space_generator_config)


        # set up the mlp
        input_dim = self.cfg.space_generator_config.img_channels 

        self.sdf_network = get_mlp(
            input_dim,
            1,
            self.cfg.mlp_network_config,
        )

        if self.cfg.n_feature_dims > 0:
            self.feature_network = get_mlp(
                input_dim,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )

        if self.cfg.normal_type == "pred":
            raise NotImplementedError("normal_type == pred is not implemented yet.")
        
        if self.cfg.isosurface_deformable_grid:
            assert (
                self.cfg.isosurface_method == "mt"
            ), "isosurface_deformable_grid only works with mt"
            self.deformation_network = get_mlp(
                input_dim,
                3,
                self.cfg.mlp_network_config,
            )

        self.noise_dim = self.cfg.space_generator_config.z_dim
        self.finite_difference_normal_eps: Optional[float] = None

    def initialize_shape(self) -> None:
        if self.cfg.shape_init is None and not self.cfg.force_shape_init:
            return

       # do not initialize shape if weights are provided
        if self.cfg.weights is not None and not self.cfg.force_shape_init:
            return
        
        if self.cfg.sdf_bias != 0.0:
            threestudio.warn(
                "shape_init and sdf_bias are both specified, which may lead to unexpected results."
            )

        get_gt_sdf: Callable[[Float[Tensor, "N 3"]], Float[Tensor, "N 1"]]
        assert isinstance(self.cfg.shape_init, str)
        if self.cfg.shape_init == "ellipsoid":
            assert (
                isinstance(self.cfg.shape_init_params, Sized)
                and len(self.cfg.shape_init_params) == 3
            )
            size = torch.as_tensor(self.cfg.shape_init_params).to(self.device)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return ((points_rand / size) ** 2).sum(
                    dim=-1, keepdim=True
                ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid

            get_gt_sdf = func
        elif self.cfg.shape_init == "sphere":
            assert isinstance(self.cfg.shape_init_params, float)
            radius = self.cfg.shape_init_params

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return (points_rand**2).sum(dim=-1, keepdim=True).sqrt() - radius

            get_gt_sdf = func
        elif self.cfg.shape_init.startswith("mesh:"):
            raise NotImplementedError("mesh initialization is not implemented yet.")
        else:
            raise ValueError(
                f"Unknown shape initialization type: {self.cfg.shape_init}"
            )

        # Initialize SDF to a given shape when no weights are provided or force_shape_init is True
        batch_size = 4
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        from tqdm import tqdm
        for _ in tqdm(
            range(1000), # range(10)
            desc=f"Initializing SDF to a(n) {self.cfg.shape_init}:",
            disable=get_rank() != 0,
        ):
            # generate space cache
            space_cache = self.generate_space_cache(
                styles = torch.randn(batch_size, self.noise_dim).to(self.device),
                text_embed = None,
            )        
            # generate points
            points_rand = (
                torch.rand(
                    (batch_size, 10000, 3),
                    dtype=torch.float32
                ).to(self.device) * self.cfg.radius * 2 - self.cfg.radius
            ) # the points should be in [-1, 1]

            sdf_gt = get_gt_sdf(points_rand)
            sdf_pred = self.forward_sdf(points_rand, space_cache)
            # compute loss
            loss = F.mse_loss(sdf_pred, sdf_gt)
            # optimize
            optim.zero_grad()
            loss.backward()
            optim.step()

        # explicit broadcast to ensure param consistency across ranks
        for param in self.parameters():
            broadcast(param, src=0)

    # this function is similar to the one in threestudio/models/geometry/impcit_sdf.py
    def get_shifted_sdf(
        self, 
        points: Float[Tensor, "*N Di"], 
        sdf: Float[Tensor, "*N 1"]
    ) -> Float[Tensor, "*N 1"]:
        sdf_bias: Union[float, Float[Tensor, "*N 1"]]
        if self.cfg.sdf_bias == "ellipsoid":
            assert (
                isinstance(self.cfg.sdf_bias_params, Sized)
                and len(self.cfg.sdf_bias_params) == 3
            )
            size = torch.as_tensor(self.cfg.sdf_bias_params).to(points)
            sdf_bias = ((points / size) ** 2).sum(
                dim=-1, keepdim=True
            ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid
        elif self.cfg.sdf_bias == "sphere":
            assert isinstance(self.cfg.sdf_bias_params, float)
            radius = self.cfg.sdf_bias_params
            sdf_bias = (points**2).sum(dim=-1, keepdim=True).sqrt() - radius
        elif isinstance(self.cfg.sdf_bias, float):
            sdf_bias = self.cfg.sdf_bias
        else:
            raise ValueError(f"Unknown sdf bias {self.cfg.sdf_bias}")
        return sdf + sdf_bias

    def generate_space_cache(
        self,
        styles: Float[Tensor, "B Z"],
        text_embed: Optional[Float[Tensor, "B T"]] = None,
    ) -> Any:
        output_dict = self.space_generator(
            z = styles,
            c = text_embed,
            truncation_psi = self.truncation_psi,
        )

        space_cache = output_dict['image']
        # activate the output
        space_cache = get_activation(self.cfg.activation)(space_cache)

        return space_cache


    def interpolate_encodings(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Float[Tensor, "B 3 C//3 H W"],
    ):
        batch_size, n_points, n_dims = points.shape
        enc_prompt = get_trilinear_feature(
                points=points,
                voxel=space_cache,
            ).reshape(*points.shape[:-1],-1)
        

        enc = enc_prompt

        return enc
            
    def forward(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Float[Tensor, "B 3 C//3 H W"],
        output_normal: bool = False,
    ):

        batch_size, n_points, n_dims = points.shape
        points_unscaled = points
        points = contract_to_unisphere_custom(
            points, 
            self.bbox, 
            self.unbounded
        )  # points normalized to (1, 1)

        if output_normal and self.cfg.normal_type == "analytic":
            raise NotImplementedError("analytic normal is not implemented yet.")

        enc = self.interpolate_encodings(points, space_cache)
        sdf = self.sdf_network(enc).view(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        output = {
                "sdf": sdf.view(batch_size * n_points, 1) # reshape to [B*N, 1]
            }
        
        if self.cfg.n_feature_dims > 0:
            features = self.feature_network(enc).view(
                *points.shape[:-1], self.cfg.n_feature_dims)
            output.update(
                    {
                        "features": features.view(batch_size * n_points, self.cfg.n_feature_dims) # reshape to [B*N, n_feature_dims]
                    }
                )

        if output_normal:
            if (
                self.cfg.normal_type == "finite_difference"
                ):
                assert self.finite_difference_normal_eps is not None
                eps: float = self.finite_difference_normal_eps
                offsets: Float[Tensor, "3 3"] = torch.as_tensor(
                    [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
                ).to(points_unscaled)
                points_offset: Float[Tensor, "... 3 3"] = (
                    points_unscaled[..., None, :] + offsets
                ).clamp(-self.cfg.radius, self.cfg.radius)
                sdf_offset: Float[Tensor, "... 3 1"] = self.forward_sdf(
                    points_offset, space_cache
                )
                sdf_grad = (sdf_offset[..., 0::1, 0] - sdf) / eps
                normal = F.normalize(sdf_grad, dim=-1)
            else:
                raise NotImplementedError(
                    f"normal_type == {self.cfg.normal_type} is not implemented yet."
                )
            output.update(
                {
                    "normal": normal.view(batch_size * n_points, 3), # reshape to [B*N, 3]
                    "shading_normal": normal.view(batch_size * n_points, 3), # reshape to [B*N, 3]
                    "sdf_grad": sdf_grad.view(batch_size * n_points, 3), # reshape to [B*N, 3]
                }
            )
        return output

    def forward_sdf(
        self,
        points: Float[Tensor, "*N Di"],
        space_cache: Float[Tensor, "B 3 C//3 H W"],
    ) -> Float[Tensor, "*N 1"]:
        batch_size = points.shape[0]
        assert points.shape[0] == batch_size, "points and space_cache should have the same batch size in forward_sdf"
        points_unscaled = points

        points = contract_to_unisphere_custom(
            points_unscaled, 
            self.bbox, self.unbounded
        )   # points normalized to (-1, 1)

        # sample from planes
        enc = self.interpolate_encodings(
            points.reshape(batch_size, -1, 3),
            space_cache
        ).reshape(*points.shape[:-1], -1)
        sdf = self.sdf_network(enc).reshape(*points.shape[:-1], 1)

        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        return sdf
    
    def forward_field(
        self, 
        points: Float[Tensor, "*N Di"],
        space_cache: Float[Tensor, "B 3 C//3 H W"],
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        # TODO: is this function correct?
        batch_size = points.shape[0]
        assert points.shape[0] == batch_size, "points and space_cache should have the same batch size in forward_sdf"
        points_unscaled = points

        points = contract_to_unisphere_custom(
            points_unscaled, 
            self.bbox, self.unbounded
        )   # points normalized to (-1, 1)

        # sample from planes
        enc = self.interpolate_encodings(points, space_cache)      
        sdf = self.sdf_network(enc).reshape(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        deformation: Optional[Float[Tensor, "*N 3"]] = None
        if self.cfg.isosurface_deformable_grid:
            deformation = self.deformation_network(enc).reshape(*points.shape[:-1], 3)
        return sdf, deformation
    
    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        # TODO: is this function correct?
        return field - threshold
    
    def export(
        self, 
        points: Float[Tensor, "*N Di"], 
        space_cache: Float[Tensor, "B 3 C//3 H W"],
    **kwargs) -> Dict[str, Any]:
        # TODO: is this function correct?
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere_custom(
            points_unscaled, 
            self.bbox, 
            self.unbounded
        )

        # sample from planes
        enc = self.interpolate_encodings(points, space_cache)
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {
                "features": features,
            }
        )
        return out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        if (
            self.cfg.normal_type == "finite_difference"
        ):
            if isinstance(self.cfg.finite_difference_normal_eps, float):
                self.finite_difference_normal_eps = (
                    self.cfg.finite_difference_normal_eps
                )
        else:
            raise NotImplementedError(
                f"normal_type == {self.cfg.normal_type} is not implemented yet."
            )
        
        # set the truncation_psi
        self.truncation_psi = C(
            self.cfg.truncation_psi, epoch, global_step, 
        )
            
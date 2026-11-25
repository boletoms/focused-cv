"""R10 strong video head: Cutie (hkchengrex/Cutie @ ec5cdd4, cloned untouched in /work/third_party/Cutie) with its
pixel encoder replaced. Cutie's PixelEncoder returns (f16, f8, f4); here f16 comes from the frozen DINOv3 ViT through
our EffModel, so any Stage B method (dense, our budget, protected ALGM, ...) and focus map applies exactly where the
efficiency claim is; f8 and f4 (decoder skips) come from Cutie's own ResNet-18 stem (conv1 to layer2, ImageNet init,
trained at Cutie's backbone learning-rate ratio, BN statistics frozen as in Cutie). Everything else (key projection,
mask encoder, object transformer, memory, decoder, losses, data pipeline) is Cutie's code."""
import os

import torch
import torch.nn as nn

from src.stage_a.model import N_PREFIX, load_backbone
from src.stage_b.model import EffModel

VIT_REPO = os.environ.get("VOS_VIT", "facebook/dinov3-vits16-pretrain-lvd1689m")
VIT_WIDTH = {"vits16": 384, "vitb16": 768, "vitl16": 1024}[VIT_REPO.split("/")[-1].split("-")[1]]   # stride-16 channels (R25)


class DinoPixelEncoder(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        from cutie.model.utils import resnet
        net = resnet.resnet18(pretrained=True, model_dir=model_cfg.get("resnet_model_path"))
        self.conv1, self.bn1, self.relu, self.maxpool = net.conv1, net.bn1, net.relu, net.maxpool
        self.res2, self.layer2 = net.layer1, net.layer2
        self.vit = EffModel(load_backbone("cpu", VIT_REPO)).requires_grad_(False)
        self.structure, self.interest, self.states = None, None, []      # set by the evaluation loop
        self.frame_inputs, self.frame_setting, self.frame_out, self.frames = None, None, None, {}   # VOS_GRAPH=2
        self.prev_mask = None    # (h, w) bool, unpadded: the focus map is built from it at the encoder's input size
        self.train_maps, self.train_fb = None, None   # R31: per-image focus maps (B, H, W) set by the trainer
        comp = os.environ.get("VOS_COMPRESS_TRAIN")                      # R31: our method in the loop during training
        if comp:
            from src.stage_b.timing import from_name
            name, _, fb = comp.partition("~")
            blocks = len(self.vit.bb.model.layer)
            self.structure = from_name(name, 30, blocks)                 # 480 x 480 training crops: 30 x 30 patches
            self.train_fb = from_name(fb, 30, blocks) if fb else None
        tag = os.environ.get("VOS_DTEM_TRAIN")                           # R21: train DTEM's metric layers through Cutie
        if tag:
            from src.stage_b.dtem import DTEM, MetricHeads
            v = self.vit
            self.dtem_heads = MetricHeads(len(v.bb.model.layer), v.bb.config.hidden_size, v.hd)
            self.structure = DTEM(64, first=0, heads=self.dtem_heads, tag=tag)
            self.structure.soft = True

    def _train_compressed(self, x):
        """R31: each image with its own focus map (the trainer's maps at the input size); an empty map runs the
        fallback (the uniform reset), or dense without one. The ViT is frozen; no gradient reaches it."""
        B, _, H, W = x.shape
        toks = []
        with torch.no_grad():
            for i in range(B):
                interest = torch.nn.functional.avg_pool2d(self.train_maps[i].float()[None, None], 16)[0, 0].flatten()[None]
                m = self.structure if bool(interest.any()) else self.train_fb
                toks.append(self.vit(x[i:i + 1], structure=m, dtype=torch.bfloat16,
                                     interest=interest if m is self.structure else None).final[:, N_PREFIX:])
        return torch.cat(toks)

    def forward(self, x):
        B, _, H, W = x.shape
        if self.train_maps is not None:
            assert self.train_maps.shape == (B, H, W), (self.train_maps.shape, x.shape)
            tok = self._train_compressed(x)
            self.train_maps = None                                       # one use per forward
            f16 = tok.float().transpose(1, 2).reshape(B, -1, H // 16, W // 16)
            f4 = self.res2(self.maxpool(self.relu(self.bn1(self.conv1(x)))))
            return f16, self.layer2(f4), f4
        if self.prev_mask is not None and self.structure is not None and getattr(self.structure, "per_image", False):
            m = torch.zeros(H, W, device=x.device)
            h, w = self.prev_mask.shape
            lh, lw = (H - h) // 2, (W - w) // 2                            # pad_divide_by pads symmetrically
            m[lh:lh + h, lw:lw + w] = self.prev_mask.float()
            self.interest = torch.nn.functional.avg_pool2d(m[None, None], 16)[0, 0].flatten()[None]
        if os.environ.get("VOS_GRAPH") == "2" and self.frame_inputs is not None:   # R40 plan L: the frame runner
            key = (tuple(x.shape[-2:]), self.vit.fast, self.frame_setting)   # setting: one runner per method (audit)
            if key not in self.frames:                                   # one runner (its graphs) at a time: videos
                from src.stage_b.frame_runner import FrameRunner         # of other sizes would add GPU memory
                if self.frames:
                    self.frames.clear()
                    torch.cuda.empty_cache()
                self.frames[key] = FrameRunner(self.vit, x.shape[-2:], self.frame_setting, fast=self.vit.fast,
                                               kernel=os.environ.get("VOS_GRAPH_KERNEL", "mask"),
                                               compile={"1": True, "segment": "segment"}.get(
                                                   os.environ.get("VOS_GRAPH_COMPILE"), False),
                                               overlap=os.environ.get("VOS_GRAPH_OVERLAP") == "1")
            fr = self.frames[key]
            if self.frame_inputs[0] == "reset":
                fr.reset()
                self.frame_inputs = self.frame_inputs[1:]
            final, st, ops, fb = fr.run(x, *self.frame_inputs)
            self.frame_out = (st, ops, fb, fr)
            self.states = (self.states + [st])[-4:]
            tok = final[:, N_PREFIX:]
            f16 = tok.transpose(1, 2).reshape(B, -1, H // 16, W // 16)
            f4 = self.res2(self.maxpool(self.relu(self.bn1(self.conv1(x)))))
            return f16, self.layer2(f4), f4
        runner = self._graph_runner(x) if os.environ.get("VOS_GRAPH") == "1" else None
        if runner is not None:                                           # R40 plan K: the expectation focus map under CUDA graphs
            final, st = runner.run(x, self.structure, self.interest)
            self.states = (self.states + [st])[-4:]
            tok = final[:, N_PREFIX:]
            f16 = tok.transpose(1, 2).reshape(B, -1, H // 16, W // 16)
            f4 = self.res2(self.maxpool(self.relu(self.bn1(self.conv1(x)))))
            return f16, self.layer2(f4), f4
        soft = getattr(self.structure, "soft", False)                    # R21 training: gradients reach DTEM's
        with torch.set_grad_enabled(soft and torch.is_grad_enabled()):  # metric layers through the frozen ViT
            sts = []
            tok = self.vit(x, structure=self.structure, dtype=torch.bfloat16, interest=self.interest,
                           state_out=sts).final[:, N_PREFIX:]
            if not soft:
                self.states.append(sts[0])
                if len(self.states) > 4:
                    self.states = self.states[-4:]
        f16 = tok.float().transpose(1, 2).reshape(B, -1, H // 16, W // 16)
        f4 = self.res2(self.maxpool(self.relu(self.bn1(self.conv1(x)))))
        f8 = self.layer2(f4)
        return f16, f8, f4

    def _graph_runner(self, x):
        """VOS_GRAPH=1: the expectation focus map and its fallback (`ExpectMap` around ALGMTiered or plain ALGM) run through
        `tiered_graph.TieredRunner`, one runner per input size (the last one only); anything else returns None."""
        from src.stage_b.expect import ExpectMap
        from src.stage_b.tiered_graph import TieredRunner, spec_of
        if not (x.is_cuda and x.shape[0] == 1 and isinstance(self.structure, ExpectMap)):
            return None
        try:
            spec_of(self.structure)
        except AssertionError:
            return None
        key = (tuple(x.shape[-2:]), self.vit.fast)
        if getattr(self, "_runner_key", None) != key:
            self._runner = TieredRunner(self.vit, x.shape[-2:], fast=self.vit.fast,
                                        kernel=os.environ.get("VOS_GRAPH_KERNEL", "mask"))
            self._runner_key = key
        return self._runner

    def train(self, mode=True):                  # as Cutie: BN statistics frozen; the ViT is always in eval mode
        self.training = False
        for m in self.children():
            m.train(False)
        return self

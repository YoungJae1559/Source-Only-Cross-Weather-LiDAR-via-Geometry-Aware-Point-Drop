import os, sys, json, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PUDM.pointnet2.models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from PUDM.pointnet2.util import calc_diffusion_hyperparams, sampling
from PUDM.pointnet2.json_reader import restore_string_to_list_in_a_dict

ROOT = os.path.abspath(os.path.join(os.getcwd(), "PUDM"))
cfg_path = os.path.join(ROOT, "exp_configs", "PUGAN.json")
ckpt_path = os.path.join(ROOT, "pkls", "pugan.pkl")

cfg = restore_string_to_list_in_a_dict(json.load(open(cfg_path)))
pointnet_cfg = cfg["pointnet_config"]
diff_cfg = cfg["diffusion_config"]
scale = (cfg.get("pugan_dataset_config") or cfg.get("pu1k_dataset_config"))["scale"]

net = PointNet2CloudCondition(pointnet_cfg).eval().cuda()
sd = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
net.load_state_dict(sd, strict=False)

hp = calc_diffusion_hyperparams(**diff_cfg)
for k in list(hp.keys()):
    if k != "T": hp[k] = hp[k].cuda()

pts = torch.randn(2048, 3, device="cuda")
with torch.no_grad():
    up = sampling(net, hp, pts, scale, R=4, T=diff_cfg["T"], step=12, gamma=0.5, normalization=True)

print("UPSAMPLED SHAPE:", tuple(up.shape))

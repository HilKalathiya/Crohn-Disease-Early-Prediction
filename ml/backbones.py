"""Pretrained backbones loaded from local weight files (GitHub-release timm weights)."""
import os, timm, torch
WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "weights")
REGISTRY = {
    "effb0_ns": ("tf_efficientnet_b0.ns_jft_in1k", "tf_efficientnet_b0_ns-c0e6a31c.pth"),
    "effb0_ra": ("efficientnet_b0.ra_in1k", "efficientnet_b0_ra-3dd342df.pth"),
    "effv2s_21k": ("tf_efficientnetv2_s.in21k_ft_in1k", "tf_efficientnetv2_s_21ft1k-d7dafa41.pth"),
    "mnv3_21k": ("mobilenetv3_large_100.miil_in21k_ft_in1k", "mobilenetv3_large_100_1k_miil_78_0-66471c13.pth"),
    "effb3_ns": ("tf_efficientnet_b3.ns_jft_in1k", "tf_efficientnet_b3_ns-9d44bf68.pth"),
    "effv2b0": ("tf_efficientnetv2_b0.in1k", "tf_efficientnetv2_b0-c7cc451f.pth"),
}
def create(key, num_classes=7, pretrained=True, **kw):
    arch, f = REGISTRY[key]
    m = timm.create_model(arch, pretrained=False, num_classes=num_classes, **kw)
    if pretrained:
        sd = torch.load(os.path.join(WEIGHTS, f), map_location="cpu", weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.startswith(("classifier.", "fc.", "head.fc."))}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert all(k.startswith(("classifier", "fc", "head")) for k in missing), missing
    return m

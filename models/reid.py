"""
models/reid.py
==============
Loads the ReID backbone (FastReID → OSNet fallback → ResNet18 fallback)
and exposes:

  reid_fn          — callable: img_bgr (np.ndarray) → embedding (np.ndarray)
  REID_DIM         — int, updated at load time to match the chosen backbone
  load_reid()      — call once at startup; sets reid_fn and REID_DIM

All logic is preserved verbatim from the original monolith.
The only structural change is that REID_DIM is both read from and written
back to config.settings so that memory/faiss_index.py picks up the
correct dimension without a circular import.
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from config.settings import FASTREID_CONFIG, FASTREID_WEIGHTS

# Mutable module-level state
reid_fn  = None
REID_DIM = 2048    # overwritten by load_reid()


# --------------------------------------------------
def load_reid():
    """
    Entry point — tries FastReID, falls back to OSNet, then ResNet18.
    Updates reid_fn and REID_DIM in this module and in config.settings.
    """
    global reid_fn, REID_DIM
    reid_fn, REID_DIM = _load_fastreid()

    # Propagate resolved dim back to settings so faiss_index picks it up
    import config.settings as _cfg
    _cfg.REID_DIM = REID_DIM

    print("[INFO] ReID loaded!")


# --------------------------------------------------
def _load_fastreid():
    """Returns (extract_fn, dim).  Tries FastReID first."""
    try:
        from fastreid.config      import get_cfg
        from fastreid.modeling    import build_model
        from fastreid.utils.checkpoint import Checkpointer
        from torchvision import transforms

        cfg = get_cfg()
        cfg.merge_from_file(FASTREID_CONFIG)
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.MODEL.WEIGHTS           = ""
        cfg.freeze()

        fr_model = build_model(cfg)
        Checkpointer(fr_model).load(FASTREID_WEIGHTS)
        fr_model.eval()
        fr_model.to("cpu")

        reid_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        def extract_embedding(img_bgr):
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor  = reid_transform(img_rgb).unsqueeze(0).to("cpu")
            with torch.no_grad():
                feat = fr_model(tensor)
            if isinstance(feat, dict):
                feat = feat["features"]
            feat = F.normalize(feat, p=2, dim=1)
            return feat.squeeze(0).cpu().numpy()

        print("[ReID] FastReID ResNet50 loaded")
        return extract_embedding, 2048

    except Exception as e:
        print(f"[ReID] FastReID unavailable ({e}), falling back to OSNet/ResNet18")
        return _load_osnet_fallback()


# --------------------------------------------------
def _load_osnet_fallback():
    """Returns (extract_fn, dim).  Tries OSNet, then ResNet18."""
    try:
        import torchreid
        from torchvision import transforms

        reid_backbone = torchreid.models.build_model(
            name="osnet_x0_25", num_classes=1000, pretrained=True
        )
        reid_backbone.eval()
        reid_backbone.to("cpu")

        reid_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        def extract_embedding(img_bgr):
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor  = reid_transform(img_rgb).unsqueeze(0).to("cpu")
            with torch.no_grad():
                feat = reid_backbone(tensor)
            feat = F.normalize(feat, p=2, dim=1)
            return feat.squeeze(0).cpu().numpy()

        print("[ReID] OSNet-x0.25 loaded (torchreid fallback)")
        return extract_embedding, 512

    except ImportError:
        from torchvision.models import resnet18, ResNet18_Weights
        from torchvision import transforms

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
        backbone.eval()
        backbone.to("cpu")

        reid_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        def extract_embedding(img_bgr):
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor  = reid_transform(img_rgb).unsqueeze(0).to("cpu")
            with torch.no_grad():
                feat = backbone(tensor).squeeze()
            feat = F.normalize(feat, p=2, dim=0)
            return feat.cpu().numpy()

        print("[ReID] Fallback ResNet18 loaded")
        return extract_embedding, 512

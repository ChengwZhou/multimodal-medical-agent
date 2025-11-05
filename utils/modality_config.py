class ModalityConfig:
    def __init__(self, name: str, in_ch: None, patch_size: None, device_idx=None, lead_idx=None):
        self.name = name
        self.in_ch = in_ch
        self.patch_size = patch_size
        self.device_idx = device_idx
        self.lead_idx = lead_idx
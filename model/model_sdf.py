import torch
import torch.nn as nn
from .stable_diffusion.latent_diffusion import LatentDiffusion


class Diffpro_SDF(nn.Module):
    # send a latent diffusion model into it
    def __init__(
        self,
        ldm: LatentDiffusion,
    ):
        """
        cond_type: {chord, texture}
        cond_mode: {cond, mix, uncond}
            mix: use a special condition for unconditional learning with probability of 0.2
        use_enc: whether to use pretrained chord encoder to generate encoded condition
        """
        # actually is equivalent to super().__init__()
        super(Diffpro_SDF, self).__init__()
        # if there is gpu then use gpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ldm = ldm
    # note that for classmethod, it's "cls" being passed instead of "self", no need an instance to call this method
    # this is just pack initialization and load_trained into the same package
    # checkpoint file path
    @classmethod
    def load_trained(
        cls,
        ldm,
        chkpt_fpath,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ldm is a LatentDiffusion type object, the following is equivalent to calling __init__
        model = cls(ldm)
        trained_leaner = torch.load(chkpt_fpath, map_location=device)
        try:
            # the key name here is named by ourselves
            model.load_state_dict(trained_leaner["model"])
        except RuntimeError:
            model_dict = trained_leaner["model"]
            # old parameter names are like ldm.cond_enc.xxx.weight, ldm.style_enc.xxx.weight
            # new names are ldm.autoreg_cond_enc.xxx.weight, ldm.external_cond_enc.xxx.weight
            model_dict = {k.replace('cond_enc', 'autoreg_cond_enc'): v for k, v in model_dict.items()}
            model_dict = {k.replace('style_enc', 'external_cond_enc'): v for k, v in model_dict.items()}
            model.load_state_dict(model_dict)
        return model

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        return self.ldm.p_sample(xt, t)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        return self.ldm.q_sample(x0, t)

    def get_loss_dict(self, batch, step):
        """
        z_y is the stuff the diffusion model needs to learn
        """
        # x = batch.float().to(self.device)
        # batch shape: (tuple of x, tuple of autoreg_cond, tuple of external_cond), while these conditions will further go throught auto encoder into tuple with shape (B, 1, d)
        x, autoreg_cond, external_cond = batch
        loss = self.ldm.loss(x, autoreg_cond, external_cond)
        return {"loss": loss}

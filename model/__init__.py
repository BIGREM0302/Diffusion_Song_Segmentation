from .stable_diffusion.latent_diffusion import LatentDiffusion
from .model_sdf import Diffpro_SDF
from .stable_diffusion.model.unet import UNetModel
from .stable_diffusion.model.autoreg_cond_encoders import *
from .stable_diffusion.model.external_cond_encoders import *

autoreg_enc_dict = {'unsup': UnsupAutoregEncoder, 'frm': None, 'ctp': CtpAutoregEncoder, 'lsh': LshAutoregEncoder, 'acc': AccAutoregEncoder}
external_enc_dict = {'frm': None, 'ctp': CtpExternalEncoder, 'lsh': LshExternalEncoder, 'acc': AccExternalEncoder}

def init_ldm_model(mode, use_autoreg_cond, use_external_cond, params, debug_mode=False):
    # use_autoreg_cond, use_external_cond are boolean values,
    # params here specify the setting of UNet model, like the number of channels
    unet_model = UNetModel(
       in_channels=params.in_channels,
       out_channels=params.out_channels,
       channels=params.channels,
       attention_levels=params.attention_levels,
       n_res_blocks=params.n_res_blocks,
       channel_multipliers=params.channel_multipliers,
       n_heads=params.n_heads,
       tf_layers=params.tf_layers,
       d_cond=params.d_cond,
    )
    # according to each mode, choose proper "classes"(cls) for condition auto encoder
    autoreg_enc_cls = autoreg_enc_dict[mode] if use_autoreg_cond else None
    external_enc_cls = external_enc_dict[mode] if use_external_cond else None
    # according to each encoder model, initialize them, while their size are fixed
    autoreg_cond_enc = None if autoreg_enc_cls is None else autoreg_enc_cls()
    external_cond_enc = None if external_enc_cls is None else external_enc_cls.create_model()
    # auto encoder is None here, so not on latent space (diffusion)
    ldm_model = LatentDiffusion(
        unet_model=unet_model,
        autoencoder=None,
        autoreg_cond_enc=autoreg_cond_enc,
        external_cond_enc=external_cond_enc,
        latent_scaling_factor=params.latent_scaling_factor,
        n_steps=params.n_steps,
        linear_start=params.linear_start,
        linear_end=params.linear_end,
        debug_mode=debug_mode
    )

    return ldm_model


def init_diff_pro_sdf(ldm_model, params, device):
    # return the Diffpro_SDF model, which enables several 
    return Diffpro_SDF(ldm_model).to(device)


def get_model_path(model_dir, model_id=None):
    # e.g. model_dir = "./models/diffpro_sdf" → model_desc = "diffpro_sdf" (description)
    model_desc = os.path.basename(model_dir)
    if model_id is None:
        model_path = os.path.join(model_dir, 'chkpts', 'weights.pt')

        # retrieve real model_id from the actual file weights.pt is pointing to
        model_id = os.path.basename(os.path.realpath(model_path)).split('-')[1].split('.')[0]

    elif model_id == 'best':
        model_path = os.path.join(model_dir, 'chkpts', 'weights_best.pt')
        # retrieve real model_id from the actual file weights.pt is pointing to
        model_id = os.path.basename(os.path.realpath(model_path)).split('-')[1].split('.')[0]
    elif model_id == 'default':
        model_path = os.path.join(model_dir, 'chkpts', 'weights_default.pt')
        if not os.path.exists(model_path):
            return get_model_path(model_dir, 'best')
    else:
        model_path = f"{model_dir}/chkpts/weights-{model_id}.pt"
    return model_path, model_id, model_desc

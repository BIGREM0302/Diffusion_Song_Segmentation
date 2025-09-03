from argparse import ArgumentParser
from params import PARAMS_DICTS
import os
from train.train_config import LdmTrainConfig

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def init_parser():
    parser = ArgumentParser(description='train a diffusion model...')
    parser.add_argument(
        "--output_dir",
        default='results',
        help='directory in which to store model checkpoints and training logs'
    )
    parser.add_argument("--mode", help="which model to train (unsup, sup, mel)")
    parser.add_argument("--use1k7", action='store_true', help="whether to use pop1k7 dataset")
    parser.add_argument('--external', action='store_true', help="whether to use external control")
    parser.add_argument('--autoreg', action='store_true', help="whether to use autoregressive control")
    parser.add_argument('--multi_label', action='store_true', help="whether to use all human phrase labels")
    parser.add_argument('--uniform_pitch_shift', action='store_true',
                        help="whether to apply pitch shift uniformly (as opposed to randomly)")
    return parser

def args_check(args):
    assert args.mode in ['unsup', 'sup', 'mel']
    if args.mode == 'unsup':
        assert not args.multi_label

def args_setting_to_fn(args):
    def to_str(x:bool, char):
        return char if x else ''
    mode = args.mode
    autoreg = to_str(args.autoreg, 'a')
    external = to_str(args.external, 'e')
    multi_label = to_str(args.multi_label, 'l')
    p_shift = to_str(args.uniform_pitch_shift, 'p')
    use1k7 = to_str(args.use1k7, 'u')

    return f"{mode}-{autoreg}{external}-{multi_label}{p_shift}{use1k7}"

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    args_check(args)

    random_pitch_aug = not args.uniform_pitch_shift
    params = PARAMS_DICTS[args.mode]

    fn = args_setting_to_fn(args)

    output_dir = os.path.join(args.output_dir, fn)

    print(f"output directory:{output_dir}")
    print(f"params:{params}")

    config = LdmTrainConfig(params, output_dir, args.mode, args.use1k7, args.autoreg, args.external, args.multi_label, random_pitch_aug)

    config.train()
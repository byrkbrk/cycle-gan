from cycle_gan import CycleGAN
from argparse import ArgumentParser



if __name__ == "__main__":
    parser = ArgumentParser(description="Generate images from Cycle GAN")
    parser.add_argument("checkpoint_name", type=str, default=None, help="Checkpoint name to use for inference")
    parser.add_argument("--allow-checkpoint-download", type=bool, default=False, help="Downloads pretrained checkpoint if True")
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset name for inference, required when 'allow-checkpoint-download' is True")
    args = parser.parse_args()

    cycle_gan = CycleGAN(args.checkpoint_name, dataset_name=args.dataset_name,
                         allow_checkpoint_download=args.allow_checkpoint_download)
    cycle_gan.generate("AB")
    cycle_gan.generate("BA")
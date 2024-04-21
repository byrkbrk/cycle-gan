from cycle_gan import CycleGAN
from argparse import ArgumentParser



if __name__ == "__main__":
    parser = ArgumentParser(description="Generate images from Cycle GAN")
    parser.add_argument("checkpoint_name", type=str, default=None, help="Checkpoint name to use for inference")
    args = parser.parse_args()

    cycle_gan = CycleGAN(args.checkpoint_name)
    cycle_gan.generate("AB")
    cycle_gan.generate("BA")
from argparse import ArgumentParser
from cycle_gan import CycleGAN



if __name__ == "__main__":
    parser = ArgumentParser(description="Train Cycle GAN")
    parser.add_argument("--dataset-name", type=str, default="horse2zebra", help="Name of the dataset to train")
    parser.add_argument("--n-epochs", type=int, default=200, help="Number of epochs for training")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for training")
    parser.add_argument("--checkpoint-name", type=str, default=None, help="Checkpoint name for pre-training")
    parser.add_argument("--device", type=str, default=None, help="Device name for training")
    parser.add_argument("--id-criterion-name", type=str, default="L1", help="Criterion name for identity loss")
    parser.add_argument("--cycle-criterion-name", type=str, default="L1", help="Criterion name for cycle consistency loss")
    parser.add_argument("--adv-criterion-name", type=str, default="mse", help="Criterion name for adversarial loss")
    parser.add_argument("--lambda-id", type=float, default=0.1, help="Hyperparameter for identity loss")
    parser.add_argument("--lambda-cycle", type=float, default=10, help="Hyperparameter for cycle consistency loss")
    parser.add_argument("--checkpoint-save-freq", type=int, default=5, help="Frequency for checkpoint saving")
    parser.add_argument("--checkpoint-save-dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--image-save-dir", type=str, default=None, help="Directory to save images during training")
    parser.add_argument("--allow-checkpoint-download", type=bool, default=False, help="Downloads pretrained checkpoint if True")
    parser.add_argument("--buffer-capacity", type=int, default=50, help="Capacity of the image buffer")
    parser.add_argument("--lr-decay-start-epoch", type=int, default=100, help="The epoch at which lr starts to decay")
    args = parser.parse_args()

    cycle_gan = CycleGAN(args.checkpoint_name, args.dataset_name, args.device, args.allow_checkpoint_download)
    cycle_gan.train(args.n_epochs, args.batch_size, args.lr, args.id_criterion_name, args.cycle_criterion_name, args.adv_criterion_name, 
                    args.lambda_id, args.lambda_cycle, args.checkpoint_save_dir, args.checkpoint_save_freq, args.image_save_dir, 
                    args.buffer_capacity, args.lr_decay_start_epoch)


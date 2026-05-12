"""
入口：
  python run.py --mode train
  python run.py --mode sample   [--ckpt ckpt.pt] [--n 64] [--steps 50] [--ddim]
  python run.py --mode evaluate [--ckpt ckpt.pt] [--n_samples 1000] [--n_display 64]
"""
import argparse

parser = argparse.ArgumentParser(description="DDPM Baseline — MNIST")
parser.add_argument("--mode",       default="train",
                    choices=["train", "sample", "evaluate"])
parser.add_argument("--ckpt",       default=None,  help="Checkpoint path")
parser.add_argument("--n",          type=int, default=64,   help="Samples for sample mode")
parser.add_argument("--steps",      type=int, default=None, help="Sampling steps")
parser.add_argument("--ddim",       action="store_true",    help="Use DDIM sampler")
parser.add_argument("--n_samples",  type=int, default=1000, help="Samples for FID (evaluate mode)")
parser.add_argument("--n_display",  type=int, default=64,   help="Samples shown in grid (evaluate mode)")
parser.add_argument("--wandb_mode", default=None,
                    choices=["online", "offline", "disabled"])
args = parser.parse_args()

if args.wandb_mode:
    from ddpm import config
    config.Config.WANDB_MODE = args.wandb_mode

if args.mode == "train":
    from ddpm.train import main
    main()

elif args.mode == "sample":
    from ddpm.sample import generate
    generate(ckpt_path=args.ckpt, n=args.n, steps=args.steps, use_ddim=args.ddim)

elif args.mode == "evaluate":
    import glob
    from ddpm import config
    ckpt = args.ckpt
    if ckpt is None:
        ckpts = sorted(glob.glob(str(config.CHECKPOINT_DIR / "ckpt_*.pt")))
        if not ckpts:
            print("ERROR: no checkpoint found. Train first.")
            exit(1)
        ckpt = ckpts[-1]
        print(f"Auto-selected: {ckpt}")

    from ddpm.evaluate import evaluate
    results = evaluate(ckpt, n_samples=args.n_samples, n_display=args.n_display)
    print(f"\nResults: {results}")

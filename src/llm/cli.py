import argparse
import logging
import os
import sys

from .config import ModelConfig
from .data import BPETokenizer, Dataset, Tokenizer
from .tensor import USING_GPU, set_seed
from .train import train_loop
from .transformer import Transformer

logging.basicConfig(level=logging.INFO, format="%(message)s")


def pre_parse_env():
    # pre-parse to set env before tensor.py import
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64", "float16"],
        default="float32",
    )
    args, _ = parser.parse_known_args()
    if args.device != "auto":
        os.environ["LLM_DEVICE"] = args.device
    os.environ["LLM_DTYPE"] = args.dtype


pre_parse_env()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("boolean expected.")


def _shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", type=str, choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64", "float16"],
        default="float32",
    )
    p.add_argument("--seed", type=int, default=None)


def _load_tokenizer(kind: str, dataset: str, text: str, vocab_size: int):
    data_dir = os.path.join(os.getcwd(), "data")
    tokenizers_dir = os.path.join(data_dir, "tokenizers")
    os.makedirs(tokenizers_dir, exist_ok=True)
    if kind == "char":
        return Tokenizer(text)
    tok = BPETokenizer()
    bpe_path = os.path.join(tokenizers_dir, f"{dataset}_{kind}_{vocab_size}.json")
    if os.path.exists(bpe_path):
        tok.load(bpe_path)
    else:
        tok.train(text, vocab_size=vocab_size, verbose=True)
        tok.save(bpe_path)
    return tok


def generate_text(
    model: Transformer,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k=None,
    top_p=None,
    repetition_penalty: float = 1.0,
    stream: bool = False,
) -> str:
    """Encode prompt, run model.generate, decode. Optionally stream to stdout."""
    idx = tokenizer.encode(prompt)
    if stream:
        print(prompt, end="", flush=True)
    out_tokens = []
    for tok in model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    ):
        out_tokens.append(tok)
        if stream:
            print(tokenizer.decode([tok]), end="", flush=True)
    if stream:
        print()
    return prompt + tokenizer.decode(out_tokens)


def cmd_train(args):
    if args.seed is not None:
        set_seed(args.seed)

    data_dir = os.path.join(os.getcwd(), "data")
    checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    data_file = os.path.join(data_dir, f"{args.dataset}.txt")
    if not os.path.exists(data_file):
        print(f"{data_file} not found")
        sys.exit(1)
    with open(data_file) as f:
        text = f.read()

    name = args.name or args.dataset

    import numpy as snp

    if args.resume and args.from_checkpoint:
        print("error: --resume and --from_checkpoint are mutually exclusive")
        print("  --resume PATH: continue an interrupted run (restores optimizer state)")
        print("  --from_checkpoint PATH: fine-tune a trained model (fresh optimizer)")
        sys.exit(1)

    # --resume / --from_checkpoint both load model + tokenizer from a checkpoint.
    # The difference: --resume also restores optimizer state (for continuing
    # an interrupted run), while --from_checkpoint starts a fresh optimizer
    # (for fine-tuning on new data).
    ckpt_path = args.resume or args.from_checkpoint
    if ckpt_path:
        if not os.path.exists(ckpt_path):
            print(f"{ckpt_path} not found")
            sys.exit(1)
        model, tokenizer = Transformer.from_checkpoint(ckpt_path)
        if tokenizer is None:
            print("error: checkpoint has no embedded tokenizer")
            sys.exit(1)
        cfg = model.config
        print(f"loaded model from {ckpt_path}")
        if args.resume:
            # Resume writes back to the same checkpoint it loaded from.
            tag = None
        else:
            tag = f"{name}_finetune_L{cfg.n_layer}_E{cfg.n_embd}_H{cfg.n_head}"
    else:
        tokenizer = _load_tokenizer(args.tokenizer, args.dataset, text, args.vocab_size)
        cfg = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            block_size=args.block_size,
            dropout=args.dropout,
            tie_weights=args.tie_weights,
            act=args.act,
            norm=args.norm,
        )
        model = Transformer(cfg)
        tag = f"{name}_{args.tokenizer}_L{args.n_layer}_E{args.n_embd}_H{args.n_head}"

    if tag is not None:
        checkpoint = os.path.join(checkpoints_dir, f"{tag}.npz")
    else:
        checkpoint = args.resume  # resume writes back to the same file
    total_params = sum(p.data.size for p in model.parameters())
    print(f"model parameters: {total_params:,}")

    tok_type = tokenizer.to_dict()["type"]
    cache = os.path.join(data_dir, f"{args.dataset}_encoded_{tok_type}_{tokenizer.vocab_size}.npy")
    if os.path.exists(cache):
        encoded_text = snp.load(cache).tolist()
    else:
        encoded_text = tokenizer.encode(text, verbose=True)
        snp.save(cache, snp.array(encoded_text, dtype=snp.uint32))

    dataset = Dataset(encoded_text, cfg.block_size)
    train_ds, val_ds = dataset.split(args.val_frac) if args.val_frac > 0 else (dataset, None)

    # --resume restores optimizer state so training picks up where it left off.
    resume_optimizer = None
    if args.resume:
        opt_path = args.resume + ".opt.npz"
        if os.path.exists(opt_path):
            resume_optimizer = opt_path
        else:
            print(f"warning: no optimizer state found at {opt_path}, starting fresh optimizer")

    history = train_loop(
        model,
        train_ds,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        grad_accum_steps=args.grad_accum_steps,
        val_dataset=val_ds,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        checkpoint_path=checkpoint,
        checkpoint_interval=args.checkpoint_interval,
        resume_optimizer=resume_optimizer,
        tokenizer=tokenizer,
    )

    print(f"saving to {checkpoint}...")
    model.save(checkpoint, tokenizer=tokenizer)
    history["optimizer"].save(checkpoint + ".opt.npz")

    import json
    import time

    params = vars(args).copy()
    params.pop("func", None)
    results = {
        "params": params,
        "loss": history["train"],
        "val": history["val"],
    }
    results["params"]["timestamp"] = time.time()
    results["params"]["total_parameters"] = total_params
    # Derive a results tag from the checkpoint basename (works for all paths).
    res_tag = os.path.splitext(os.path.basename(checkpoint))[0]
    res_file = os.path.join(
        results_dir,
        f"{res_tag}_{cfg.act}_{cfg.norm}_S{args.steps}_{int(time.time())}.json",
    )
    with open(res_file, "w") as f:
        json.dump(results, f)


def cmd_generate(args):
    if args.seed is not None:
        set_seed(args.seed)

    model, tokenizer = Transformer.from_checkpoint(args.checkpoint)
    if tokenizer is None:
        print("error: checkpoint has no embedded tokenizer")
        sys.exit(1)
    generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stream=True,
    )


def main():
    print(f"--- using {'gpu (cupy)' if USING_GPU else 'cpu (numpy)'} ---")

    parser = argparse.ArgumentParser(description="llm-from-scratch CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="train a transformer")
    _shared_args(p_train)
    p_train.add_argument(
        "--dataset", type=str, default="shakespeare", help="name of the .txt file in data/"
    )
    p_train.add_argument(
        "--name", type=str, default=None, help="checkpoint name (defaults to --dataset)"
    )
    p_train.add_argument("--tokenizer", type=str, choices=["char", "bpe"], default="bpe")
    p_train.add_argument("--vocab_size", type=int, default=3000)
    p_train.add_argument("--n_layer", type=int, default=4)
    p_train.add_argument("--n_embd", type=int, default=128)
    p_train.add_argument("--n_head", type=int, default=4)
    p_train.add_argument("--block_size", type=int, default=128)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--batch_size", type=int, default=16)
    p_train.add_argument("--steps", type=int, default=100)
    p_train.add_argument("--weight_decay", type=float, default=0.01)
    p_train.add_argument("--grad_clip", type=float, default=1.0)
    p_train.add_argument("--warmup_steps", type=int, default=100)
    p_train.add_argument("--grad_accum_steps", type=int, default=1)
    p_train.add_argument("--val_frac", type=float, default=0.1)
    p_train.add_argument("--eval_interval", type=int, default=0)
    p_train.add_argument("--eval_iters", type=int, default=20)
    p_train.add_argument("--checkpoint_interval", type=int, default=0)
    p_train.add_argument(
        "--resume",
        type=str,
        default=None,
        help="resume an interrupted run from a checkpoint (restores optimizer state)",
    )
    p_train.add_argument(
        "--from_checkpoint",
        type=str,
        default=None,
        help="fine-tune from an existing checkpoint (fresh optimizer)",
    )
    p_train.add_argument("--tie_weights", type=str2bool, default=True)
    p_train.add_argument("--act", type=str, choices=["swiglu", "gelu", "relu"], default="swiglu")
    p_train.add_argument("--norm", type=str, choices=["rmsnorm", "layernorm"], default="rmsnorm")
    p_train.set_defaults(func=cmd_train)

    # generate
    p_gen = sub.add_parser("generate", help="generate text from a checkpoint")
    _shared_args(p_gen)
    p_gen.add_argument("--checkpoint", type=str, required=True)
    p_gen.add_argument("--prompt", type=str, default="The ")
    p_gen.add_argument("--max_new_tokens", type=int, default=100)
    p_gen.add_argument("--temperature", type=float, default=0.8)
    p_gen.add_argument("--top_k", type=int, default=None)
    p_gen.add_argument("--top_p", type=float, default=None)
    p_gen.add_argument("--repetition_penalty", type=float, default=1.1)
    p_gen.set_defaults(func=cmd_generate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

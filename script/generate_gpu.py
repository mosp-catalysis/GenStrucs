import os
import argparse
import pickle
from tqdm import tqdm

import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="Output pkl filename (without extension)")
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--save-path", required=True)
    p.add_argument("--n-generation", type=int, default=500000)

    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=1.0)

    # Performance-related
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda")

    # Prompt, optional
    p.add_argument("--input-prompt", type=str, default="")

    # Storage dtype for token ids
    p.add_argument("--dtype", type=str, default="int32", choices=["int64", "int32", "int16"],help="Saved token dtype.")
  
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model = GPT2LMHeadModel.from_pretrained(args.ckpt_path).to(device)
    model.eval()

    prompt_ids = tokenizer.encode(args.input_prompt, add_special_tokens=True)
    if len(prompt_ids) == 0:
        if tokenizer.bos_token_id is not None:
            prompt_ids = [tokenizer.bos_token_id]
        elif tokenizer.eos_token_id is not None:
            prompt_ids = [tokenizer.eos_token_id]
        else:
            prompt_ids = [tokenizer.pad_token_id]

    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, L_prompt]
    eos_id = tokenizer.eos_token_id

    # Select dtype
    save_dtype = {"int64": torch.int64, "int32": torch.int32, "int16": torch.int16}[args.dtype]

    generated = []
    total = args.n_generation
    bs = args.batch_size

    total_len = 0
    hit_eos = 0
    saved_n = 0
    max_len_seen = 0

    with tqdm(total=total) as pbar:
        n_done = 0
        while n_done < total:
            cur_bs = min(bs, total - n_done)

            input_ids = prompt.repeat(cur_bs, 1)  # [B, L_prompt]

            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model.generate(
                        input_ids=input_ids,
                        max_length=args.max_length,
                        do_sample=True,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )

            # Move entire batch to CPU once
            out_cpu = out.detach().to("cpu")

            for j in range(out_cpu.size(0)):
                seq1d = out_cpu[j]

                found_eos = False
                if eos_id is not None:
                    eos_pos = (seq1d == eos_id).nonzero(as_tuple=False)
                    if eos_pos.numel() > 0:
                        k = int(eos_pos[0].item())
                        seq1d = seq1d[: k + 1]
                        found_eos = True

                seq1d = seq1d.contiguous().clone()
                seq1d = seq1d.to(save_dtype)
                generated.append(seq1d.unsqueeze(0))

                L = int(seq1d.numel())
                total_len += L
                saved_n += 1
                if found_eos:
                    hit_eos += 1
                if L > max_len_seen:
                    max_len_seen = L

            n_done += cur_bs
            pbar.update(cur_bs)

    os.makedirs(args.save_path, exist_ok=True)
    out_pkl = os.path.join(args.save_path, args.name + ".pkl")

    with open(out_pkl, "wb") as fw:
        pickle.dump(generated, fw, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[done] wrote: {out_pkl}")
    if saved_n > 0:
        print(
            f"[stats] n={saved_n} avg_len={total_len/saved_n:.1f} "
            f"hit_eos={hit_eos/saved_n:.3f} max_len={max_len_seen}"
        )
        print(f"[stats] saved dtype={args.dtype} (int32 recommended; int16 if safe)")

    try:
        size_mb = os.path.getsize(out_pkl) / 1024**2
        print(f"[file] size={size_mb:.2f} MB")
    except OSError:
        pass


if __name__ == "__main__":
    main()

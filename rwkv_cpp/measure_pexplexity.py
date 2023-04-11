# Measures perplexity and per-token latency of an RWKV model on a given text file.
# Perplexity is defined here as exp() of average cross-entropy loss.
# Usage: python measure_pexplexity.py C:\rwkv.cpp-169M.bin C:\text.txt 1024

import os
import time
import pathlib
import argparse
import tokenizers
import torch
import rwkv_cpp_model
import rwkv_cpp_shared_library
from typing import List

def parse_args():
    parser = argparse.ArgumentParser(description='Measure perplexity and per-token latency of an RWKV model on a given text file')
    parser.add_argument('model_path', help='Path to model checkpoint file')
    parser.add_argument('text_path', help='Path to text file in UTF-8 encoding')
    parser.add_argument('ignore_first_n_tokens', help='How many tokens should be skipped before loss is measured', type=int, default=1024)
    return parser.parse_args()

args = parse_args()

# ---

print('Loading 20B tokenizer')
tokenizer_path: pathlib.Path = pathlib.Path(os.path.abspath(__file__)).parent / '20B_tokenizer.json'
tokenizer: tokenizers.Tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))

print('Loading text')
text: str = open(args.text_path, encoding='utf-8').read()
tokens: List[int] = tokenizer.encode(text).ids
token_count: int = len(tokens)
print(f'{token_count} tokens in the text')

assert token_count - args.ignore_first_n_tokens > 1, 'Need at least 2 tokens for evaluation'

# ---

def format_loss(loss: torch.Tensor) -> str:
    return str(['%.3f' % (loss[i].item(),) for i in range(len(loss))]).replace('\'', '')[1:-1]

def format_loss_with_perplexity(loss: torch.Tensor) -> str:
    return f'loss [{format_loss(loss)}], perplexity {"%.3f" % (torch.exp(loss[0]).item(),)}'

# ---

model: rwkv_cpp_model.RWKVModel = rwkv_cpp_model.RWKVModel(
    rwkv_cpp_shared_library.load_rwkv_shared_library(),
    args.model_path
)

logits, state = None, None

loss_sum: torch.Tensor = torch.tensor([0.0])
loss_count: int = 0

start: float = time.time()

run_count: int = token_count - 1

for i in range(run_count):
    token: int = tokens[i]
    target: int = tokens[i + 1]

    logits, state = model.eval(token, state, state, logits)

    if args.ignore_first_n_tokens == 0 or i + 1 >= args.ignore_first_n_tokens:
        losses = torch.tensor([
            torch.nn.functional.cross_entropy(logits, torch.tensor(target, dtype=torch.long), reduction='none').item()
        ])

        loss_sum += losses
        loss_count += 1

    if i % 10 == 0:
        avg_loss_so_far = loss_sum / loss_count

        duration: float = time.time() - start
        duration_per_token: float = duration / (i + 1)
        runs_remaining: int = run_count - i - 1
        duration_remaining: int = int(runs_remaining * duration_per_token)

        print(f'Token #{i}/{token_count}, '
              f'{int(100.0 * i / token_count)}%, '
              f'ETA {duration_remaining // 60} m {duration_remaining % 60} s', end='')

        if loss_count > 0:
            print(f', averages so far: {format_loss_with_perplexity(avg_loss_so_far)}')
        else:
            print()

print()
print(f'Average latency: {int((time.time() - start) * 1000 / run_count)} ms per token')

print()
print(f'Model: {os.path.basename(args.model_path)}, '
      f'data: {os.path.basename(args.text_path)} with {token_count} tokens, '
      f'skipped {args.ignore_first_n_tokens} tokens, '
      f'averages: {format_loss_with_perplexity(loss_sum / loss_count)}')

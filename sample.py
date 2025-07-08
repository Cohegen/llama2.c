"""
Sample from the trained model with PyTorch
"""
import os
import pickle
from contextlib import nullcontext
import torch
from model import ModelArgs, Transformer
from tokenizer import Tokenizer

from tinystories import get_tokenizer_model_path
import argparse
import logging

# -----------------------------------------------------------------------------
checkpoint = 'out/ckpt.pt'
start = "" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 100 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 300 # retain only the top_k most likely tokens, clamp others to have 0 probability
tokenizer = "" # override the tokenizer model path
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
dtype = "float32"
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained model with PyTorch.")
    parser.add_argument('--checkpoint', type=str, default='out/ckpt.pt', help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='', help='Prompt string or FILE:filename')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to draw')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=300, help='Top-k sampling')
    parser.add_argument('--tokenizer', type=str, default='', help='Path to tokenizer model')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu/cuda)')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32','bfloat16','float16'], help='Data type')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile for speed')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], help='Logging level')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    checkpoint = args.checkpoint
    start = args.prompt
    num_samples = args.num_samples
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_k = args.top_k
    tokenizer = args.tokenizer
    seed = args.seed
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = args.dtype
    compile = args.compile
    exec(open('configurator.py').read()) # still allow config override
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format='[%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)
else:
    # fallback for legacy usage
    pass

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
checkpoint_dict = torch.load(checkpoint, map_location=device)
gptconf = ModelArgs(**checkpoint_dict['model_args'])
model = Transformer(gptconf)
state_dict = checkpoint_dict['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(device)
if compile:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# load the tokenizer
vocab_source = checkpoint_dict["config"].get("vocab_source", "llama2")
vocab_size = gptconf.vocab_size
if tokenizer:
    # a specific tokenizer is provided, use it
    tokenizer_model = tokenizer
else:
    # let's try to find the tokenizer model automatically. bit gross here...
    query_vocab_size = 0 if vocab_source == "llama2" else vocab_size
    tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
enc = Tokenizer(tokenizer_model=tokenizer_model)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = enc.encode(start, bos=True, eos=False)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            logger.info(enc.decode(y[0].tolist()))
            logger.info('---------------')

# To run this code for 8 GPUs, use the following command
# torchrun --standalone --nproc_per_node=8 gpt2.py

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import time
import math
import inspect
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import os
from transformers import AutoTokenizer
import wandb
import numpy as np
from hellaswag import render_example, iterate_examples
import tiktoken
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import os
import re

path = os.path.dirname(os.path.abspath(__file__))

if path != '/fs/nexus-scratch/thilakcm/848k-project':
    pattern = r'c848k\d+'
    account = re.findall(pattern, path)[0]
    save_folder = f'/fs/class-projects/fall2024/cmsc848k/{account}/FIRE'
    os.makedirs(save_folder, exist_ok=True)
else:
    save_folder = '/fs/nexus-scratch/thilakcm/FIRE'
    os.makedirs(save_folder, exist_ok=True)
#%%
# This is for distributed data parallelism
ddp = int(os.environ.get('RANK', -1)) != -1

# If ddp is true, then we need to initialize the process group
if ddp:  # For legends
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])  # GPU 0 has rank 0, GPU 1 has rank 1, etc.
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # Local rank within the node
    ddp_world_size = int(os.environ['WORLD_SIZE']) # Number of GPUs

    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)   
    master_process = ddp_rank == 0  
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = 'cpu' # For noobs
    if torch.cuda.is_available(): # For adults
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # For kids
        device = "mps"
    print(f"using device: {device}" )

# pytorch can be serious about device vs device type so we need to set it correctly
# TODO: understand the difference between device and device type
device_type = "cuda" if device.startswith("cuda") else "cpu"

# This is for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

if master_process:
    # Initialize wandb to this project
    wandb.init(project="GPT 2 848K Nexus Cluster")

    wandb.run.tags = ["GPT2", "124M params", "10B tokens", "Flash Attention","Fire PE", "Testing", "Gelu"]

# GPT-2 is a decoder only transformer model
#This is for MLP block
class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        # TODO: Instead of using tanh, we can use non approximate version also
        #There is not much difference between the time for tanh and real GELU
        self.gelu = nn.GELU(approximate='tanh') # gpt2 uses tanh approximation so we're using it
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed) 
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# This is for self attention block
class CausalSelfAttention(nn.Module):
    def __init__(self, config):

        super().__init__() 
        assert config.n_embed % config.n_head == 0
        
        # This is for query, key and value projections
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)  # combined linear projection for all three Q, K, V
        # This is for output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        # This is for scaling the weights
        self.c_proj.NANOGPT_SCALE_INIT = 1.0


        # Regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed


        self.fire_causal_mask= FIRE(num_heads=config.n_head)
        # not really a bias but a mask, but following OpenAI naming convention
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) 

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence Length, Embedding dimensionality (n_embed)
        # d_k = d_v = n_embed // n_head
        # n_head -> Number of heads in the multi-head attention
        # create query, key, value matrices
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)  # B x T x n_embed -> B x T x 3*n_embed -> q, k, v each of shape B x T x n_embed
        # hs (head size) = n_embed // n_head
        # C == n_emb == n_head * hs == n_head * d_k == n_head * d_v
        q = q.view(B, T, self.n_head, self.n_embed // self.n_head).transpose(1, 2)  # q (BxTxC) reshaped to B x T x n_head x hs then transposed to B x n_head x T x hs
        k = k.view(B, T, self.n_head, self.n_embed // self.n_head).transpose(1, 2)  # same for k
        v = v.view(B, T, self.n_head, self.n_embed // self.n_head).transpose(1, 2)  # same for v


        # Attention mechanism
        # att = (q @ k.transpose(-2, -1)) * (1.0 / ((k.size(-1)) ** 0.5))
        # # Masked Attention
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v

        # Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False, attn_mask= self.fire_causal_mask(q)) # wow who knew flash attention was so easy to implement
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * (self.n_embed // self.n_head))
        # Output projection
        y = self.c_proj(y)
        return y

class FIRE(nn.Module):
    def __init__(self, num_heads=12, mlp_width=32, init_c=0.1, init_L=512., eps=1e-6):

        #  FIRE attention bias module.
        #  Args:
        #  num_heads: number of attention heads.
        #  mlp_width: Width of MLP.
        #  init_c: initial value of log transformation parameter
        #  init_L: initial value of thresholding parameter
        #  eps: small constant for numerical stability

        super(FIRE, self).__init__()

        # Define the MLP layers
        self.mlp = nn.Sequential(
        nn.Linear(1, mlp_width),
        nn.ReLU(),
        nn.Linear(mlp_width, num_heads) )

        # Initialize c (log transformation parameter)
        self.c = nn.Parameter(torch.tensor(init_c))
        # Initialize L (threshold)
        self.init_L = nn.Parameter(torch.tensor(init_L),requires_grad=False)
        # Learn a multiplier to L
        self.L_multiplier = nn.Parameter(torch.tensor(1.0))

        self.eps = eps

    def forward(self, x: torch.Tensor):
 
        #Compute FIRE attention bias.
        #Args:x: input sequence,shape [bsz, num_heads, seq_len, hidden_dim]Returns:attention bias,
        #shape [1, num_heads, seq_len, seq_len]

        seq_length = x.size(2)
        positions = torch.arange(seq_length,dtype=torch.float,device=x.device)
        rel_distance = positions[:, None] - positions[None, :]

        # Thresholding the normalizer
        threshold = torch.abs(self.L_multiplier * self.init_L)
        pos_normalizer = torch.max(positions, threshold)
        pos_normalizer = pos_normalizer[:, None]

        # Amplifying differences among local positions with log transform
        rel_distance = torch.log(torch.abs(self.c * rel_distance) + 1)
        pos_normalizer = torch.log(torch.abs(self.c * pos_normalizer) + 1) + self.eps

        # Progressive interpolation
        normalized_distance = rel_distance / pos_normalizer
        fire_bias = self.mlp(normalized_distance.unsqueeze(-1))
        # The commented and the uncommented code are the same but uncommented code is faster
        # fire_bias = fire_bias.unsqueeze(0).permute(0, 3, 1, 2)
        # fire_bias = fire_bias.permute(2, 1, 0).unsqueeze(0)
        fire_bias = fire_bias.permute(2, 0, 1)
        mask = torch.ones(seq_length, seq_length).tril(diagonal=0).repeat(fire_bias.shape[0], 1, 1)
        fire_bias = fire_bias.masked_fill(mask.logical_not().to(device), float('-inf')).unsqueeze(0)
        return fire_bias
    
# This is for transformer block
class Block(nn.Module):
    # write 

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # didn't realize it was this easy to implement residual connections
        x = x + self.attn(self.ln_1(x)) # clean residual connections are desrable for deep models form an optimization perspective
        x = x + self.mlp(self.ln_2(x)) # also we perform layer normalization before self attention and MLP, in contrast to the original transformer
        # this is because it is more stable to normalize the input to each sub-layer, rather than the output
        # this is called pre-normalization and is used in the "An Image is Worth 16x16 Words" paper
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # Maximum sequence length
    vocab_size: int = 50257 # 50k "Byte Pair Encodings" (BPE) vocab size + 256 bytes tokens + 1 <|endoftoken|>
    # special end of sequence token delimits document boundaries and can start generation as well
    n_layer: int = 12 # Number of transformer blocks (how deep is the model)
    n_head: int = 12 # Number of heads in the multi-head attention (how wide is the model)
    n_embed: int = 768 # Embedding dimensionality

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


        # Developing Transformer
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embed), # Token embedding weights
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # All transformer blocks
            'ln_f': nn.LayerNorm(config.n_embed)
        })

        # Final Linear layer after all transformer blocks
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
    
        # Weight sharing scheme
        # This is for sharing the weights between token and positional embeddings
        # Reason: Since they are semantically similar, they should have similar weights
        self.lm_head.weight = self.transformer['wte'].weight

        # Initialize parameters with mean 0 and standard deviation 0.02 because 1/sqrt(768), 1/sqrt(1600)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # Here 2 is because one is for MLP and other is for attention mechanism
                # config.n_layer is number of transformer blocks
                # Why this? 
                # We send each layer after adding the residual connection and normalization
                # To make results' distribution normal with standard deviation = 0.02
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            # if module.padding_idx is not None:
            #     torch.nn.init.zeros_(module.weight[module.padding_idx])

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, model block size is {self.config.block_size}"

        # IMP: Token and Positional Embeddings
        tok_emb = self.transformer.wte(idx)  #Token Embeddings of shape (B, T, n_embed)
        x = tok_emb # broadcast along the batch dimension

        # Forward pass through each transformer block
        for block in self.transformer.h:
            x = block(x)
        
        # Final Linear layer
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)

        # Loss function
        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
        return x, loss


    @classmethod
    def from_pretrained(cls, model_type):
        ## This is for loading the pretrained model
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import AutoModelForCausalLM
        print("Loading weights from pretrained gpt:", model_type)

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embed=768),           #124M parameters
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embed=1024),   #345M parameters
            'gpt2-large': dict(n_layer=36, n_head=20, n_embed=1280),    #774M parameters
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embed=1600)        #1558M parameters
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('attn.bias')]

        # init a huggingface GPT2 model
        model_hf = AutoModelForCausalLM.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Check parameters match
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('attn.bias')]
        assert set(sd_keys) == set(sd_keys_hf), f"Length mismatch - keys: {len(sd_keys)} and huggingface keys: {len(sd_keys_hf)}"

        # Copy shared weights
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in sd_keys:
            if any(k.endswith(x) for x in transposed):
                assert sd[k].shape == sd_hf[k].shape[::-1]
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())  #This is transpose but will not give warnings
            else:
                assert sd[k].shape == sd_hf[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, lr, device):
        # We need all the named parameters that require gradients
        param_dict = {k: v for k, v in self.named_parameters()}
        param_dict = {k: v for k, v in param_dict.items() if v.requires_grad}
        
        # Bias does not need weight decay 
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        # Optimizer
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]

        # Return number of elements (numel), which is the number of parameters
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)

        # Check AdamW optimizer and use the fused verison if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay, fused=use_fused)
        if master_process:
            print(100*'-')
            print(f"Using Fused AdamW: {fused_available}")
            print(f"# Decayed parameter tensors: {len(decay_params)} with {num_decay_params} parameters")
            print(f"# No Decay parameter tensors: {len(no_decay_params)} with {num_no_decay_params} parameters")
        return optimizer
#%%
def load_tokens(filename):
    try: npt = np.load(filename, allow_pickle=True)
    except: npt = np.fromfile(filename, dtype=np.uint16)  # Replace dtype as needed

    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt 

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, device ='cpu',):
        self.B, self.T = B, T
        self.device = device
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        master_process = process_rank ==0
        #get the shared filenames
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.join(CURRENT_DIR, "edu_fineweb10B")
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        
        #state, init and shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        # UserWarning: To copy construct from a tensor, 
        # it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), 
        # rather than torch.tensor(sourceTensor)
        # buf = torch.tensor(self.tokens[self.current_position:self.current_position + B*T + 1])
        buf = self.tokens[self.current_position:self.current_position + B*T + 1].clone().detach()#.requires_grad_(True)
        x = buf[:-1].view(B, T).to(self.device) #inputs
        y = buf[1:].view(B, T).to(self.device)  #targets
        
        # We need to advance position B*T*num_processes to get the next batch in tensor
        self.current_position += B*T*self.num_processes

        # If loading the next shard would be out of bounds, advance to the next shard
        if self.current_position +(B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x,y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

enc = tiktoken.get_encoding('gpt2')

# Good, bad and ugly numbers - Why 50304? 50304 % 128 = 0 and is even 
model = GPT(GPTConfig(vocab_size=50304)).to(device)
model.to(device)

# count number of parameters
num_params = sum(p.numel() for p in model.parameters())
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if master_process:
    print(100 * '-')
    print(f"Total number of parameters: {num_params}, Trainable parameters: {num_trainable_params}")

total_tokens = 1e10 # 10B tokens

# log parameters to wandb
if master_process: 
    wandb.watch(model, log="all")
# Python interpreter is very slow. So, we need to compile the model
# If compiled, in GPU, instead of traversing from HBM to cache for each single operation, 
# computation is done by traversing once  
# This is for linux only
model = torch.compile(model)
# This is for ddp
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank)
raw_model = model.module if ddp else model # Always contains the "raw" unwrapped model

if master_process:
    for name, param in raw_model.named_parameters():
        print(f"Layer: {name} | Number of parameters: {param.numel()}")


max_steps = 19073 #19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

# Now GPT-3 parameters are used for GPT-2
weight_decay = 0.1

# This is for gradient accumulation
#total_batch_size = 2**19 # 500K tokens
B, T = 8, 1024


if master_process: # To print jsut one single time
    wandb.config.update({
    # Training parameters
    "batch_size": B,
    "sequence_length": T,
    #"total_batch_size": total_batch_size,
    "device": device,

    # Model parameters
    "embedding_size": raw_model.config.n_embed,
    "num_layers": raw_model.config.n_layer,
    "num_heads": raw_model.config.n_head,
    "vocab_size": raw_model.config.vocab_size,
    "dropout": 0,

    # Parameter counts
    "total_params": num_params,
    "trainable_params": num_trainable_params,
    })
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", device=device)

torch.cuda.empty_cache()
# This is for TF32 - 19 bits: 1 sign, 8 range and 10 mantissa
torch.set_float32_matmul_precision('high')
best_train_loss_accum = 1e9
avg_time = 0
avg_tokens_per_sec = 0


max_steps = 19073
# for epoch in range(0, 21000, 1000):

for epoch in range(max_steps):
    last_step = (epoch == max_steps - 1)
    
    if not ((epoch > 0 and epoch % 1000 == 0) or last_step): 
        if master_process: wandb.log({'nonsense': 1})
        continue
    
    if epoch != last_step:
        raw_model.load_state_dict(torch.load(f"{save_folder}/model_{epoch}.pth", weights_only=True))
    else: 
        raw_model.load_state_dict(torch.load(f"{save_folder}/final_epoch_model.pth", weights_only=True))

    model.eval()

    ########## once in a while evaluate our validation loss
    if master_process:  print("evaluating validation loss:")
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
        print(f"validation loss: {val_loss_accum.item():.4f}")
        wandb.log({"val_loss": val_loss_accum.item()})
            # you might also want to add optimizer.state_dict() and
            # rng seeds etc., if you wanted to more exactly resume training


    #################### once in a while evaluate hellaswag
    if master_process: print('evaluating hellaswag benchmark performance')
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        # only process examples where epoch % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)

    # reduce the stats across all processes
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if master_process:
        # log the accuracy
        hellaswag_accuracy = acc_norm
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={hellaswag_accuracy:.4f}")
        wandb.log({"hellaswag_accuracy": hellaswag_accuracy})


    ############## once in a while generate from the model (except epoch 0, which is noise)
    num_return_sequences = 4
    max_length = 32
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(xgen) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"rank {ddp_rank} sample {i}: {decoded}")

    if master_process: wandb.log({'nonsense': 1})

# Destroy all processes if ddp is true
if ddp: destroy_process_group()

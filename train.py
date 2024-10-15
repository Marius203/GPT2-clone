
from classes import *

num_return_sequences = 5
max_length = 30
device = 'cuda'
min_train_loss = 10

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Prefix tokens
import tiktoken

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoader:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")

        if split == 'train':
            self.current_shard = 1
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.currnet_position = self.B * self.T
        else:
            self.current_shard = 0
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.currnet_position = self.B * self.T

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.currnet_position : self.currnet_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance position
        self.currnet_position += B*T

        #reset if next batch is out of bounds 
        if self.currnet_position + (B * T + 1) > len(self.tokens):
            print("went next shard")
            self.current_shard = self.current_shard+1 % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.currnet_position = B*T
        return x, y

import os

# Function to save the checkpoint
def save_checkpoint(model, optimizer, step, path="gpt_checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at step {step}, min train loss {min_train_loss}")

# Function to load the checkpoint
def load_checkpoint(model, optimizer, path="gpt_checkpoint.pth"):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        print(f"Checkpoint loaded from step {step}")
        return step
    else:
        print("No checkpoint found, starting from scratch.")
        return 0


torch.manual_seed(7)
torch.cuda.manual_seed(7)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 131072 # 2^17 -> nice number 
B = 8 #small batch size
T = 256 #number of tokens in batch
assert total_batch_size % (B*T) == 0, "make sure total batch size is divisible by b*t"
grad_accum_steps = total_batch_size //(B*T)
print(f"total desired batch size: {total_batch_size}")
print(f"-----> calculated gradient accumulation steps: {grad_accum_steps}") 


train_loader = DataLoader(B = B, T = T, split="train")
val_loader = DataLoader(B=B, T=T, split="val")

torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig(vocab_size=50304)) # different than the default but "nicer number" (will actually make things run a bit faster)
model.to(device)
model = torch.compile(model) # compile the program for faster training

max_lr = 6e-4
min_lr = max_lr * 0.1
max_steps = 76293

def get_lr(it):
    
    # 1. return min lr if we are at the end 
    if it > max_steps:
        return min_lr
    
    # 2. if in between, use cosine decay down to min lr
    decay_ratio = (it) / (max_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # starts at 1 and goes to 0 
    return min_lr * coeff * (max_lr - min_lr)

#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# Path for saving the checkpoint
checkpoint_path = "gpt_checkpoint.pth"

# Load checkpoint if exists
start_step = load_checkpoint(model, optimizer, path=checkpoint_path)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range (max_steps):
    last_step = (step == max_steps - 1)
    # once in a while evaluate our validation loss
    if step % 100 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

        print(f"validation loss: {val_loss_accum.item():.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")
    if step > 0 and (step % 5000 == 0 or last_step):
        # optionally write model checkpoints
        checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
        checkpoint = {
        'model': model.state_dict(),
        'config': model.config,
        'step': step,
        'val_loss': val_loss_accum.item()}
        torch.save(checkpoint, checkpoint_path)

    # once in a while generate from the model
    if ((step % 100 == 0) or last_step):
        model.eval()
        num_return_sequences = 4
        max_length = 70
        tokens = enc.encode("Hello, I am a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(7)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
            print(f"sample {i}: {decoded}")
        print("current shard = ", train_loader.current_shard)



    optimizer.zero_grad()
    loss_accum = 0.0
    for mini_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16): #more efficient memory management but less precision (insignificant)
            logits, loss = model(x,y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:  
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    print(f"step {step}, loss: {loss_accum.item()}")
    if loss_accum.item() < min_train_loss:
        min_train_loss = loss_accum.item()

    if step % 100 == 0:
        save_checkpoint(model, optimizer, step, path=checkpoint_path)


import os
import time
import torch
import torch.optim as optim
from tokenizer import CustomTokenizer
from model import GPTLanguageModel

# Check device: CUDA if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset(filepath, tokenizer):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    lines = content.splitlines()
    has_dialogue = any(line.strip().startswith("User:") for line in lines) and \
                   any(line.strip().startswith("AI:") for line in lines)
                   
    if has_dialogue:
        dialogues = []
        current_user = None
        for line in lines:
            line = line.strip()
            if line.startswith("User:"):
                current_user = line[len("User:"):].strip()
            elif line.startswith("AI:") and current_user is not None:
                ai_text = line[len("AI:"):].strip()
                dialogues.append((current_user, ai_text))
                current_user = None
                
        # Compile a master text to build vocabulary
        all_raw_text = " ".join([f"{u} {a}" for u, a in dialogues])
        tokenizer.build_vocab(all_raw_text)
        
        # Tokenize and format all pairs
        all_ids = []
        for user_text, ai_text in dialogues:
            user_ids = tokenizer.encode(user_text)
            ai_ids = tokenizer.encode(ai_text)
            
            # Structure: <user> user_tokens <ai> ai_tokens <end>
            pair_ids = (
                [tokenizer.vocab["<user>"]] +
                user_ids +
                [tokenizer.vocab["<ai>"]] +
                ai_ids +
                [tokenizer.vocab["<end>"]]
            )
            all_ids.extend(pair_ids)
        return torch.tensor(all_ids, dtype=torch.long), "dialogue"
    else:
        # Plain text mode!
        tokenizer.build_vocab(content)
        all_ids = tokenizer.encode(content)
        return torch.tensor(all_ids, dtype=torch.long), "plain_text"

def get_batch(data, block_size, batch_size):
    # Generate random starting indices in the dataset
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    # Targets are shifted by 1 to predict the next token
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def train_model(dataset_path, vocab_save_path, weights_save_path, hyperparams=None, status_callback=None):
    """
    Main training function. Can be called from server thread or CLI.
    """
    # Default parameters for a fast local CPU/GPU training
    params = {
        "n_embd": 64,
        "n_head": 4,
        "n_layer": 4,
        "block_size": 32,
        "learning_rate": 3e-4,
        "max_iters": 1000,
        "eval_interval": 100,
        "batch_size": 16,
        "dropout": 0.1
    }
    if hyperparams:
        params.update(hyperparams)
        
    tokenizer = CustomTokenizer()
    data, dataset_format = load_dataset(dataset_path, tokenizer)
    
    # Validate that we have enough tokens to extract at least one training block
    if len(data) <= params["block_size"]:
        raise ValueError(f"The formatted dataset is empty or too small (found only {len(data)} tokens). Please check your column mappings and click 'Load & Format' again to ensure dialogues are formatted correctly in dataset.txt.")
        
    tokenizer.save(vocab_save_path)
    
    import json
    
    start_iter = 0
    target_iters = params["max_iters"]
    
    # Instantiate model
    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        n_embd=params["n_embd"],
        n_head=params["n_head"],
        n_layer=params["n_layer"],
        block_size=params["block_size"],
        dropout=params["dropout"]
    ).to(device)
    
    # Try to load existing weights if hyperparameters and vocab match
    if os.path.exists(weights_save_path) and os.path.exists(vocab_save_path):
        try:
            old_tokenizer = CustomTokenizer()
            old_tokenizer.load(vocab_save_path)
            
            if old_tokenizer.vocab_size == tokenizer.vocab_size:
                model.load_state_dict(torch.load(weights_save_path, map_location=device))
                print("Found existing weights with matching vocabulary. Loaded checkpoint.")
                
                # Check for state file to load iteration count
                state_file_path = os.path.join(os.path.dirname(weights_save_path), "training_state.json")
                if os.path.exists(state_file_path):
                    with open(state_file_path, "r", encoding="utf-8") as f:
                        state_data = json.load(f)
                        h = state_data.get("hyperparams", {})
                        
                        # Only resume iteration count if hyperparameters are identical
                        if (h.get("n_embd") == params["n_embd"] and 
                            h.get("n_head") == params["n_head"] and 
                            h.get("n_layer") == params["n_layer"]):
                            start_iter = state_data.get("iteration", 0)
                            print(f"Resuming training from iteration {start_iter}...")
                            
                            # Adjust target iteration count
                            if start_iter >= target_iters:
                                target_iters = start_iter + params["max_iters"]
                                print(f"Training for an additional {params['max_iters']} iterations up to {target_iters}...")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting fresh training.")

    # Notify callback of the format immediately now that start_iter and target_iters are resolved
    if status_callback:
        status_callback({
            "iteration": start_iter,
            "max_iters": target_iters,
            "loss": 0.0,
            "speed": 0.0,
            "status": "training",
            "dataset_format": dataset_format
        })

    optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"])
    
    print(f"Training Custom GPT model on {device}...")
    print(f"Parameters: Embedding size={params['n_embd']}, Layers={params['n_layer']}, Heads={params['n_head']}, Vocab={tokenizer.vocab_size}")
    
    start_time = time.time()
    
    # Run the loop from start_iter to target_iters
    for i in range(start_iter, target_iters):
        model.train()
        
        # Get a batch
        xb, yb = get_batch(data, params["block_size"], params["batch_size"])
        
        # Forward pass and loss
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Report status every 10 iterations or on final step
        if i % 10 == 0 or i == target_iters - 1:
            current_loss = loss.item()
            elapsed = time.time() - start_time
            speed = (i - start_iter + 1) / elapsed if elapsed > 0 else 0
            
            print(f"Iter {i:4d}: loss {current_loss:.4f} | speed {speed:.1f} it/s")
            
            if status_callback:
                # Let callback know if we want to stop
                should_stop = status_callback({
                    "iteration": i,
                    "max_iters": target_iters,
                    "loss": current_loss,
                    "speed": speed,
                    "status": "training",
                    "dataset_format": dataset_format
                })
                if should_stop:
                    print("Training stopped by user request.")
                    break
                    
    # Save the model weights
    torch.save(model.state_dict(), weights_save_path)
    print(f"Model saved to {weights_save_path}")
    
    # Save training_state.json to keep hyperparameters in sync
    state_file_path = os.path.join(os.path.dirname(weights_save_path), "training_state.json")
    try:
        state_data = {}
        if os.path.exists(state_file_path):
            with open(state_file_path, "r", encoding="utf-8") as f:
                state_data = json.load(f)
        
        state_data["status"] = "completed"
        state_data["iteration"] = target_iters
        state_data["max_iters"] = target_iters
        state_data["loss"] = current_loss if 'current_loss' in locals() else 0.0
        state_data["hyperparams"] = params
        state_data["dataset_format"] = dataset_format
        
        with open(state_file_path, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=4)
        print(f"Training state and hyperparameters saved to {state_file_path}")
    except Exception as e:
        print(f"Warning: Could not save training state json: {e}")
        
    if status_callback:
        status_callback({
            "iteration": target_iters,
            "max_iters": target_iters,
            "loss": current_loss if 'current_loss' in locals() else 0.0,
            "speed": 0,
            "status": "completed",
            "dataset_format": dataset_format
        })

if __name__ == "__main__":
    # If run directly, train with standard CLI parameters
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = os.path.join(current_dir, "dataset.txt")
    vocab_file = os.path.join(current_dir, "vocab.json")
    weights_file = os.path.join(current_dir, "model_weights.pth")
    
    train_model(dataset, vocab_file, weights_file)

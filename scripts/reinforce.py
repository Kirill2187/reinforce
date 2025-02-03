import torch
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import GenerationConfig
from datasets import Dataset
from tqdm import tqdm
import wandb
import gc

def prepare_messages(batch, tokenizer):
    messages = [[{"role": "user", "content": prompt}] for prompt in batch['prompt']]
    return list(map(lambda x: tokenizer.apply_chat_template(x, tokenize=False), messages))


def process_batch(tokenizer, model, ref_model, reward_model, messages, device, gen_length=256):
    tokenized = tokenizer(messages, return_tensors="pt", padding=True).to(device)
    prompt_length = tokenized["input_ids"].shape[1]
    if prompt_length > 1024:
        return None, None, None, None
    outputs = model.generate(
        **tokenized,
        max_length=tokenized["input_ids"].shape[1] + gen_length,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    responses = outputs[:, prompt_length:]
    
    del tokenized
    torch.cuda.empty_cache()
    gc.collect()
    
    with torch.no_grad():
        ref_logits = ref_model(outputs).logits[:, prompt_length:]
    logits = model(outputs).logits[:, prompt_length:]
                    
    log_probs = torch.log_softmax(logits, dim=-1)
    ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                
    sampled_log_probs = log_probs.gather(2, responses.unsqueeze(-1)).squeeze(-1).sum(-1)
    sampled_ref_log_probs = ref_log_probs.gather(2, responses.unsqueeze(-1)).squeeze(-1).sum(-1)
    kl = (sampled_log_probs - sampled_ref_log_probs).detach()
    
    del ref_logits, ref_log_probs, sampled_ref_log_probs
    torch.cuda.empty_cache()
    gc.collect()
    
    rewards = reward_model(outputs).logits
    if rewards.dim() > 1 and rewards.shape[-1] == 1:
        rewards = rewards.squeeze(-1)
        
    return outputs, sampled_log_probs, kl, rewards


def run_validation(val_loader, tokenizer, model, ref_model, reward_model, device, episode):
    total_reward = 0.0
    total_kl = 0.0
    count = 0

    table = wandb.Table(columns=["prompt", "generation"])
    model.eval()
    with torch.no_grad():
        for val_batch in tqdm(val_loader, desc="Validation"):
            messages = prepare_messages(val_batch, tokenizer)
            outputs, _, kl_val, rewards_val = process_batch(tokenizer, model, ref_model, reward_model, messages, device)
            if outputs is None:
                continue
            total_reward += rewards_val.mean().item()
            total_kl += kl_val.mean().item()
            count += 1
            prompts = val_batch["prompt"]
            for i in range(len(prompts)):
                generation = tokenizer.decode(outputs[i])
                table.add_data(prompts[i], generation)
    mean_reward = total_reward / count
    mean_kl = total_kl / count
    print(f"Validation - Avg. Reward: {mean_reward:.4f}, Avg. KL: {mean_kl:.4f}")
    wandb.log({
        "episode": episode,
        "val/avg_reward": mean_reward,
        "val/avg_kl": mean_kl,
        "val/generations_table": table
    })
    model.train()


def reinforce_finetune(
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    reward_model_name: str = "flypew/reward_model",
    train_dataset: Dataset = None,
    val_dataset: Dataset = None,
    batch_size: int = 2,
    val_batch_size: int = 2,
    gradient_accumulation_steps: int = 32,
    learning_rate: float = 1e-5,
    num_steps: int = 64 * 500,
    beta = 0.03,
    validate_every: int = 64 * 10,
    wandb_api_key: str = None,
    wandb_project: str = None,
    hf_token: str = None,
    hf_model_id: str = "flypew/rlhf_model"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.train()
    
    ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    ref_model.eval()
    
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name).to(device)
    reward_model.eval()
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    if wandb_api_key and wandb_project:
        wandb.login(key=wandb_api_key)
        if wandb.run is not None:
            wandb.finish()
        wandb.init(project=wandb_project)
    
    optimizer.zero_grad()
    running_reward_sum = 0.0
    running_count = 0

    train_iterator = iter(train_loader)
    for step in range(num_steps):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        
        messages = prepare_messages(batch, tokenizer)
        
        outputs, sampled_log_probs, kl, rewards = process_batch(tokenizer, model, ref_model, reward_model, messages, device)
        if outputs is None:
            print("Skipping batch due to excessive length")
            continue
        
        print("Prompt:", batch["prompt"][0])
        print("Generated:", tokenizer.decode(outputs[0]))
        print("KL:", kl.detach().cpu().numpy())
        print("Rewards:", rewards.detach().cpu().numpy())
                
        rlhf_reward = rewards - beta * kl

        batch_mean_reward = rlhf_reward.mean().item()
        baseline = batch_mean_reward if running_count == 0 else running_reward_sum / running_count
        advantage = rlhf_reward - baseline

        loss = -(sampled_log_probs * advantage).mean() / gradient_accumulation_steps
        loss.backward()

        running_reward_sum += batch_mean_reward
        running_count += 1

        episode = (step + 1) * batch_size
        wandb.log({
            "episode": episode,
            "train/avg_reward": rewards.mean().item(),
            "train/avg_kl": kl.mean().item(),
            "train/avg_rlhf_reward": batch_mean_reward,
        })

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (step + 1) % validate_every == 0:
            run_validation(val_loader, tokenizer, model, ref_model, reward_model, device, episode)

        print('-' * 50)
        
        del outputs, sampled_log_probs, kl, rewards, rlhf_reward, batch_mean_reward, baseline, advantage, loss
        torch.cuda.empty_cache()
        gc.collect()
        
    if hf_token and hf_model_id:
        from huggingface_hub import login
        login(token=hf_token)
        model.save_pretrained(hf_model_id)
        model.push_to_hub(hf_model_id)
    
    if wandb_api_key and wandb_project:
        wandb.finish()

                
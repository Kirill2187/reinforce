import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from trl import RewardTrainer, RewardConfig
from utils import set_all_seeds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--dataset", type=str, default="esfrankel17/HelpSteer2_binarized")
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default="./reward_model")
    parser.add_argument("--hf_token", type=str, default=None, help="HF token for pushing to the hub")
    args = parser.parse_args()
    set_all_seeds(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, padding_side="left")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, num_labels=1
    ).to(device)
    reward_model.config.pad_token_id = 0

    dataset = load_dataset(args.dataset)['average_rating_split'].train_test_split(test_size=0.1)
    if "prompt" in dataset["train"].features:
        dataset = dataset.remove_columns("prompt")
    train_dataset = dataset['train']
    val_dataset = dataset['test']

    training_args = RewardConfig(
        num_train_epochs=args.train_epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        max_length=1024,
        fp16=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        center_rewards_coefficient=0.1,
    )

    trainer = RewardTrainer(
        model=reward_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    metrics = trainer.evaluate(num_print_samples=0)
    print(metrics)
    trainer.save_metrics("eval", metrics)

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
        trainer.push_to_hub()

if __name__ == "__main__":
    main()
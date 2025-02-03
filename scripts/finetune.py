import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from reinforce import reinforce_finetune
from trl import RewardTrainer, RewardConfig
from utils import set_all_seeds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="esfrankel17/HelpSteer2_binarized")
    parser.add_argument("--hf_token", type=str, default=None, help="HF token for pushing to the hub")
    parser.add_argument("--wandb_token", type=str, default=None, help="Wandb API token")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
    parser.add_argument("--val_size", type=int, default=64, help="Validation size")
    args = parser.parse_args()
    set_all_seeds(42)
    
    dataset = load_dataset(args.dataset)['average_rating_split'].train_test_split(test_size=args.val_size)

    reinforce_finetune(
        train_dataset=dataset['train'],
        val_dataset=dataset['test'],
        wandb_api_key=args.wandb_token,
        wandb_project=args.wandb_project,
        hf_token=args.hf_token
    )

if __name__ == "__main__":
    main()
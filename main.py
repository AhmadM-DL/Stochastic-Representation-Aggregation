import yaml
from dataset_loader import get_dataloaders_from_hf
from datasets import load_dataset
from model_loader import load_vit_model
from trainer import train_model

def main(model_name, dataset_name, strategy, config_path, checkpoint_root_path, load_checkpoint):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    model_id = [model['huggingface_id'] for model in config['models'] if model['name']==model_name][0]
    dataset_id = [dataset['huggingface_id'] for dataset in config['datasets'] if dataset["name"] == dataset_name][0]
    train_args = config['training']

    model, preprcessor = load_vit_model(model_id)
    dataset_hf = load_dataset(dataset_id)

    train_dl, val_dl, test_dl = get_dataloaders_from_hf(dataset_hf, train_args['batch_size'], preprcessor)

    for trial in range(train_args.get('num_trials')):
        print(f"Running trial {trial + 1} for model {model_name} on dataset {dataset_name}")
        train_model(model, dataset_name, strategy, trial,
                    train_dl, val_dl, test_dl,
                    train_args['num_epochs'],
                    train_args['learning_rate'],
                    train_args['optimizer'],
                    train_args['momentum'],
                    train_args['weight_decay'],
                    train_args['scheduler'],
                    load_checkpoint = load_checkpoint,
                    saving_path = checkpoint_root_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True, help='Model name')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Dataset name')
    parser.add_argument('--config', '-c',  type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--strategy', '-s',  type=str, required=True, help='Aggregation strategy to use')
    parser.add_argument('--checkpoint', '-chkp', action='store_true', default=False, help='Load from checkpoint')
    parser.add_argument('--checkpoint_root_path', '-chkpp', type=str, default='results', help='Root path to save results')
    args = parser.parse_args()
    main(args.model, args.dataset, args.strategy, args.config, args.checkpoint_root_path, args.checkpoint)
import torch

from config.structured import THuman2Config, DataloaderConfig, ProjectConfig
from .dataset import THuman

def get_dataset(cfg: ProjectConfig):
    dataset_cfg: THuman2Config = cfg.dataset
    dataloader_cfg: DataloaderConfig = cfg.dataloader

    train_dataset = THuman(dataset_cfg, 'train', cfg.run.job)
    val_dataset = THuman(dataset_cfg, 'val', cfg.run.job)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=dataloader_cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=dataloader_cfg.num_workers)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=dataloader_cfg.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=dataloader_cfg.num_workers)
    
    if cfg.run.job in ['sample', 'vis']:
        batch_size = dataloader_cfg.batch_size
    else:
        batch_size = 1

    if dataset_cfg.type == 'hi4d':
        batch_size = batch_size // 2
        def collate_fn(batch):
            # Initialize an empty dictionary to store the batch
            batch_dict = {}

            # Iterate over each sample in the batch
            for sample in batch:
                # Iterate over each dictionary in the sample
                for dictionary in sample:
                    # Iterate over each key-value pair in the dictionary
                    for key, value in dictionary.items():
                        # Append the value to the list corresponding to the key in the batch dictionary
                        if key not in batch_dict:
                            batch_dict[key] = []
                        batch_dict[key].append(value)

            return batch_dict

    elif dataset_cfg.type == 'multihuman':
        def collate_fn(batch):
            # Initialize an empty dictionary to store the batch
            batch_dict = {}

            # Iterate over each sample in the batch
            for sample in batch:
                # Iterate over each key-value pair in the dictionary
                for key, value in sample.items():
                    # Append the value to the list corresponding to the key in the batch dictionary
                    if key not in batch_dict:
                        batch_dict[key] = []
                    batch_dict[key].append(value)

            return batch_dict

    else: collate_fn = None

    test_dataset = THuman(dataset_cfg, 'test', cfg.run.job)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=dataloader_cfg.num_workers,
        collate_fn=collate_fn)
    
    return train_data_loader, val_data_loader, test_data_loader

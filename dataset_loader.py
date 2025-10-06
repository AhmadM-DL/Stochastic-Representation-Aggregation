from torch.utils.data import Dataset, DataLoader

def get_dataloaders_from_hf(dataset_hf, batch_size, preprocessor, num_workers=2):

    train_dataset_hf = dataset_hf['train']
    split_dataset_hf = train_dataset_hf.train_test_split(test_size=0.2, seed=42)
    
    train_dataset_hf = split_dataset_hf['train']
    val_dataset_hf = split_dataset_hf['test']
    test_dataset_hf = dataset_hf['test']

    train_dataset = CustomDataset(train_dataset_hf, preprocessor)
    val_dataset = CustomDataset(val_dataset_hf, preprocessor)
    test_dataset = CustomDataset(test_dataset_hf, preprocessor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

class CustomDataset(Dataset):
    def __init__(self, dataset, preprocessor):
        self.dataset = dataset
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.dataset)

    def __get_label_key__(self):
        keys = self.dataset['train'].column_names
        label_keys = ('label', 'fine_label', 'category_id', 'target', 'labels', 'species', 'class')
        for key in label_keys:
            if key in keys:
                return key
        raise Exception(f"Unsupported dataset. Available keys: {keys}")
    
    def __get_image_key__(self):
        keys = self.dataset['train'].column_names
        label_keys = ('image', 'img', 'pixels')
        for key in label_keys:
            if key in keys:
                return keys
        raise Exception(f"Unsupported dataset. Available keys: {keys}")

    def __getitem__(self, idx):
        item = self.dataset[idx]
        label = item[self.__get_label_key__()]
        image = item[self.__get_image_key__()]
        inputs = self.preprocessor(image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        return pixel_values, label


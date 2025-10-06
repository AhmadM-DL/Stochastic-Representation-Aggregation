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

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Handle different dataset formats
        image = None
        label = None
        # Common cases
        if 'image' in item:
            image = item['image']
        elif 'img' in item:
            image = item['img']
        elif 'pixels' in item:
            image = item['pixels']
        # Label field can vary
        if 'label' in item:
            label = item['label']
        elif 'fine_label' in item:
            label = item['fine_label']
        elif 'category_id' in item:
            label = item['category_id']
        elif 'target' in item:
            label = item['target']
        elif 'labels' in item:
            label = item['labels']
        # Some datasets (e.g. Fungi) use 'species' or 'class'
        elif 'species' in item:
            label = item['species']
        elif 'class' in item:
            label = item['class']
        # Apply model-specific preprocessing
        inputs = self.preprocessor(image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        return pixel_values, label
    

import torch
from transformers import AutoModel, AutoImageProcessor

def load_vit_model(model_name):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This implementation requires a GPU.")
    
    model = AutoModel.from_pretrained(model_name)
    model = model.cuda()
    preprocessor = AutoImageProcessor.from_pretrained(model_name)
    model.eval()
    return model, preprocessor
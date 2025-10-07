import torch
from transformers import AutoModel, AutoImageProcessor

def load_vit_model(model_name):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This implementation requires a GPU.")
    
    model = AutoModel.from_pretrained(model_name)
    model = model.cuda()
    preprocessor = AutoImageProcessor.from_pretrained(model_name, use_fast= True)
    model.eval()
    return model, preprocessor

def get_model_hidden_size(model):
    model_name = type(model).__name__ 
    if "clip" in model_name.lower():
        return model.vision_model.config.hidden_size
    else:
        return model.config.hidden_size

def get_raw_features(model, inputs):
    model_name = type(model).__name__ 
    if "clip" in model_name.lower():
        outputs = model.vision_model(inputs)
        return outputs.last_hidden_state
    else:
        outputs = model(inputs)
        return outputs.last_hidden_state
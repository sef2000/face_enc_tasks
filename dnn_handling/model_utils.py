import os
import torch
import torch.nn as nn
import torchvision.models as models
import warnings
"""
Code by John Miao <3
"""

# ==========================================
# Architecture 1: Caffe Style (Standard)
# ==========================================
class AlexNetCaffe(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNetCaffe, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # 0
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),  # 3

            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # 4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),  # 7

            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # 8
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # 10
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ==========================================
# Architecture 2: BN Large (No First Pool)
# ==========================================
class AlexNetBNLarge(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNetBNLarge, self).__init__()
        self.features = nn.Sequential(
            # Block 1: Conv -> BN -> ReLU (NO POOLING)
            # Indices: 0, 1, 2
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # MISSING MAXPOOL HERE (Explains why next layer is at 3)

            # Block 2: Conv -> BN -> ReLU -> MaxPool
            # Indices: 3, 4, 5, 6
            # We use stride=2 here to compensate for the missing pool above
            nn.Conv2d(64, 192, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Block 3: Conv -> BN -> ReLU
            # Indices: 7, 8, 9
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # Block 4: Conv -> BN -> ReLU
            # Indices: 10, 11, 12
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 5: Conv -> BN -> ReLU
            # Indices: 13, 14, 15
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # NO FINAL POOLING
        )

        self.avgpool = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(43264, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ==========================================
# Loader Function
# ==========================================
def load_alexnet_auto(model_path, num_classes=1000, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Path not found: {model_path}")

    print(f"\n--- Loading: {os.path.basename(model_path)} ---")

    # 1. Load weights
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Failed to load file: {e}")
        return None

    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

    # Clean keys
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    state_dict = new_state_dict

    # 2. Auto-Detect
    model = None

    # Check features.3.weight
    # If it exists, it means Layer 3 is a Conv layer (BN-Large structure).
    # If it does NOT exist (and is LRN/Pool), it is Caffe structure.

    layer3_weight = state_dict.get('features.3.weight')

    if layer3_weight is not None:
        print(f">> Detection: BN-Large Architecture (Shifted indices).")
        model = AlexNetBNLarge(num_classes=num_classes)
    else:
        print(f">> Detection: Caffe Architecture.")
        model = AlexNetCaffe(num_classes=num_classes)

    # 3. Load
    model = model.to(device)
    try:
        model.load_state_dict(state_dict, strict=True)
        print(">> Success: Loaded with strict=True")
    except RuntimeError:
        print(">> strict=True failed (likely last layer mismatch). Retrying strict=False...")
        model.load_state_dict(state_dict, strict=False)
        print(">> Success: Loaded with strict=False")

    model.eval()
    return model

# def load_model(model_path=None, model_type='vgg16', use_gpu=True, multi_gpu=False, num_classes=2137, device=None):
#     if device is None:
#         device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
#
#     # --- INSPECTION STEP ---
#     # We must load the state_dict FIRST to decide which architecture to build
#     state_dict = None
#     if model_path and os.path.exists(model_path):
#         print(f"Inspecting checkpoint: {model_path}...")
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             checkpoint = torch.load(model_path, map_location=device, weights_only=False)
#
#         state_dict = checkpoint.get('state_dict', checkpoint)
#
#         # CLEAN KEYS (remove 'module.')
#         new_state_dict = {}
#         for k, v in state_dict.items():
#             name = k[7:] if k.startswith('module.') else k
#             new_state_dict[name] = v
#         state_dict = new_state_dict
#
#     # --- MODEL SELECTION ---
#     model = None
#
#     if model_type == 'alexnet':
#         if state_dict is not None:
#             # CHECK 1: Look at First Convolution Filters
#             conv1_weight = state_dict.get('features.0.weight')
#
#             # CHECK 2: Look at Classifier Input Size
#             fc1_weight = state_dict.get('classifier.1.weight')  # Linear(input, 4096)
#
#             if conv1_weight is not None and conv1_weight.shape[0] == 96:
#                 print(">> Detected Architecture: AlexNet Caffe (96 filters)")
#                 model = AlexNetCaffe(num_classes=num_classes)
#
#             elif fc1_weight is not None and fc1_weight.shape[1] == 43264:
#                 print(">> Detected Architecture: AlexNet Large-BN (64 filters, 13x13 output)")
#                 model = AlexNetBNLarge(num_classes=num_classes)
#             else:
#                 # Fallback
#                 print(">> Warning: Architecture unclear. Defaulting to AlexNetCaffe.")
#                 model = AlexNetCaffe(num_classes=num_classes)
#         else:
#             # Default if no path provided
#             model = AlexNetCaffe(num_classes=num_classes)
#
#     elif model_type == 'vgg16':
#         model = models.vgg16(num_classes=num_classes)
#     elif model_type == 'resnet50':
#         model = models.resnet50(num_classes=num_classes)
#     elif model_type == 'vgg19':
#         model = models.vgg19(weights=None)
#
#     # --- LOAD WEIGHTS ---
#     if state_dict is not None:
#         try:
#             model.load_state_dict(state_dict, strict=True)
#             print("Success! Model loaded perfectly (Strict).")
#         except RuntimeError as e:
#             print(f"Strict load failed. Message:\n{str(e)[:200]}...")  # Print first 200 chars
#             print("Retrying with strict=False...")
#             model.load_state_dict(state_dict, strict=False)
#             print("Model loaded with strict=False.")
#
#     model = model.to(device)
#     model.eval()
#     return model

def load_model(model_path=None, model_type='vgg16', use_gpu=True, multi_gpu=False, num_classes=2137, device=None):
    if device is None:
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # --- INSPECTION STEP ---
    state_dict = None

    # 1. Check if path is provided
    if model_path is not None:
        # 2. Check if file exists
        if not os.path.exists(model_path):
            print(f"❌ CRITICAL ERROR: Model path provided but file NOT found: {model_path}")
            return None  # Stop here if file is missing

        print(f"Inspecting checkpoint: {model_path}...")

        # 3. Attempt to load the file
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Failed to load file. It might be corrupted. Error: {e}")
            return None

        # 4. Extract state_dict safely
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            print(f"❌ CRITICAL ERROR: Checkpoint is not a valid dictionary or state_dict.")
            return None

        # CLEAN KEYS (remove 'module.')
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        state_dict = new_state_dict

    # --- MODEL SELECTION ---
    model = None

    if model_type == 'alexnet':
        if state_dict is not None:
            # CHECK 1: Look at First Convolution Filters
            conv1_weight = state_dict.get('features.0.weight')

            # CHECK 2: Look at Classifier Input Size
            fc1_weight = state_dict.get('classifier.1.weight')  # Linear(input, 4096)

            if conv1_weight is not None and conv1_weight.shape[0] == 96:
                print(">> Detected Architecture: AlexNet Caffe (96 filters)")
                # Assuming AlexNetCaffe is defined elsewhere or imported
                model = AlexNetCaffe(num_classes=num_classes)

            elif fc1_weight is not None and fc1_weight.shape[1] == 43264:
                print(">> Detected Architecture: AlexNet Large-BN (64 filters, 13x13 output)")
                # Assuming AlexNetBNLarge is defined elsewhere or imported
                model = AlexNetBNLarge(num_classes=num_classes)
            else:
                # Fallback
                print(">> Warning: Architecture unclear. Defaulting to AlexNetCaffe.")
                model = AlexNetCaffe(num_classes=num_classes)
        else:
            # Default if no path provided
            model = AlexNetCaffe(num_classes=num_classes)

    elif model_type == 'vgg16':
        model = models.vgg16(num_classes=num_classes)
    elif model_type == 'resnet50':
        model = models.resnet50(num_classes=num_classes)
    elif model_type == 'vgg19':
        model = models.vgg19(weights=None)

    # If model creation failed for some reason
    if model is None:
        print(f"❌ ERROR: Model type '{model_type}' not recognized or failed to initialize.")
        return None

    # --- LOAD WEIGHTS ---
    if state_dict is not None:
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✅ Success! Model loaded perfectly (Strict).")
        except RuntimeError as e:
            print(f"⚠️ Strict load failed. Message:\n{str(e)[:200]}...")  # Print first 200 chars
            print("Retrying with strict=False...")
            try:
                model.load_state_dict(state_dict, strict=False)
                print("✅ Model loaded with strict=False.")
            except Exception as e2:
                print(f"❌ CRITICAL ERROR: Could not load weights even with strict=False. {e2}")
                return None
    elif model_path is not None:
        # Path was provided but we failed to get state_dict logic correct above
        print("❌ CRITICAL ERROR: Model path provided but state_dict is None.")
        return None

    model = model.to(device)
    model.eval()
    return model

def get_layers(model, layer_types=None):
    """
    Get layers of specified types from a model.

    Args:
        model: PyTorch model
        layer_types: List of layer types to extract (e.g., [nn.Conv2d, nn.ReLU, nn.MaxPool2d])
                    If None, defaults to only convolutional layers for backward compatibility

    Returns:
        list: List of tuples (layer_name, layer_module) for specified layer types
    """
    if layer_types is None:
        # Default to only convolutional layers for backward compatibility
        layer_types = [nn.Conv2d]

    layers = []

    # Handle DataParallel models
    if hasattr(model, 'module'):
        model = model.module

    # For VGG models, layers are in the features module
    if hasattr(model, 'features'):
        for i, layer in enumerate(model.features):
            if any(isinstance(layer, layer_type) for layer_type in layer_types):
                layers.append((f'features.{i}', layer))

    # For other models, search through all named modules
    else:
        for name, module in model.named_modules():
            if any(isinstance(module, layer_type) for layer_type in layer_types):
                layers.append((name, module))

    return layers


def get_conv_layers(model):
    """
    Get all convolutional layers from a model.

    Args:
        model: PyTorch model

    Returns:
        list: List of tuples (layer_name, layer_module) for convolutional layers
    """
    # Maintain backward compatibility by only extracting Conv2d layers
    return get_layers(model, layer_types=[nn.Conv2d])


def get_activation_layers(model):
    """
    Get all activation layers from a model.

    Args:
        model: PyTorch model

    Returns:
        list: List of tuples (layer_name, layer_module) for activation layers
    """
    return get_layers(model, layer_types=[nn.ReLU, nn.LeakyReLU, nn.GELU, nn.Sigmoid, nn.Tanh])


def get_pooling_layers(model):
    """
    Get all pooling layers from a model.

    Args:
        model: PyTorch model

    Returns:
        list: List of tuples (layer_name, layer_module) for pooling layers
    """
    return get_layers(model, layer_types=[nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d])


def get_layer_by_index(model, layer_index, layer_types=None):
    """
    Get a specific layer from the model by its index in the list of layers of specified types.

    Args:
        model: PyTorch model
        layer_index (int): Index of the layer
        layer_types: List of layer types to consider (e.g., [nn.Conv2d, nn.ReLU, nn.MaxPool2d])
                    If None, defaults to only convolutional layers for backward compatibility

    Returns:
        tuple: (layer_name, layer_module)
    """
    layers = get_layers(model, layer_types)

    if 0 <= layer_index < len(layers):
        return layers[layer_index]
    else:
        raise IndexError(f"Layer index {layer_index} out of range (0-{len(layers) - 1})")
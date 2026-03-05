import torch
from model_utils import load_model
from PIL import Image
import glob
import numpy as np
# import vgg preproc
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = "dual_random"  # choose from 'face_vgg', 'inanimate_vgg', 'face_one', 'inanimate_one', 'dual_vgg', 'dual_random'

activations = None

def activation_hook(module, input, output):
    global activations
    activations = output.detach().cpu().numpy()

if network == "face_vgg":
    model_path = "/raid/katha/BranchingNets/checkpoints/vgg/face_vgg_large/epoch_100.pth.tar"
    output = 1714
elif network == "inanimate_vgg":
    model_path = "/raid/katha/BranchingNets/checkpoints/vgg/inanimate_vgg_large/epoch_100.pth.tar"
    output = 423
elif network == "face_one":
    model_path = "/raid/katha/BranchingNets/checkpoints/vgg/face_inanimate_400k_facesconsolidated/epoch_110.pth.tar"
    output = 424
elif network == "inanimate_one":
    model_path = "/raid/katha/BranchingNets/checkpoints/vgg/face_inanimate_400k_objectsconsolidated/epoch_110.pth.tar"
    output = 1715
elif network == "dual_vgg":
    model_path = "/raid/katha/BranchingNets/checkpoints/vgg/face_inanimate_400k_seed/epoch_50.pth.tar"
    output = 2137
elif network == "dual_random":
    model_path = "/raid/katha/BranchingNets/checkpoints/vgg/face_inanimate_400k_seed2/epoch_0.pth.tar"
    output = 2137
else:
    raise ValueError("Invalid network name. Choose from 'face_vgg', 'inanimate_vgg', 'face_one', 'inanimate_one', 'dual_vgg'.")

vgg = load_model(model_path=model_path, num_classes=output, use_gpu=True, multi_gpu=False, device=device)

print(vgg)

# register hook; post relu
vgg.classifier[5].register_forward_hook(activation_hook)

# get vgg transforms
vgg_preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

path = "/data/saskia_fohs/enc_phys/CelebA_face_new"
all_ims = glob.glob(f"{path}/*.jpg")

# put first img through network
for i, stim in tqdm(enumerate(all_ims)):
    img = Image.open(stim).convert("RGB")
    img_preprocessed = vgg_preprocess(img).unsqueeze(0).to(device)
    vgg.eval()
    with torch.no_grad():
        _ = vgg(img_preprocessed)
    if i == 0:
        all_activations = activations
    else:
        all_activations = np.concatenate((all_activations, activations), axis=0)

# save as npy
np.save("/data/saskia_fohs/enc_phys/celeb_penult_activations_" + network + ".npy", all_activations)
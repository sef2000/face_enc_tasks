import numpy as np
from encoder import EncodingRidge

models = ["face_vgg", "inanimate_vgg", "face_one", "inanimate_one", "dual_vgg", "dual_random"]
targets = np.load("/data/saskia_fohs/enc_phys/Data/celeb_neurons.npy") # shape (neurons, trials)

# switch targets to shape (trials, targets) for sklearn
targets = targets.swapaxes(0,1) # shape (trials, neurons)

model_input = []
for model in models:
    activations = np.load(f"/data/saskia_fohs/enc_phys/celeb_penult_activations_" + model + ".npy") # shape (trials, features)
    model_input.append(activations)
model_input = np.array(model_input) # shape (models, trials, features)

# check if there are any nans in model_input or targets
# check in what target is nan and print it
nan_target = np.where(np.isnan(targets))
targets[nan_target] = 0 # as in her paper https://www.nature.com/articles/s41562-025-02218-1#Abs1

print(model_input.shape)
print(targets.shape)

encoding = EncodingRidge(scoring="explained_variance", cv=5)
scores = encoding.folds(model_input, targets) # shape (cv, models, neurons)

# save scores
np.save("/data/saskia_fohs/enc_phys/encoding_scores_celeb_pls.npy", scores)
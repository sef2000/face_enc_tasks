import numpy as np
import matplotlib.pyplot as plt

# read encoding scores
encoding_scores = np.load("/data/saskia_fohs/enc_phys/encoding_scores_celeb.npy")
print(encoding_scores.shape)

# collapse first dimension (cv) by taking mean across it
mean_encoding_scores = np.mean(encoding_scores, axis=0) # shape (models, neurons)

# count how many vals per model are above 0
for i in range(mean_encoding_scores.shape[0]):
    count_above_zero = np.sum(mean_encoding_scores[i] > 0)
    print(f"Model {i+1} has {count_above_zero} neurons with mean encoding score above 0.")
    # index print of neurons with mean encoding score above 0#
    above_zero_indices = np.where(mean_encoding_scores[i] > 0)[0]
    print(f"Neurons with mean encoding score above 0 for model {i+1}: {above_zero_indices}")
    # print mean encoding score of neurons with mean encoding score above 0
    mean_scores_above_zero = mean_encoding_scores[i][above_zero_indices]
    print(f"Mean encoding scores of neurons with mean encoding score above 0 for model {i+1}: {mean_scores_above_zero}")

models = ["face_vgg", "inanimate_vgg", "face_one", "inanimate_one", "dual_vgg", "dual_random"]

# plot histogram of mean encoding scores in subplots
plt.figure(figsize=(12, 8))
for i in range(mean_encoding_scores.shape[0]):
    plt.subplot(2, 3, i+1)
    plt.hist(mean_encoding_scores[i], bins=20)
    plt.title(f"Model {i+1} ({models[i]})")
plt.tight_layout()
plt.show()

# scatter plot of model 1, model 2, model 3, model 4, model 5, model 6
plt.figure(figsize=(10, 6))
plt.scatter(mean_encoding_scores[0], mean_encoding_scores[5], alpha=0.5)
plt.ylabel("Mean Encoding Score Model 6 (dual_random)")
plt.xlabel("Mean Encoding Score Model 1 (face_vgg)")
plt.show()
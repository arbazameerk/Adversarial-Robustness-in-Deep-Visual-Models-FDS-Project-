# Adversarial Attacks on ResNet - MNIST Digit Classification

**Author:** Arbaz Ameer Khan

---

## Contents

1. [Objective](#objective)
2. [Methodology](#methodology)
   - [Dataset Preparation](#dataset-preparation)
   - [Model Architecture](#model-architecture)
   - [Training Setup](#training-setup)
   - [FGSM Attack Implementation](#fgsm-attack-implementation)
   - [Attack Hyperparameters](#attack-hyperparameters)
   - [Evaluation Metrics](#evaluation-metrics)
3. [Experiments](#experiments)
   - [Setup](#setup)
   - [Evaluation](#evaluation)
4. [Challenges](#challenges)
5. [References](#references)

---

## Objective

For this project, I chose adversarial attacks to investigate how vulnerable a trained ResNet model is when fooled with adversarial examples. I took correctly classified images and added tiny perturbations—almost invisible to humans—that mess up predictions. I trained a ResNet on MNIST digits, implemented the FGSM attack, and measured attack success and how strength affects performance.

---

## Methodology

### Dataset Preparation

I used the MNIST dataset with digits 0 to 9, consisting of 60,000 training images and 10,000 test images. Images were resized from 28×28 to 32×32 for ResNet compatibility.

The transformation applied can be written as:

*Transform = Resize to 32×32 → Convert to tensor → Normalize*

For normalization, the formula is:

```
normalized_pixel = (original_pixel - 0.5) / 0.5
```

This shifts pixels from the [0, 1] range to the [-1, 1] range for better gradient flow. The training loader had `shuffle=True` for random batches, while testing had `shuffle=False`.

### Model Architecture

I used a simplified ResNet with skip connections to avoid vanishing gradients. The initial convolutional layer converts a single channel to 8 features using 3×3 convolutions.

The ResNet has three main components:

**1. Initial Convolutional Layer**

This takes the single-channel grayscale image and converts it to 8 feature channels using 3×3 convolutions. The operation is:

```
output = ReLU(BatchNorm(Conv2D(input)))
```

The convolution formula for each output pixel is:

```
output(i,j) = Σ (input_values × kernel_weights) + bias
```

I kept it at 8 channels instead of the usual 16 or 32 because I wanted a simpler model that wouldn't overfit on MNIST. After the convolutional layer, I apply batch normalization and ReLU activation.

Batch normalization works like this:

```
normalized_value = (value - batch_mean) / √(batch_variance + ε)
```

Then it applies learnable scale and shift parameters.

ReLU activation is:

```
ReLU(x) = max(0, x)
```

So negative values become zero and positive values stay the same.

**2. Three ResNet Blocks**

Each block has two convolutional layers with batch normalization, and the critical part is the skip connection.

The ResNet block formula is:

```
output = ReLU(F(x) + x)
```

where `F(x)` is the result of the two convolutional layers with batch norm and `x` is the input. This addition of `x` is the skip connection and it's what makes ResNet special.

The three blocks go from 8→8 channels, then 8→16, then 16→32. The second and third blocks use stride 2, which downsamples the spatial dimensions. Stride 2 means the filter moves 2 pixels at a time instead of 1, so the output is half the size.

**3. Final Layers**

Finally, I have adaptive average pooling that squashes the spatial dimensions down to 1×1, and then a fully connected layer that outputs 10 values for the 10 digit classes.

Average pooling formula is:

```
pooled_value = Σ(values_in_region) / number_of_values
```

The fully connected layer is:

```
output = input × weight_matrix + bias_vector
```

I kept the model relatively small with only about 20,000 parameters total. This was intentional because MNIST is not that complex and I didn't want to overfit.

### Training Setup

**Hyperparameters:**
- Batch size: 64
- Learning rate: 0.01
- Epochs: 3
- Training device: CPU

**Optimizer:** Adam with moving averages:

```
first_moment = 0.9 × old_first_moment + 0.1 × gradient
second_moment = 0.999 × old_second_moment + 0.001 × gradient²
```

Parameter update:

```
new_parameter = old_parameter - learning_rate × first_moment / (√second_moment + ε)
```

**Loss Function:** CrossEntropyLoss:

```
loss = -Σ (true_label × log(predicted_probability))
```

For a single correct class:

```
loss = -log(probability_assigned_to_correct_class)
```

Backpropagation uses the chain rule. I saved the model as a pickle file with weights, optimizer state, and architecture.

### FGSM Attack Implementation

The Fast Gradient Sign Method (FGSM) calculates the gradient of loss with respect to input and adjusts the image to increase loss.

The FGSM formula is:

```
adversarial_image = original_image + ε × sign(∇_input Loss)
```

The sign function returns:

```
sign(x) = +1 if x > 0, else -1
```

I'm basically asking which direction each pixel should move to hurt the model most, and then moving it in that direction by epsilon amount. After adding the perturbation, I clip values to stay in the valid range:

```
clipped_value = min(1, max(-1, value))
```

The reason I take just the sign instead of the full gradient is because I want to make the most efficient attack possible. The sign gives the direction that increases loss the most per pixel. The gradient of loss with respect to input is computed using backpropagation just like for training, except I keep track of gradients for the input pixels instead of model weights.

### Attack Hyperparameters

The key parameter epsilon (ε) controls perturbation strength. I tested seven epsilon values from 0.0 to 0.3 in 0.05 steps. Perturbation is bounded by:

```
perturbation ∈ [-ε, +ε]
```

I tested on 1,000 samples per epsilon. I used ε = 0.15 for confusion matrix and visualizations.

### Evaluation Metrics

I tracked several metrics to understand attack performance:

**Clean Accuracy:**
```
clean_accuracy = (number_of_correct_predictions / total_images) × 100
```

**Adversarial Accuracy:**
```
adversarial_accuracy = (correct_predictions_on_perturbed_images / total_images) × 100
```

**Attack Success Rate (ASR):**
```
ASR = 100 - adversarial_accuracy
```

or equivalently:

```
ASR = (misclassified_adversarial_images / total_images) × 100
```

I also tracked confidence scores from the softmax function:

```
confidence_i = exp(output_i) / Σ exp(output_j)
```

This converts raw model outputs into probabilities that sum to 1.

---

## Experiments

### Setup

**CNN Architecture:** Simplified ResNet with 8→8, 8→16, 16→32 channel blocks with skip connections, stride 2 downsampling, ~20,000 parameters total.

**Dataset:** MNIST, 60,000 training, 10,000 test images, resized from 28×28 to 32×32, normalized to [-1, 1].

**Training Settings:** 3 epochs, batch size 64, learning rate 0.01, Adam optimizer, CrossEntropyLoss, CPU training.

**Model Performance:** 98.41% test accuracy, 98.40% training accuracy. Figure 6 shows loss decreasing from 0.214 to 0.064 to 0.051 across epochs.

<img width="1785" height="735" alt="graph6_training_curves" src="https://github.com/user-attachments/assets/206c025f-4492-453d-8e5f-1833e91fa4ef" />

### Evaluation

#### Basic Evaluation

I implemented FGSM attacks with epsilon values ranging from 0.0 to 0.3 in steps of 0.05. Each configuration was tested on 1,000 correctly classified samples.

<img width="1486" height="882" alt="graph1_accuracy_bar_chart" src="https://github.com/user-attachments/assets/1e7f9046-c1d5-4656-8f25-894d2a022cf3" />


FGSM attacks tested epsilon 0.0 to 0.3 in 0.05 steps on 1,000 samples. Figure 1 shows clean versus adversarial accuracy:

- At ε = 0.05: Attack succeeded 11.33%, adversarial accuracy 88.67%
- At ε = 0.1: Success 40.61%
- At ε = 0.15: Success 67.14%, only 32.86% classified correctly
- At ε = 0.3: Success 89.39%, only 10.61% correct

<img width="1486" height="884" alt="graph2_asr_bar_chart" src="https://github.com/user-attachments/assets/c0346e88-d2b8-430c-9355-d10157330c69" />


Figure 2 displays the attack success rates for different epsilon values, showing how effectiveness increases with perturbation strength.

#### Advanced Evaluation

<img width="1486" height="884" alt="graph3_epsilon_vs_accuracy" src="https://github.com/user-attachments/assets/a1143f3f-412d-4874-abf4-4176c3ca4036" />


Figure 3 shows the epsilon versus accuracy curve, which really illustrates the tradeoff. The green line for clean accuracy stays flat at 100% since I only test on correctly classified images. The red line for adversarial accuracy drops sharply as epsilon increases from 100% at ε = 0 down to about 60% at ε = 0.1, then around 20% at ε = 0.2, and bottoming out around 10% for ε = 0.3.

<img width="1486" height="884" alt="graph4_epsilon_vs_asr" src="https://github.com/user-attachments/assets/f4932cbe-b300-4960-a118-5428f63113f8" />


Figure 4 shows the epsilon versus ASR curve, which is basically the mirror image starting at 0% and climbing steadily, reaching nearly 90% by ε = 0.3.

**Confidence Analysis:**

At ε = 0.15, the average confidence for clean images was 98.4%, while after adding adversarial perturbations the average confidence dropped to 82.01%—a confidence drop of about 16.39%.

<img width="1486" height="884" alt="graph8_confidence_comparison" src="https://github.com/user-attachments/assets/3ce911d1-2679-4748-b4cc-fcf8fb544a46" />


Figure 8 shows the confidence distribution histogram where most clean predictions are clustered near 100% confidence, while for adversarial images the distribution spreads out more with a longer tail toward lower confidence values. The green dashed line shows average clean confidence and the red dashed line shows average adversarial confidence.

<img width="1718" height="1330" alt="graph5_image_grid" src="https://github.com/user-attachments/assets/704445e7-502e-4653-a36d-f310929dffc7" />

Figure 5 shows an image grid with original images in the top row, adversarial images in the middle, and magnified perturbations at the bottom. Adversarial images look identical but predictions are wrong: 2 becomes 3, 1 becomes 4, 0 becomes 9. Perturbations magnified 10× look like noise but are crafted to confuse the model.

<img width="450" height="180" alt="sample_predictions" src="https://github.com/user-attachments/assets/0f957ed5-afb3-433c-8718-461e84a050a5" />


Figure 9 shows additional test samples where you can see the true labels and predicted labels along with the images themselves.

<img width="2023" height="902" alt="graph7_confusion_matrices" src="https://github.com/user-attachments/assets/721cecd0-6283-4d5b-b082-73724f44a4f6" />


Figure 7 shows confusion matrices. The clean matrix is diagonal, while the adversarial matrix lights up everywhere. Some digits are more robust; 3 and 5 are confused often. Matrix element (i,j) represents:

```
number of images with true_label=i that were predicted as label=j
```

#### Interpretation

Results show neural networks are vulnerable to adversarial perturbations. The ResNet went from 98.41% to 32.86% accuracy with ε = 0.15. This vulnerability exists because networks learn complex boundaries in high-dimensional space that are not robust to crafted perturbations. FGSM works with a single gradient step, showing the fragility of these models.

For real-world digit recognition, attackers could fool systems with designed noise while humans read correctly. Confidence analysis shows the model is still confident when wrong (80%+ average). I cannot threshold on confidence to detect attacks. High clean accuracy does not equal robustness. Models need defenses like adversarial training.

---

## Challenges

### Limitations

Hardware limitations were the main challenge. I had an RTX 2050 GPU but limited RAM—the kernel crashed with GPU training and matplotlib. I solved this by:
- CPU training for less RAM usage
- Aggressive memory management with garbage collection
- Saving the model immediately after training
- Using Agg backend for matplotlib to save directly without displaying

### Epsilon Selection

The challenge was deciding epsilon values. Too small and the attack fails; too large and perturbations become obvious. I tested 0 to 0.3 with 0.05 increments for the effectiveness versus visibility tradeoff. I used ε = 0.15 as the sweet spot.

### Batch Processing for Attacks

Attacks require computing gradients per input and cannot be batch processed. I tested 1,000 images per epsilon instead of 10,000 for reliable statistics and manageable runtime.

---

## References

- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). *Explaining and harnessing adversarial examples*. arXiv.org. https://arxiv.org/abs/1412.6572

- Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., & Fergus, R. (2013). *Intriguing properties of neural networks*. arXiv.org. https://arxiv.org/abs/1312.6199

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep residual learning for image recognition*. arXiv.org. https://arxiv.org/abs/1512.03385

- Contributors, P. (2023). *CrossEntropyLoss*. https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

---

*Note: Images will be added to the figures referenced throughout this document.*

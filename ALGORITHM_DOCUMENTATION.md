# Brain Tumor Detection System Documentation

## 1. Project Overview

This project is a brain MRI image classification system. Its goal is to analyze an uploaded MRI scan and predict whether the image belongs to one of the following classes:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

The system uses deep learning with transfer learning. A pretrained `ResNet50` model is used as the base feature extractor, and custom classification layers are added on top to predict the final tumor class.

The web application is built with Flask and allows users to upload an MRI image, validate it, run prediction, and view a Grad-CAM explanation image.

## 2. Main Components

The project is mainly divided into these parts:

- [app.py](/D:/Sem%206%20prj/Brain%20Tumour%20Detection/Brain%20Tumour%20Detection/tumor/tumor/app.py): Flask web application for upload and prediction
- [main.py](/D:/Sem%206%20prj/Brain%20Tumour%20Detection/Brain%20Tumour%20Detection/tumor/tumor/main.py): model training script
- [debug_model.py](/D:/Sem%206%20prj/Brain%20Tumour%20Detection/Brain%20Tumour%20Detection/tumor/tumor/debug_model.py): evaluation and debugging script
- [gradcam.py](/D:/Sem%206%20prj/Brain%20Tumour%20Detection/Brain%20Tumour%20Detection/tumor/tumor/gradcam.py): Grad-CAM heatmap generation
- [mri_validator.py](/D:/Sem%206%20prj/Brain%20Tumour%20Detection/Brain%20Tumour%20Detection/tumor/tumor/mri_validator.py): image validation logic
- [prepare_dataset_splits.py](/D:/Sem%206%20prj/Brain%20Tumour%20Detection/Brain%20Tumour%20Detection/tumor/tumor/prepare_dataset_splits.py): safe dataset split preparation

## 3. Dataset Structure

The corrected dataset structure is:

```text
dataset/
  train/
    glioma/
    meningioma/
    notumor/
    pituitary/
  val/
    glioma/
    meningioma/
    notumor/
    pituitary/
  test/
    glioma/
    meningioma/
    notumor/
    pituitary/
```

### Why this structure is important

This structure prevents data leakage more effectively than randomly splitting the old `Training` folder during training. Validation and test sets should contain untouched original images only. Augmented images must never be placed inside validation or test folders.

## 4. Full System Workflow

The system works in two major phases:

1. Training phase
2. Prediction phase

---

## 5. Training Phase Step by Step

Training is handled in [main.py](/D:/Sem%206%20prj/Brain%20Tumour%20Detection/Brain%20Tumour%20Detection/tumor/tumor/main.py).

### Step 1: Load dataset

The script loads images from:

- `dataset/train`
- `dataset/val`
- `dataset/test`

Each folder contains four class subfolders. Keras automatically assigns labels based on folder names.

### Step 2: Resize images

All input images are resized to `224 x 224`.

This is necessary because `ResNet50` expects a fixed image size.

### Step 3: Apply data augmentation

During training, the model applies random transformations to training images:

- horizontal and vertical flip
- small rotation
- zoom
- brightness adjustment
- contrast adjustment

This helps the model generalize better instead of memorizing exact training images.

Important:

- augmentation is applied only during training
- validation and test images remain unchanged

### Step 4: Use transfer learning

The model uses `ResNet50` pretrained on ImageNet.

This means the model already knows how to detect useful visual patterns such as edges, shapes, and textures. Instead of training from zero, the project reuses that knowledge and adapts it to MRI classification.

### Step 5: Add custom classification layers

On top of the `ResNet50` base, custom layers are added:

- `GlobalAveragePooling2D`
- `Dropout(0.5)`
- `Dense(128, activation="relu", kernel_regularizer=l2(0.001))`
- `Dropout(0.5)`
- `Dense(4, activation="softmax")`

These layers convert extracted image features into probabilities for the four classes.

### Step 6: First training stage

At first, the `ResNet50` base model is frozen.

That means:

- pretrained convolution layers do not change
- only the new top layers are trained

This lets the classifier learn the task without damaging the pretrained knowledge too early.

### Step 7: Save initial weights

After the first training stage, weights are saved into:

- `initial_weights.weights.h5`

This is used before starting fine-tuning.

### Step 8: Fine-tuning stage

In the second training stage:

- part of `ResNet50` is unfrozen
- deeper layers are allowed to learn from MRI-specific patterns
- the learning rate is reduced

This helps the network adapt better to brain tumor images.

### Step 9: Use callbacks

Training uses callbacks for stability:

- `EarlyStopping`: stops training if validation accuracy stops improving
- `ModelCheckpoint`: saves the best model
- `ReduceLROnPlateau`: lowers learning rate when validation improvement slows down

### Step 10: Evaluate on test set

After training is complete, the final model is evaluated on `dataset/test`.

This gives the final test loss and accuracy.

The final saved model file is:

- `brain_tumor_model.keras`

---

## 6. Prediction Phase Step by Step

Prediction is handled in [app.py](/D:/Sem%206%20prj/Brain%20Tumour%20Detection/Brain%20Tumour%20Detection/tumor/tumor/app.py).

### Step 1: User uploads MRI image

The user opens the web application and uploads a brain MRI image.

### Step 2: File type checking

The app checks whether the uploaded file has an allowed image extension such as:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.webp`

### Step 3: Save uploaded file

The file is stored in the `static/uploads` folder with a timestamp-based name.

This avoids filename conflicts.

### Step 4: MRI validation

The app validates whether the image is likely to be a brain MRI.

Validation includes:

- a custom MRI validator from [mri_validator.py](/D:/Sem%206%20prj/Brain%20Tumour%20Detection/Brain%20Tumour%20Detection/tumor/tumor/mri_validator.py)
- grayscale-like image check
- edge structure check

These checks help reject invalid images like random photos or blank images.

### Step 5: Preprocess image

Before prediction:

- the image is read using OpenCV
- converted from BGR to RGB
- resized to `224 x 224`
- converted to `float32`
- expanded into batch format

### Step 6: Run model prediction

The model predicts four probabilities, one for each class:

- glioma
- meningioma
- notumor
- pituitary

Example:

```text
glioma      = 0.10
meningioma  = 0.18
notumor     = 0.07
pituitary   = 0.65
```

The largest probability normally becomes the predicted class.

### Step 7: Pituitary correction rule

The app contains a custom post-processing rule for pituitary detection.

Logic:

- if `pituitary_prob > 0.20`, predict `pituitary`
- else if `notumor_prob > 0.85` and `pituitary_prob < 0.15`, predict `notumor`
- otherwise use normal `argmax`

This means the final deployed system is not only deep learning. It is deep learning plus a manual rule.

### Step 8: Confidence handling

If the model confidence is below the threshold, the app shows a low-confidence warning message.

Current threshold:

- `0.60`

### Step 9: Grad-CAM explanation

The app creates a Grad-CAM heatmap to show which image region most influenced the prediction.

This gives visual interpretability and helps users understand where the model focused.

### Step 10: Show result

The app returns:

- uploaded image
- predicted class
- confidence percentage
- Grad-CAM image
- warning if confidence is low

---

## 7. Model Architecture Summary

The model architecture can be understood like this:

```text
Input MRI Image
   ->
Data Augmentation
   ->
ResNet50 Pretrained Base
   ->
Global Average Pooling
   ->
Dropout
   ->
Dense(128, ReLU, L2)
   ->
Dropout
   ->
Dense(4, Softmax)
   ->
Final Tumor Class
```

## 8. Why ResNet50 Is Used

`ResNet50` is used because:

- it is a strong pretrained CNN for image tasks
- it learns rich visual features
- it performs better than training a small network from scratch in many limited-data cases
- transfer learning reduces training time

## 9. Loss Function and Optimization

The project uses:

- loss: `sparse_categorical_crossentropy`
- optimizer: `Adam`

Why this is used:

- `sparse_categorical_crossentropy` is suitable because each image belongs to exactly one class
- `Adam` is commonly used because it trains efficiently and adapts learning rates automatically

## 10. What the Model Actually Learns

The model does not understand medical meaning like a doctor.

Instead, it learns image patterns such as:

- texture differences
- shape differences
- bright and dark region patterns
- structural abnormalities
- tumor region appearance

It converts those patterns into class probabilities.

## 11. Data Leakage Problem Found Earlier

During review, a data leakage risk was identified in the earlier workflow.

### Old problem

Previously:

- training and validation were being split from the old `dataset/Training` folder
- offline augmented files existed inside the dataset
- some augmented-style files were also present in testing

This could lead to unrealistic accuracy because train, validation, and test could contain highly similar images.

### Fix applied

The fix was:

1. create new explicit `train`, `val`, and `test` folders
2. remove offline augmented images from the rebuilt splits
3. deduplicate exact image matches
4. train only on `dataset/train`
5. validate only on `dataset/val`
6. evaluate only on `dataset/test`

This makes evaluation more trustworthy.

## 12. Role of `prepare_dataset_splits.py`

[prepare_dataset_splits.py](/D:/Sem%206%20prj/Brain%20Tumour%20Detection/Brain%20Tumour%20Detection/tumor/tumor/prepare_dataset_splits.py) was added to create safer dataset splits.

It performs these actions:

1. reads the old `Training` and `Testing` folders
2. ignores files that appear to be offline augmented
3. removes exact duplicate images using file hashes
4. rebuilds fresh `train`, `val`, and `test` folders
5. copies only cleaned original images into the new structure

## 13. Role of `debug_model.py`

[debug_model.py](/D:/Sem%206%20prj/Brain%20Tumour%20Detection/Brain%20Tumour%20Detection/tumor/tumor/debug_model.py) is used for diagnosis and analysis.

It helps with:

- class distribution check
- sample visualization
- confusion matrix generation
- classification report
- Grad-CAM on bad predictions
- train vs validation overfitting check

This script is useful to understand where the model is failing.

## 14. Grad-CAM Results Section

Grad-CAM results can absolutely be included in the project documentation because they help explain how the model made its decision.

### What Grad-CAM shows

Grad-CAM highlights the region of the MRI image that most influenced the model's prediction.

This is useful because:

- it makes the model more explainable
- it helps verify whether the model is focusing on the tumor region
- it can reveal if the model is focusing on the wrong area

### Where Grad-CAM results come from

Grad-CAM outputs are generated in two places:

- during web app prediction in [app.py](/D:/Sem%206%20prj/Brain%20Tumour%20Detection/Brain%20Tumour%20Detection/tumor/tumor/app.py)
- during debugging in [debug_model.py](/D:/Sem%206%20prj/Brain%20Tumour%20Detection/Brain%20Tumour%20Detection/tumor/tumor/debug_model.py)

### Current saved debug outputs

At the moment, the following result images are already available in [debug_outputs](/D:/Sem%206%20prj/Brain%20Tumour%20Detection/Brain%20Tumour%20Detection/tumor/tumor/debug_outputs):

- [class_samples.png](/D:/Sem%206%20prj/Brain%20Tumour%20Detection/Brain%20Tumour%20Detection/tumor/tumor/debug_outputs/class_samples.png)
- [confusion_matrix_current.png](/D:/Sem%206%20prj/Brain%20Tumour%20Detection/Brain%20Tumour%20Detection/tumor/tumor/debug_outputs/confusion_matrix_current.png)
- [distribution_training.png](/D:/Sem%206%20prj/Brain%20Tumour%20Detection/Brain%20Tumour%20Detection/tumor/tumor/debug_outputs/distribution_training.png)
- [distribution_testing.png](/D:/Sem%206%20prj/Brain%20Tumour%20Detection/Brain%20Tumour%20Detection/tumor/tumor/debug_outputs/distribution_testing.png)

### Current Grad-CAM status

No saved Grad-CAM result files are currently present in the `debug_outputs` folder.

That usually means one of these:

- the Grad-CAM generation step has not been run yet after the latest changes
- the model did not produce the target misclassifications used by `debug_model.py`
- the output was generated through the Flask app and saved in `static/uploads` instead

### How to describe Grad-CAM results in the report

You can add a result explanation like this:

> The Grad-CAM heatmap was used to visualize the region of the MRI scan that most influenced the model prediction. In correctly classified images, the highlighted region should overlap with the abnormal tumor area. If the heatmap focuses on unrelated regions, it may indicate weak model reasoning or dataset bias.

### Suggested result format

For each Grad-CAM example, document:

1. input MRI image
2. predicted class
3. confidence score
4. Grad-CAM heatmap
5. short interpretation of whether the highlighted area looks clinically meaningful

### Example interpretation

Example:

> In this sample, the model predicted `pituitary` with high confidence. The Grad-CAM heatmap highlighted the central abnormal region in the MRI, suggesting that the prediction was based on medically relevant image structure rather than background noise.

## 15. End-to-End Summary

The complete system works like this:

1. clean dataset is prepared
2. model is trained using transfer learning with `ResNet50`
3. trained model is saved
4. user uploads MRI image in Flask app
5. app validates the image
6. image is resized and converted
7. model predicts four class probabilities
8. custom pituitary rule adjusts output if needed
9. Grad-CAM explains the focus area
10. final result is shown to the user

## 16. Simple Explanation for Presentation

You can explain your project like this:

> Our system takes a brain MRI image and classifies it into glioma, meningioma, pituitary, or no tumor. We use a pretrained ResNet50 model with transfer learning, apply data augmentation during training, and use a softmax output layer for four-class prediction. The Flask app validates uploaded MRI images, performs prediction, and generates Grad-CAM heatmaps to visually explain the decision. We also improved the dataset pipeline to avoid data leakage by creating clean train, validation, and test splits.

## 17. Important Limitation

Even after fixing the dataset layout, one important medical limitation remains:

- the model is an image classifier, not a clinical diagnosis system

It should be treated as a decision-support or academic project, not as a replacement for a radiologist or doctor.

## 18. Next Recommended Steps

- repair the Python environment so training can run again
- retrain the model on the cleaned splits
- compare new train, validation, and test accuracy
- inspect confusion matrix after retraining
- if available, split data by patient identity instead of only image files

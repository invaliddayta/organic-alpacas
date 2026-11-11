# Phase I - Human only: How does the model behave?
Jupyter Notebooks (with additional utils.py files) hold all the important code and figures. Here we explain what we have found and what it means.
 
## Scope
We look at the model as a black box only. We look at inputs, predictions, and model outputs for the images. We do not look inside the model.
 
## Findings
**1. The mistakes go to similar diseases**
In the confusion matrix, almost everything sits on the diagonal. The few mistakes were diseases that look alike, not random mistakes. For example, tomato diseases are hard to get right. The model only mixes up diseases that are close.
 
**2. Class size has some influence, but is not the whole story**
Half of the classes have no errors at all. Some of those classes are big and some of them are small. The highest error rates are healthy potato and bacterial spot. The healthy potato class is small, so too few examples is a likely reason. Bacterial spot is large and still fails, so it causes far more wrong predictions. The class looks similar to other tomato diseases, so those classes are likely harder to predict. There are two different causes: too little data, and visual similarity.
 
**3. Noise is worse for the model than brightness or blur**
Adding noise to the input images decreases the performance drastically. Even small noise can lead to a complete loss of performance, even when humans can still see the image well. On the other side, brightness and blur lead to a performance loss much later. The model probably relies on fine details that get destroyed by noise, but stay with blur or brightness.
 
**4. Weak classes break faster**
The classes that perform worse under the corruption (noise, brightness or blur) are in general the classes with the lowest accuracy. The class with the worst accuracy (healthy potato) shows up when we look for the biggest performance decreases under the corruption. This class has very few images, so the model probably learned only a narrow idea of it, and small changes affect performance disproportionately. Overall accuracy hides this.
 
**5. The model does not cheat on the background**
We compare the original image, only the foreground, and only the background. The model should not be able to predict well with only the background, as the whole information is contained in the leaf in the foreground. The dataset still might include some clues in the background, e.g. some classes might have been photographed on a different table. The foreground performs much better, while the background only drops significantly in performance. The drop in accuracy from foreground only might come from the fact that the model has never seen a removed background in the training dataset.
 
## Summary
The model is accurate and looks at the right places. But noise breaks the model. Small class numbers are its weakness.
 
## Open questions
- What fine details does the model rely on? Does noise really destroy these details?
- The performance on the background only task was not 0. How does the model perform without any foreground information?

# File structure
| File | Description |
|------|-------------|
| `additional_experiments.ipynb` | Additional experiments in Google Colab |
| `experiments.ipynb` | Main findings for the phase 1 |
| `model_utils.py` | Helper functions for the model |
| `phase_one_utils.py` | Helper functions for `experiments.ipynb` |

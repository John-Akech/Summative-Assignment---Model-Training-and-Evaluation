## NEURAL NETWORK OPTIMIZATION AND MODEL COMPARISON

**Problem Statement**

Flooding in South Sudan has devastating consequences, leading to significant loss of lives, displacement, and economic damage. The country’s low-lying terrain and inadequate drainage infrastructure make it highly vulnerable to floods. Existing flood prediction systems often fall short due to:

- **Limited local data:** Insufficient region-specific datasets for accurate modeling.
- **Poor accuracy:** High false positive rates and unreliable predictions.
- **Lack of actionable warnings:** Inability to provide timely and actionable alerts to affected communities.

Machine Learning (ML) has shown promise in flood prediction, as demonstrated by studies such as Smith et al. (2020), Adebayo et al. (2019), and Zhou et al. (2021). However, challenges remain, including:

- **Data scarcity:** Limited availability of high-quality, region-specific data.
- **Model generalization:** Difficulty in achieving robust performance across diverse scenarios.
- **Real-time integration:** Lack of systems capable of providing real-time predictions.

This project, **FloodSense South Sudan**, addresses these gaps by developing a tailored ML-based flood prediction and early warning system. The system leverages historical flood data and meteorological records to create a localized solution for South Sudan. The primary goal is to optimize a Neural Network and compare its performance against traditional ML algorithms. The study involves experimenting with various hyperparameter configurations to identify the best-performing model for a multi-class classification task. Techniques such as optimizer selection, regularization, and early stopping were applied to enhance model performance.

**Dataset Acquisition**

The project utilizes the Kaggle dataset for rainfall statistics and historical flood records. This dataset provides critical inputs for training and evaluating the machine learning models.

**Key Objectives**

- Develop a localized flood prediction system.
- Optimize machine learning models to improve accuracy, reduce false positives, and enhance generalization.
- Compare the performance of Neural Networks with traditional ML algorithms such as Random Forest, XGBoost, and Logistic Regression.

**Approach**

The project involves:

- **Data preprocessing:** Cleaning, normalizing, and preparing the dataset for modeling.
- **Model optimization:** Experimenting with hyperparameters such as optimizers, regularization methods, and learning rates.
- **Performance evaluation:** Comparing models based on metrics like accuracy, F1 score, precision, recall, and ROC AUC.

**Expected Outcomes**

- A highly accurate flood prediction model.
- Insights into the effectiveness of Neural Networks versus traditional ML algorithms for flood prediction.

**Findings**

Below is a summary of different training instances with their respective configurations and results:

| Training Instance | Optimizer | Regularizer (L1/L2) | Epochs | Early Stopping | Layers | Learning Rate | Accuracy | F1 Score | Precision | Recall | ROC AUC Score |
|-------------------|-----------|---------------------|--------|----------------|--------|---------------|----------|----------|-----------|--------|---------------|
| Instance 1        | Default   | None                | 50     | No             | 3      | Default       | 0.9736   | 0.9595   | 0.9816    | 0.9383 | 0.9951        |
| Instance 2        | Adam      | L2 (0.01)           | 100    | Yes            | 4      | 0.001         | 0.9795   | 0.9685   | 0.9908    | 0.9471 | 0.9983        |
| Instance 3        | RMSProp   | L1 (0.001)          | 120    | Yes            | 5      | 0.0005        | 0.9663   | 0.9483   | 0.9679    | 0.9295 | 0.9952        |
| Instance 4        | Adam      | L2 (0.005)          | 80     | No             | 4      | 0.001         | 0.9677   | 0.9500   | 0.9812    | 0.9207 | 0.9958        |
| Instance 5        | Adam      | L1 & L2 (0.001)     | 150    | Yes            | 6      | 0.0001        | 0.9809   | 0.9713   | 0.9735    | 0.9692 | 0.9984        |

**Summary**

The best performance was achieved in Instance 5, which used:

- **Optimizer:** Adam
- **Regularization:** L1 & L2 (0.001)
- **Early Stopping:** Enabled
- **Learning Rate:** 0.0001
- **Layers:** 6

This configuration yielded the highest accuracy (0.9809) and the best F1-score (0.9713), demonstrating a well-balanced model with strong generalization capabilities.

**ML Algorithm vs. Neural Network**

The optimized Neural Network was compared with traditional ML models:

| Model                    | Accuracy | F1 Score | Precision | Recall | ROC AUC |
|--------------------------|----------|----------|-----------|--------|---------|
| Random Forest            | 0.9266   | 0.8837   | 0.9359    | 0.8370 | 0.9831  |
| XGBoost                  | 0.9428   | 0.9099   | 0.9563    | 0.8678 | 0.9872  |
| Logistic Regression      | 0.9897   | 0.9843   | 1.0000    | 0.9691 | 0.9997  |
| Best Neural Network      | 0.9809   | 0.9713   | 0.9734    | 0.9691 | 0.9984  |

Surprisingly, Logistic Regression outperformed the Neural Network, achieving the highest accuracy and ROC AUC score. This suggests that for this dataset, a simpler model with well-tuned hyperparameters (e.g., regularization strength, solver type) may generalize better than a deep neural network.

**Discussion**

**Key Insights**

- **Regularization:** Using both L1 and L2 regularization improved model generalization. The best-performing instance incorporated both L1 and L2 regularization.
  
- **Early Stopping:** Helped prevent overfitting and improved model performance in most cases.

- **Learning Rate:** A smaller learning rate (0.0001) facilitated better model convergence and improved accuracy.

- **Optimizers:** The Adam optimizer consistently outperformed other optimizers.

- **Model Complexity:** Despite extensive optimization, Logistic Regression outperformed the Neural Network, highlighting the effectiveness of simpler models under the right conditions.

**Why Did Logistic Regression Outperform?**

- **Data Characteristics:** The dataset may not have complex patterns that require deep learning.

- **Overfitting:** The Neural Network may have overfit the training data despite regularization and early stopping.

- **Hyperparameter Tuning:** Logistic Regression may have been better tuned for this specific dataset.

Feel free to explore my work and provide feedback for further improvements!

Below is the link to the recorded video:  
[Recorded Video](https://drive.google.com/file/d/1OfDelEjUxitLQZGbWmjUjrDGBsa6s5tZ/view?usp=sharing)

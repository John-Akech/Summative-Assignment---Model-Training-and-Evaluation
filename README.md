# Neural Network Optimization and Model Comparison

## Problem Statement

This project aims to optimize a Neural Network and compare its performance against traditional Machine Learning algorithms. The goal is to identify the best-performing model for a multi-class classification task by experimenting with different hyperparameter configurations. Various optimization techniques, including optimizers, regularization methods, and early stopping mechanisms, were tested to enhance performance.

## Findings

Below is a summary of different training instances with their respective configurations and results:

| Training Instance | Optimizer | Regularizer (L1/L2) | Epochs | Early Stopping | Layers | Learning Rate | Accuracy | Loss | F1 Score | Precision | Recall | ROC AUC Score |
|-------------------|-----------|---------------------|--------|----------------|--------|---------------|----------|------|----------|-----------|--------|---------------|
| Instance 1        | Default   | None                | 50     | No             | 3      | Default       | 0.9736   | --   | 0.9595   | 0.9816    | 0.9383 | 0.9951        |
| Instance 2        | Adam      | L2 (0.01)           | 100    | Yes            | 4      | 0.001         | 0.9795   | --   | 0.9685   | 0.9908    | 0.9471 | 0.9983        |
| Instance 3        | RMSProp   | L1 (0.001)          | 120    | Yes            | 5      | 0.0005        | 0.9663   | --   | 0.9483   | 0.9679    | 0.9295 | 0.9952        |
| Instance 4        | Adam      | L2 (0.005)          | 80     | No             | 4      | 0.001         | 0.9677   | --   | 0.9500   | 0.9812    | 0.9207 | 0.9958        |
| Instance 5        | Adam      | L1 & L2 (0.001)     | 150    | Yes            | 6      | 0.0001        | 0.9809   | --   | 0.9713   | 0.9735    | 0.9692 | 0.9984        |

## Summary

The best performance was achieved in **Instance 5**, which used the **Adam optimizer with both L1 and L2 regularization, early stopping, and a learning rate of 0.0001**. This configuration achieved the highest accuracy (0.9809) and the best F1-score (0.9713), indicating a well-balanced model.

## ML Algorithm vs. Neural Network

The performance of the optimized Neural Network was also compared to traditional Machine Learning models:

| Model                   | Accuracy     | F1 Score     | Precision     | Recall      | ROC AUC      |
|-------------------------|--------------|--------------|---------------|-------------|--------------|
| Random Forest           | 0.9266       | 0.8837       | 0.9359        | 0.8370      | 0.9831       |
| XGBoost                 | 0.9428       | 0.9099       | 0.9563        | 0.8678      | 0.9872       |
| Logistic Regression     | 0.9897       | 0.9843       | 1.0000        | 0.9691      | 0.9997       |
| **Best Neural Network** | **0.9809**   | **0.9713**   | **0.9734**    | **0.9691**  | **0.9984**   |

Interestingly, **Logistic Regression outperformed the Neural Network**, achieving the highest accuracy and ROC AUC score. This suggests that for this dataset, a simpler model with well-tuned hyperparameters (e.g., regularization strength, solver type) may generalize better than a deep neural network.

## Discussion

The results highlight several key insights:

1. **Regularization**: Adding L1 and L2 regularization improved model generalization, with the best-performing instance using both.
2. **Early Stopping**: Helped prevent overfitting and contributed to better performance in most cases.
3. **Learning Rate**: A smaller learning rate (0.0001) resulted in better model convergence and higher accuracy.
4. **Optimizers**: Adam performed consistently well compared to other optimizers.

Despite optimizing hyperparameters, **Logistic Regression surprisingly outperformed the Neural Network**, suggesting that simpler models can be highly effective given the right conditions.

## Next Steps

- **Further optimize** the Neural Network by fine-tuning dropout rates and batch normalization.
- **Experiment with ensemble techniques**, combining Neural Networks with traditional ML models.
- **Explore alternative architectures** such as CNNs or Transformers if applicable.
- **Deploy the best model** and create an API for real-world predictions.

## Submission Details

The project includes:

- **Complete code** for training and evaluating the models.
- **A 5-minute video** explaining the implementation, hyperparameters, and findings.

Feel free to explore my work and suggest improvements!


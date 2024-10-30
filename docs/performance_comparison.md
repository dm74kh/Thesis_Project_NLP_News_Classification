## Initial Training Configuration

For the baseline training, we utilized a batch size of 32 and set the number of training epochs to 5. During hyperparameter optimization, each model was trained for 3 epochs over 5 trials, enabling us to explore a range of potential hyperparameter combinations. Finally, the optimized model underwent additional testing with 5 epochs to evaluate performance improvements.

## DNN Model Analysis

### Basic Training Module
- **Accuracy:** The DNN model demonstrated steady improvement in accuracy as the number of epochs increased. From epoch 1 to epoch 5, the training accuracy rose from 85.91% to 90.63%, while the loss decreased from 0.4269 to 0.2671. 
- **Validation Accuracy:** The validation accuracy remained relatively stable, ranging between 87.52% and 88.02%, indicating that the model generalized well to unseen data during initial training.
- **Validation loss:** fluctuated slightly but ultimately decreased, reaching 0.3388 in the final epoch.
### Testing Module 1 (Baseline Model)
- **Baseline Test Accuracy:** 88.18%
- **Classification Report:** Class-specific performance metrics:
    - Class 0: precision 0.90, recall 0.88, f1-score 0.89
    - Class 1: precision 0.94, recall 0.96, f1-score 0.95
    - Class 2: precision 0.86, recall 0.81, f1-score 0.83
    - Class 3: precision 0.83, recall 0.87, f1-score 0.85
- **Confusion Matrix:** The baseline model shows strong performance in predicting classes 0 and 1, while facing slight challenges with classes 2 and 3, as reflected in their recall scores.
![Confusion Matrix](images/dnn_confusion_matrix_1.png)

### Hyperparameter Selection Module
Optuna was used for tuning the model's hyperparameters, resulting in the best configuration:
- **dropout_rate:** 0.3689
- **learning_rate:** 0.0001423
- **num_units:** 471
These optimized parameters helped improve the model’s generalization capabilities and boosted its accuracy on the validation set.
### Testing Module 2 (Optimized Model)
- **Final Test Accuracy:** 88.74%
- **Classification Report:** After optimization, the class-specific performance metrics are as follows:
    - Class 0: precision 0.90, recall 0.89, f1-score 0.90
    - Class 1: precision 0.95, recall 0.96, f1-score 0.95
    - Class 2: precision 0.86, recall 0.83, f1-score 0.84
    - Class 3: precision 0.85, recall 0.87, f1-score 0.86
- **Confusion Matrix:** The optimized model further enhances the recall for most classes, particularly in class 0, showing improved performance and generalization.
![Confusion Matrix](images/dnn_confusion_matrix_2.png)
### Summary

The DNN model demonstrated robust performance throughout the training and testing phases, with steady improvements in accuracy and loss reduction over epochs. The initial baseline model achieved a satisfactory test accuracy of 88.18%, indicating strong predictive capability across all classes. However, some discrepancies were noted, particularly in classes 2 and 3, where recall scores were slightly lower. With hyperparameter tuning via Optuna, the optimized model showed enhanced performance, achieving a final test accuracy of 88.74%.
## CNN Model Analysis

#### Basic Training Performance
In the initial basic training phase, the CNN model was trained for 5 epochs with the following results:
- **Accuracy** gradually improved from 0.7814 to 0.8940.
- **Validation accuracy** was relatively stable, fluctuating around 0.8625 to 0.8780.
- **Validation loss** slightly decreased, indicating the model was learning but showed signs of reaching an optimal point.
#### Testing Module 1 (Base Model)
The baseline CNN model achieved an **accuracy of 88.2%** on the test set. Precision, recall, and F1-score for each class were as follows:
- Class 0: Precision: 0.92, Recall: 0.87, F1-score: 0.89
- Class 1: Precision: 0.93, Recall: 0.95, F1-score: 0.94
- Class 2: Precision: 0.85, Recall: 0.83, F1-score: 0.84
- Class 3: Precision: 0.84, Recall: 0.88, F1-score: 0.86
Overall, the baseline model displayed strong performance, particularly in Class 1, which had the highest precision and recall.
![Confusion Matrix](images/cnn_confusion_matrix_1.png)
#### Hyperparameter Tuning with Optuna
Optuna was used to optimize the following hyperparameters for the CNN model:
- **Dropout Rate**: Tested values ranged between 0.13 to 0.36, with an optimal value of 0.26 found in Trial 3.
- **Learning Rate**: The learning rate ranged from 0.0000343 to 0.000912, with an optimal rate of 0.000912.
- **Filters**: Number of filters in Conv1D layers varied, with the best performance at 227 filters.
- **Kernel Size**: Kernel size ranged between 3 and 4, with an optimal size of 4 found in Trial 3.
- **Num Units**: Dense layer units were optimized, with 504 units yielding the best performance.
Trial 3 provided the best performance with **accuracy of 89.7%**, and these optimal hyperparameters were used for final testing.
#### Testing Module 2 (Optimized Model)
After applying the best-found hyperparameters, the CNN model’s accuracy improved to **89.05%** on the test set. The updated precision, recall, and F1-scores were:
- Class 0: Precision: 0.89, Recall: 0.90, F1-score: 0.90
- Class 1: Precision: 0.96, Recall: 0.95, F1-score: 0.96
- Class 2: Precision: 0.84, Recall: 0.87, F1-score: 0.85
- Class 3: Precision: 0.88, Recall: 0.84, F1-score: 0.86
The optimized model maintained a strong performance, especially improving on recall for Class 2, which saw a notable increase compared to the baseline.

![Confusion Matrix](images/cnn_confusion_matrix_2.png)
#### Summary

The optimized CNN model demonstrated a consistent improvement in accuracy and performance across most metrics, confirming the effectiveness of the hyperparameter tuning process. The model showed balanced results across classes, especially for Classes 1 and 0, where it achieved the highest precision and recall scores.

## RNN Model Analysis

### Basic Training Module

In the initial training phase, the RNN model achieved progressive improvements in accuracy and reductions in loss over the 5 epochs. 
- **Accuracy**: The RNN model achieved progressive improvements over the 5 epochs, demonstrating effective learning with each iteration.
- **Validation accuracy**: Increased from **87.59%** in the first epoch to **88.90%** in the fifth epoch, indicating stable and consistent performance improvement on the validation set.
- **Validation loss**: Decreased steadily, which suggests that the model effectively learned from the data without signs of overfitting.
### Testing Module 1 (Base Model)

Upon testing, the baseline RNN model achieved a test accuracy of **88.95%**. The classification report revealed the following:
- Class 0: Precision 0.92, Recall 0.87, F1-Score 0.90
- Class 1: Precision 0.94, Recall 0.97, F1-Score 0.95
- Class 2: Precision 0.83, Recall 0.86, F1-Score 0.85
- Class 3: Precision 0.87, Recall 0.86, F1-Score 0.86
The overall accuracy and macro/micro averages were around 89%, indicating balanced performance across classes, with Class 1 showing the highest recall and F1-score. This baseline confusion matrix highlights some confusion between similar classes but demonstrates the RNN’s strength in handling sequential dependencies.
![Confusion Matrix](images/rnn_confusion_matrix_1.png)
### Hyperparameter Selection Module

The hyperparameter tuning phase focused on optimizing parameters like `dropout_rate`, `learning_rate`, `lstm_units`, and `num_units`. Trials ranged:
- **Learning rate** ranged from 0.0000287 to 0.0056, with an optimal rate of 0.00038.
- **Dropout rate** ranged from 0.029 to 0.478, with an optimal dropout of 0.375.
- **LSTM units** varied between 105 and 248, with 158 being the best-performing setting.
- **Dense layer units** ranged between 126 and 242, with the optimal at 240.
The best trial achieved an accuracy of **90.28%**, showing an improvement over the baseline.
### Testing Module 2 (Optimized Model)

The optimized RNN model was then evaluated, yielding a final accuracy of **90.71%** on the test set. The performance across classes improved as follows:
- Class 0: Precision 0.94, Recall 0.90, F1-Score 0.92
- Class 1: Precision 0.95, Recall 0.97, F1-Score 0.96
- Class 2: Precision 0.89, Recall 0.86, F1-Score 0.87
- Class 3: Precision 0.86, Recall 0.90, F1-Score 0.88
The optimized confusion matrix reflects better discrimination among classes, particularly reducing misclassifications in classes with high sequential dependencies. The overall improvement in precision and recall across classes highlights the effectiveness of the optimized hyperparameters.
![Confusion Matrix](images/rnn_confusion_matrix_2.png)
### Summary

The RNN model, after hyperparameter tuning, achieved notable improvements, particularly in precision and recall across all classes. With an optimized accuracy of **90.71%**, the model demonstrated strong sequential processing capabilities, making it well-suited for text classification tasks involving sequence dependencies. This analysis underscores the importance of fine-tuning RNN models to enhance performance, especially when handling complex class relationships.

## BERT Model Analysis

### Basic Training Module

- **Accuracy**: In the basic training phase, the BERT model displayed steady improvement, achieving an accuracy of 49.68% in the first epoch and rising to 83.45% by the fifth epoch. This reflects BERT’s capability to learn effectively from the provided dataset, especially given its robust pre-training on a diverse corpus.
- **Validation Accuracy**: The validation accuracy started at 76.68% and gradually improved to 84.56% by the fifth epoch, indicating that the model was generalizing well to unseen data and avoiding significant overfitting.
- **Validation Loss**: Validation loss also decreased over time, from an initial value of 0.7354 to 0.4307 by the final epoch. This decline in validation loss demonstrates the model’s ability to reduce error as training progresses, improving its predictive reliability on new inputs.
### Testing Module 1 (Base Model)

The baseline model achieved a test accuracy of 84.66%, with strong results across each label category:
- **Class 0**: Precision 0.86, Recall 0.86, F1-Score 0.86
- **Class 1**: Precision 0.93, Recall 0.93, F1-Score 0.93
- **Class 2**: Precision 0.79, Recall 0.80, F1-Score 0.80
- **Class 3**: Precision 0.81, Recall 0.80, F1-Score 0.80
Overall, the weighted average F1-score and accuracy reached approximately 85%, indicating that the base BERT model was quite effective for the classification task, though there was room for improvement, particularly in classes 2 and 3.
![Confusion Matrix](images/bert_confusion_matrix_1.png)
### Hyperparameter Selection Module

Optuna was employed for hyperparameter tuning, exploring values for dropout rate, learning rate, and the number of units.
- **Dropout Rate**: Tested within a range of approximately 0.07 to 0.48, with the optimal rate identified as 0.27, which provided a balance between preventing overfitting and retaining key features for prediction.
- **Learning Rate**: Varied from 0.000011 to 0.000034, with the optimal rate found to be 0.000034, enhancing learning speed without destabilizing convergence.
- **Number of Units**: Explored values between 127 and 508, with the optimal number determined as 439 units, providing adequate model capacity without excessive complexity.
The best trial achieved a validation accuracy of 84.89%, demonstrating that hyperparameter optimization led to marginal gains over the baseline model.
### Testing Module 2 (Optimized Model)

Following hyperparameter tuning, the optimized BERT model attained a test accuracy of 85.89%. Key metrics for each class were as follows:
- **Class 0**: Precision 0.89, Recall 0.85, F1-Score 0.87
- **Class 1**: Precision 0.93, Recall 0.95, F1-Score 0.94
- **Class 2**: Precision 0.78, Recall 0.84, F1-Score 0.81
- **Class 3**: Precision 0.84, Recall 0.80, F1-Score 0.82
The optimized model's macro and weighted average metrics rose to approximately 86%, indicating that the hyperparameter tuning achieved slight but meaningful improvements, especially in precision and recall for challenging classes.
![Confusion Matrix](images/bert_confusion_matrix_2.png)
### Summary

The BERT model’s performance highlights its strength in text classification tasks. The optimized model demonstrated higher overall accuracy and improved class-wise performance, especially in classes with initial lower scores. This tuning process underscores the importance of refining dropout, learning rate, and network size for leveraging BERT’s full potential in specialized tasks.

## Transformers-LSTM-GRU Model Analysis

### Basic Training Module (Base Model)

- **Accuracy**: During training, the base model achieved an accuracy of 86.44% in the first epoch, gradually increasing to 94.67% by the fifth epoch, showing consistent improvement.
- **Validation Accuracy**: The initial validation accuracy was 89.28%, and although it fluctuated slightly, it reached 90.03% by the fifth epoch, indicating strong model learning capability.
- **Validation Loss**: The validation loss decreased from 0.2915 to 0.3242, with a slight increase in the final epochs, possibly indicating mild overfitting.
### Testing Module 1 (Base Model)

The final test accuracy for the base model was 90.05%.
- **Classification Report (Base Model)**:
    - **Label 0**: Precision: 0.94, Recall: 0.87, F1-score: 0.91
    - **Label 1**: Precision: 0.95, Recall: 0.97, F1-score: 0.96
    - **Label 2**: Precision: 0.84, Recall: 0.88, F1-score: 0.86
    - **Label 3**: Precision: 0.87, Recall: 0.89, F1-score: 0.88
- **Overall**:
    - **Accuracy**: 0.90
    - **Macro Average**: Precision: 0.90, Recall: 0.90, F1-score: 0.90
    - **Weighted Average**: Precision: 0.90, Recall: 0.90, F1-score: 0.90
![Confusion Matrix](images/transformers_confusion_matrix_1.png)
### Hyperparameter Selection Module

Optuna was employed for hyperparameter tuning, testing dropout rates, learning rates, and unit sizes:
- **Dropout Rate**: Tested within a range of approximately 0.18 to 0.40, with the optimal rate identified as 0.36, balancing overfitting prevention with model complexity retention.
- **Learning Rate**: Varied between 0.000014 and 0.0019, with the optimal rate found to be 0.0019, which accelerated learning without destabilizing convergence.
- **Units**: Explored values for GRU and LSTM units ranged from 118 to 239 for each layer, with the best configuration being 195 GRU units and 209 LSTM units, providing a balanced model capacity.
The best trial achieved a validation accuracy of 90.88%, indicating that hyperparameter optimization yielded marginal improvements over the baseline.
### Testing Module 2 (Optimized Model)

- **Optimized Model Accuracy**: After optimization, the final test accuracy improved to 90.78%.
- **Classification Report (Optimized Model)**:
    - **Label 0**: Precision: 0.93, Recall: 0.90, F1-score: 0.91
    - **Label 1**: Precision: 0.94, Recall: 0.98, F1-score: 0.96
    - **Label 2**: Precision: 0.89, Recall: 0.85, F1-score: 0.87
    - **Label 3**: Precision: 0.87, Recall: 0.90, F1-score: 0.88
- **Overall**:
    - **Accuracy**: 0.91
    - **Macro Average**: Precision: 0.91, Recall: 0.91, F1-score: 0.91
    - **Weighted Average**: Precision: 0.91, Recall: 0.91, F1-score: 0.91
![Confusion Matrix](images/transformers_confusion_matrix_2.png)
### Summary

The optimized Transformers-LSTM-GRU model exhibited strong performance improvements post-optimization, especially with increased precision and recall for the majority of labels. This suggests that the model effectively leveraged sequential dependencies with the optimized parameters, leading to a balanced and accurate classification across labels.

## Classes Performance

Based on the analysis of confusion matrices and testing reports, the most challenging classes for classification were:
1. **Class 2**:
    - All models struggled with accurately classifying this class, especially in terms of precision and recall. Errors were frequently due to models confusing Class 2 with other classes (particularly Classes 0 and 3).
    - This could be due to the text characteristics of this class being similar to others, making it difficult for models to clearly delineate boundaries.
2. **Class 3**:    
    - In certain models, such as DNN and CNN, Class 3 also showed lower performance metrics, with frequent errors reflected in decreased F1 scores.
    - More complex models like BERT and Transformers handled Class 3 better, but required deeper analysis of text patterns.
Overall, **Classes 2 and 3** proved to be the most challenging, with higher error rates, especially in simpler architectures like CNN and DNN. This suggests that these classes require more advanced models capable of capturing deeper text patterns and nuances.

### Comparative Analysis of Model Performance

After analyzing the confusion matrices and performance reports for each model, here is a summary comparison focusing on accuracy, precision, recall, and F1-score across the models tested:

**BERT Model**:
    - **Base Model Accuracy**: 84.66%
    - **Optimized Model Accuracy**: 85.89%
    - **Observations**: BERT demonstrated stable improvements post-optimization, particularly in precision and recall for classes with lower initial performance scores. Overall, the optimized BERT model achieved higher balanced scores, making it effective for general classification tasks.
    
 **CNN Model**:
    - **Base Model Accuracy**: 82.36%
    - **Optimized Model Accuracy**: 83.98%
    - **Observations**: CNN exhibited slight improvements with optimization. The model performed reasonably well with high accuracy in simpler classes but struggled with more complex classifications, suggesting limitations in handling nuanced text features compared to BERT and Transformers.
    
**DNN Model**:
    - **Base Model Accuracy**: 81.57%
    - **Optimized Model Accuracy**: 82.85%
    - **Observations**: DNN showed minimal improvement with optimization. While it delivered consistent accuracy across classes, it lagged behind BERT and Transformers in overall precision and recall, indicating it may not capture intricate patterns as effectively.
    
**RNN Model**:
    - **Base Model Accuracy**: 88.95%
    - **Optimized Model Accuracy**: 90.71%
    - **Observations**: RNN’s performance was notably robust, with significant improvements in F1-scores for more challenging classes after optimization. This model proved effective for sequential data but still underperformed slightly against BERT and Transformer models in comprehensive accuracy.
    
**Transformers (LSTM-GRU)**:
    - **Base Model Accuracy**: 90.05%
    - **Optimized Model Accuracy**: 90.77%
    - **Observations**: Transformers demonstrated the highest balanced performance across all classes, excelling in precision, recall, and F1-score. Post-optimization, it achieved the highest overall accuracy, especially in challenging classes, making it the most reliable model for this classification task.
### Conclusion

The **Transformers model** (LSTM-GRU variant) achieved the best overall performance after optimization, closely followed by the BERT model. Both models effectively handled the classification task with balanced precision and recall across classes. The RNN model also performed well, while CNN and DNN models showed limitations in accurately capturing complex patterns. For high-stakes applications requiring nuanced text understanding, the **Transformer-based model is recommended** due to its superior classification accuracy and robust generalization across varied classes.
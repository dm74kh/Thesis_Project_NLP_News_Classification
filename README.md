# Diploma

# Project Title

## Overview
This project focuses on the analysis, training, and evaluation of multiple machine learning models on a text classification task. The aim is to explore different model architectures and compare their performance on the AG News dataset.

## Directory Structure
The repository is organized into the following main folders:

- **analysis**: Contains exploratory data analysis files for the dataset.
  - `exploratory_analysis.ipynb`: Notebook detailing data preprocessing, visualization, and statistical analysis of the dataset.

- **data**: Contains training and testing datasets.
  - `agn_train.csv`: Training dataset.
  - `agn_test.csv`: Testing dataset.

- **docs**: Documentation and theoretical background information.
  - `aims_and_framework.md`: Overview of the project objectives and theoretical framework.
  - `models_overview.md`: Detailed description of model architectures used in the project.
  - `performance_comparison.md`: Summary and analysis of model performance across various metrics.
  - `theoretical_background.ipynb`: Notebook covering theoretical aspects relevant to the models and data processing.

- **models**: Contains Jupyter Notebooks with code for each model implemented in the project.
  - `BERT_model.ipynb`: Implementation of the BERT model.
  - `CNN_model.ipynb`: Implementation of the CNN model.
  - `DNN_model.ipynb`: Implementation of the DNN model.
  - `RNN_model.ipynb`: Implementation of the RNN model.
  - `Transformers-LSTM-GRU_model.ipynb`: Implementation of a combined Transformers-LSTM-GRU model.

- **results**: Results and analysis of model performance.
  - `BERT_model.ipynb`, `CNN_model.ipynb`, etc.: Executed notebooks for each model, showing results and evaluation metrics.
  - `images/`: Contains images such as confusion matrices to visually represent model performance.
  - `performance_comparison.md`: Analysis and performance comparison of all models.

## Requirements
To run the notebooks, ensure you have the following packages installed:
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- tensorflow/torch (depending on models used)
- transformers (for BERT and related models)

## Usage

1. **Data Preparation**: Place your dataset files in the `data` folder.
2. **Exploratory Analysis**: Open and run the `exploratory_analysis.ipynb` notebook in the `analysis` folder to explore and preprocess the dataset.
3. **Model Training**: Use the notebooks in the `models` folder to train each model.
4. **Evaluation and Comparison**: Run the notebooks in the `results` folder to see each model's performance metrics. For a comparative summary, refer to `performance_comparison.md` in the `docs` folder.

## Results

The performance of each model is documented in the `performance_comparison.md` file in the `docs` folder. Additionally, confusion matrices and other visualizations are stored in the `results/images` folder for further analysis.

## License

CC BY-NC-SA 4.0

## Contact

Dmytro Mykhailychenko
The National Technical University "Kharkiv Polytechnic Institute"
E-mail: dmytro.mykhailychenko@cs.khpi.edu.ua
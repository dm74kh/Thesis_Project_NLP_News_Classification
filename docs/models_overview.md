## Model code structure

The model code is structured into the following blocks: **Import libraries**, **Preprocessing module**, **Basic training module**, **Testing Module 1 (Base Model)**, **Hyperparameter selection module**, and **Testing module 2 (Optimized Model)**.

This structure is chosen to streamline the workflow of model development, training, and evaluation, ensuring clarity and modularity. Each block has a specific function:

1. **Import libraries**: Includes essential libraries such as `TensorFlow`, `Keras`, `Optuna` for hyperparameter optimization, `sklearn` for performance evaluation, and libraries for data processing and visualization.
    
2. **Preprocessing module**: Loads, cleans, and prepares the text data, handling steps such as tokenization, stop-word removal, and embedding generation, which are essential for text data to be effectively utilized by neural networks.
    
3. **Basic training module**: Provides initial model training on the base architecture without optimization, allowing a baseline performance assessment.
    
4. **Testing Module 1 (Base Model)**: Tests the initial model on a validation set to evaluate its baseline accuracy and other metrics.
    
5. **Hyperparameter selection module**: Utilizes Optuna to find optimal hyperparameters (e.g., learning rate, dropout rate, number of units) for enhancing model performance.
    
6. **Testing Module 2 (Optimized Model)**: Re-trains and re-evaluates the model with the optimized hyperparameters, assessing performance improvements and comparing it to the base model.
    
This structure allows for clear, sequential testing and comparison of different model architectures and configurations, enabling efficient performance assessment and refinement of each model.

## Preprocessing Module

### Preprocessing Module for DNN, CNN, RNN, and Transformers-LSTM-GRU Models

The preprocessing module prepares text data for model input by following several key steps. Each step ensures that the input is clean, standardized, and compatible with deep learning models:

1. **Data Loading**:
   - Loads training and test datasets from CSV files hosted online, using `pd.read_csv`. Each dataset contains text fields "Title" and "Description," which are then combined to form a single "clean_text" column for analysis. This combined text captures more context for each data point.

2. **Text Cleaning and Normalization**:
   - Converts text to lowercase to ensure case consistency.
   - Removes specific patterns such as HTML tags, HTML entities, punctuation, and other non-alphabetical characters. This cleaning makes the text uniform and removes irrelevant noise that could affect model performance.
   - Normalizes specific terms (e.g., "U.S." to "USA") to reduce variability in expressions.

3. **Tokenization and Lemmatization**:
   - Splits the cleaned text into individual words (tokenization) and removes stop words that do not add semantic value (e.g., "and," "the").
   - Applies lemmatization to reduce words to their root forms (e.g., "running" to "run"), ensuring that different forms of a word are recognized as the same, further simplifying the vocabulary.

4. **Token Conversion**:
   - Uses the `Tokenizer` class from Keras to convert words into integer sequences. This step replaces each word in the text with a unique integer ID based on a frequency cap of 5000 words, creating manageable and efficient inputs for the model.
   
5. **Sequence Padding**:
   - Pads all sequences to a fixed length, which matches the longest sequence in the data, ensuring that input sequences are of uniform length. This uniformity is essential for batch processing in deep learning models.

6. **Embedding Preparation**:
   - Downloads and loads pre-trained GloVe embeddings, which provide dense vector representations for words. These embeddings capture semantic relationships between words and improve model performance by offering a richer representation of the text.
   - Constructs an `embedding_matrix`, mapping each word in the vocabulary to its GloVe vector. This matrix is later passed to the embedding layer in the model to initialize it with pre-trained values.

This preprocessing setup is crucial for preparing the text data uniformly across models (DNN, CNN, RNN, and Transformers-LSTM-GRU) and provides a strong foundation for model training by ensuring that each model receives standardized, semantically rich input data.

The `glove.6B.100d.txt` file is used because it provides pre-trained word embeddings with a dimensionality of 100, which strikes a balance between capturing semantic information and maintaining computational efficiency. These embeddings, trained on a large corpus (Wikipedia and Gigaword 5), offer a rich representation of word meanings and relationships, enhancing model performance without needing extensive computational resources. The 100-dimensional vectors are generally sufficient to capture nuanced language features, making them a practical choice for text classification tasks like ours.
### Preprocessing Module for BERT

The preprocessing module for BERT leverages BERT's own tokenizer, specifically `BertTokenizer`, to prepare text data in a way that aligns with BERT's architecture. This involves converting text into token IDs, applying truncation or padding to a fixed length (`max_len`), and creating an attention mask. BERT’s tokenizer efficiently handles sub-word tokenization (e.g., handling unknown words or complex structures) and adds special tokens `[CLS]` and `[SEP]` required for BERT’s input format. This approach ensures the text data is compatible with BERT’s model requirements and maximizes its capability to capture contextual information.

## Model architectures

### DNN

The DNN model architecture follows a straightforward dense (fully connected) network approach. It consists of two main inputs—`Title` and `Description`—each processed through a shared embedding layer using pre-trained GloVe embeddings. These embeddings are then flattened and merged, enabling the model to combine the distinct representations from each input. 

The merged output is fed into two dense layers with ReLU activations, allowing the network to learn complex, non-linear relationships. The final output layer uses a softmax activation, producing a probability distribution across four classes. This simple, fully connected approach provides a baseline for comparison with more complex architectures. Early stopping is included to prevent overfitting by halting training when the validation loss stops improving.

### CNN

The CNN model architecture uses convolutional layers to capture local patterns in the text sequences, making it effective for detecting key features in shorter contexts like phrases or word combinations. Similar to the DNN model, it takes two inputs, `Title` and `Description`, processed separately through a shared embedding layer using GloVe embeddings.

For each input, a 1D convolutional layer with 128 filters and a kernel size of 3 captures local dependencies between neighboring words, followed by max-pooling to reduce dimensionality and retain significant features. The pooled outputs are flattened and then concatenated, allowing both inputs to contribute to the final feature space.

The merged features pass through two dense layers with ReLU activations for further non-linear transformation, and a softmax output layer yields class probabilities. Early stopping prevents overfitting by stopping training when validation loss no longer improves. This CNN-based structure is more adept at capturing local word patterns than the DNN model.
### RNN

The RNN model architecture leverages LSTM (Long Short-Term Memory) layers to capture sequential dependencies and contextual information from the text inputs, which is particularly valuable for handling longer-term word relationships in text.

Each input, `Title` and `Description`, is processed separately with shared GloVe embeddings. An LSTM layer with 128 units follows each embedding, allowing each sequence to capture temporal dependencies without returning sequences. This setup is ideal for extracting meaningful context from sequences, as LSTMs are known for handling vanishing gradient issues and preserving information over longer input sequences.

The outputs from the LSTM layers are concatenated, combining both inputs, and passed through two fully connected dense layers with ReLU activations for deeper feature transformations. A softmax output layer provides class probabilities for the classification task. Early stopping is implemented to halt training when validation performance plateaus, preventing overfitting. This RNN-based model structure is optimized for capturing contextual flow within the text.
### Transformers-LSTM-GRU

The Transformers-LSTM-GRU model combines GRU and LSTM layers to leverage both sequence memory capabilities for text processing. This model structure aims to capture both short-term and long-term dependencies in the text inputs (`Title` and `Description`), ensuring a robust context representation for each input.

Each input begins with GloVe embeddings, followed by a GRU layer with 128 units, which captures immediate dependencies and filters essential features within each sequence. Then, an LSTM layer with 128 units processes the GRU output, adding the ability to capture longer-term dependencies without returning sequences, resulting in a focused, final output for each input.

The outputs from the LSTM layers are concatenated, merging the two input sources into a single representation. This merged layer is then processed through two fully connected layers with ReLU activations, further refining the feature space. Finally, a softmax layer outputs class probabilities for classification. Early stopping is applied to avoid overfitting by monitoring the validation loss. 

This combined GRU-LSTM structure provides an enhanced sequence model designed to harness both recent and distant context in text data.
### BERT

The BERT-based model uses a custom BERT layer to leverage contextualized embeddings, capturing deep semantic understanding from the text data. In this structure, the BERT model (`bert-base-uncased`) processes the `input_ids` and `attention_masks`, providing a robust representation of text sequences. This approach allows the model to learn intricate language patterns by pre-trained attention mechanisms.

The `BertLayer` outputs a pooled representation of the input sequences, which is fed into a fully connected (dense) layer with ReLU activation to learn classification-specific features. A dropout layer follows to mitigate overfitting. The final layer is a softmax for multi-class classification.

This architecture is powerful for tasks needing deep contextual language understanding, making it suitable for more complex natural language processing tasks.
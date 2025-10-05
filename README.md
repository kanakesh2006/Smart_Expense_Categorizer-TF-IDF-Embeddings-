-----

# Smart Expense Categorizer (Transaction Text Classification) 

## 1\. Project Objective

The primary goal of this project is to build a **Smart Expense Categorizer** using Natural Language Processing (NLP) techniques. The system automatically classifies short, cryptic transaction narratives (like SMS alerts or bank statement descriptions) into predefined expense categories.

This project addresses a common challenge in personal finance tools: converting unstructured text data into structured, actionable categories for budgeting and analysis.

**Example:**

  * `TXN SUCCESSFUL FOR 250 AT SWIGGY` $\rightarrow$ **Food**
  * `TXN FOR 300 AT UBER` $\rightarrow$ **Travel**

## 2\. Technical Stack (Tech Stack) 🛠️

This project utilizes a standard Python-based data science stack, running entirely within a **Google Colab** environment for easy setup and resource access.

| Component | Technology | Purpose & Explanation |
| :--- | :--- | :--- |
| **Language** | `Python 3.x` | The core programming language. |
| **Environment** | `Google Colab` | Cloud-based Jupyter Notebook environment for execution. |
| **Data Handling** | `Pandas`, `NumPy`, `re` | Used for structured data manipulation, numerical operations, and **Regular Expressions (`re`)** for text cleaning. |
| **ML Framework** | `scikit-learn` | Used for traditional ML components: data splitting, feature extraction, and the Logistic Regression model. |
| **Feature Extraction** | `TfidfVectorizer` (scikit-learn) | **TF-IDF (Term Frequency-Inverse Document Frequency)**: A statistical measure used to evaluate how important a word is to a document in a collection. It transforms text into a numerical matrix suitable for ML models. |
| **Traditional ML Model** | `LogisticRegression` (scikit-learn) | A highly effective and interpretable linear model used for baseline text classification. |
| **Deep Learning** | `TensorFlow/Keras` | Used to build a simple **Word Embedding**-based neural network model as an alternative approach. |

## 3\. Implementation Details

The project is implemented in a single Google Colab notebook, divided into three main phases:

### Phase A: Data Preparation (Synthetic Generation)

  * **Necessity:** Real transaction data is highly sensitive and proprietary. Therefore, a **synthetic dataset** was generated using Python dictionaries and loops, mimicking realistic merchant names (e.g., `SWIGGY`, `UBER`, `AMAZON`) paired with random amounts.
  * **Data Structure:** The resulting data frame contains two key columns:
      * `Transaction_Text`: The raw input string.
      * `Category`: The ground truth label (e.g., 'Food', 'Travel').
  * **Text Cleaning:** A custom function was applied using **Regular Expressions** to preprocess the raw text by:
    1.  Converting text to lowercase.
    2.  Removing common stop words and generic bank phrases (`TXN SUCCESSFUL FOR`, `VIA DEBIT CARD`).
    3.  Removing numerical values (transaction amounts) to force the model to categorize solely based on the *merchant/keyword* rather than the amount.

### Phase B: Model 1 - TF-IDF + Logistic Regression (Baseline)

1.  **Feature Vectorization:** The cleaned text data is transformed into a sparse matrix using **`TfidfVectorizer`**. This step converts the text documents into a numerical representation where each value reflects the importance of a word.
2.  **Training:** A **Logistic Regression** model is trained on the TF-IDF vectors. This model learns the linear relationship between the word importance scores and the target categories.
3.  **Evaluation:** The model is evaluated using **Accuracy** and a detailed **Classification Report** (Precision, Recall, F1-score) to assess its performance across all categories.

### Phase C: Model 2 - Word Embeddings (Deep Learning Alternative)

1.  **Tokenization:** The text is tokenized using Keras's `Tokenizer`, converting words into integer indices.
2.  **Encoding:** The categorical labels are converted to numerical integers using `LabelEncoder`.
3.  **Padding:** Sequences are padded to a uniform length to be compatible with the neural network's input layer.
4.  **Model Architecture:** A simple **Sequential Keras model** is built:
      * **Embedding Layer:** Converts the integer indices into dense, fixed-size vectors (embeddings), allowing the model to learn semantic relationships between merchants.
      * **Flatten Layer:** Prepares the embeddings for the dense classification layer.
      * **Dense Layers (Classification Head):** A final set of dense layers with a `softmax` activation predicts the probability for each expense category.

## 4\. Screenshots and Working Model (Simulated)

Below are the expected outputs and structure you would see in the Google Colab environment.

### 4.1. Data Sample and Distribution

![Data Labels](target_categories.jpg)

### 4.2. TF-IDF + Logistic Regression Results

This shows the performance of the traditional ML approach.

**Code Output Snippet:**

```
--- TF-IDF + Logistic Regression Results ---
Accuracy: 1.0000
Classification Report:
               precision    recall  f1-score   support

Entertainment       1.00      1.00      1.00        20
Food                1.00      1.00      1.00        20
Groceries           1.00      1.00      1.00        20
Shopping            1.00      1.00      1.00        20
Travel              1.00      1.00      1.00        20
Utilities           1.00      1.00      1.00        20

    accuracy                            1.00       120
   macro avg      1.00      1.00      1.00       120
weighted avg      1.00      1.00      1.00       120

Prediction Test:
Text: 'txn for 450 at PIZZA HUT' -> Predicted Category: Food
Text: 'txn for 2500 at FLIPKART' -> Predicted Category: Shopping
```

  * ***Note on 100% Accuracy:*** *Due to the synthetic nature of the dataset (where merchant names are perfectly unique and clean), the model achieves perfect separation. Real-world data, with misspellings and noise, would produce a more realistic accuracy (e.g., 85-95%).*

### 4.3. Simple Embedding Model Results

This shows the performance of the neural network approach.

**Code Output Snippet:**

```
--- Simple Embedding Model (NN) Results ---
Test Accuracy: 1.0000

Prediction Test (NN):
Text: 'txn for 450 at PIZZA HUT' -> Predicted Category: Food
Text: 'txn for 2500 at FLIPKART' -> Predicted Category: Shopping
```

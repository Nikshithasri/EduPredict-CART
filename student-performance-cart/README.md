# Student Performance Prediction Using Classification and Regression Trees (CART)

A complete, beginner-friendly Machine Learning web application implementing the **Classification and Regression Trees (CART)** decision tree algorithm to predict whether a student will **PASS** or **FAIL** based on their academic profile.

This project is designed for **Machine Learning Unit 5 – Combining Models**, demonstrating the practical implementation, mathematical theory, and visualization of tree-based models.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Technology Stack](#-technology-stack)
3. [The CART Algorithm (Unit 5 Theory)](#-the-cart-algorithm-unit-5-theory)
4. [Dataset & Features](#-dataset--features)
5. [Project Architecture](#-project-architecture)
6. [Directory Structure](#-directory-structure)
7. [Installation & Setup](#-installation--setup)
8. [Running in VS Code](#-running-in-vs-code)
9. [Application Walkthrough](#-application-walkthrough)
10. [Model Evaluation & Insights](#-model-evaluation--insights)

---

## 🎯 Project Overview
- **Objective:** Predict student academic outcomes (`PASS` or `FAIL`) using continuous features.
- **Algorithm:** CART (Decision Tree Classifier with **Gini Impurity**).
- **Split Ratio:** 80% Training, 20% Testing (Stratified).
- **Backend:** Python + Flask.
- **Frontend:** Semantic HTML5, CSS3 (Modern Academic Theme in Navy Blue & Purple), Vanilla JavaScript.
- **Model Persistence:** Joblib (`model.pkl`).

---

## 💻 Technology Stack
- **Python 3.8+**
- **Flask (v3.0+)**: Lightweight web framework for routing and template rendering.
- **Scikit-learn**: CART implementation (`DecisionTreeClassifier(criterion='gini')`).
- **Pandas & NumPy**: Data processing and numeric matrix transformations.
- **Matplotlib**: Generation of Decision Tree, Confusion Matrix, and Feature Importance charts.
- **Joblib**: Serializing and loading the trained machine learning pipeline.

*No external database, React, authentication, or external API dependencies are used.*

---

## 🧠 The CART Algorithm (Unit 5 Theory)

### What is CART?
Introduced by Leo Breiman, Jerome Friedman, Richard Olshen, and Charles Stone (1984), **CART** stands for **Classification and Regression Trees**:
- **Binary Splitting:** At each internal node, CART partitions data into exactly **two branches** based on a single feature and threshold (e.g., `Internal Marks <= 48.5`).
- **Classification vs. Regression:**
  - When the target variable is categorical (`Pass` / `Fail`), it creates a **Classification Tree**.
  - When the target variable is continuous, it creates a **Regression Tree** (minimizing Mean Squared Error).

### Splitting Criterion: Gini Impurity
Unlike ID3 or C4.5 which use Shannon Entropy and Information Gain, CART uses **Gini Impurity**:

$$Gini(D) = 1 - \sum_{i=1}^{C} p_i^2$$

Where:
- $C$ is the number of classes (here $C=2$: Pass and Fail).
- $p_i$ is the probability/fraction of samples belonging to class $i$ at node $D$.

**Interpretation:**
- $Gini = 0.0$: Perfectly pure node (100% of samples belong to one class).
- $Gini = 0.5$: Maximum impurity (equal 50/50 split between Pass and Fail).

### Finding the Best Split
CART evaluates every potential split point for all 4 features and calculates the weighted Gini impurity:

$$Gini_{split}(D) = \frac{|D_1|}{|D|} Gini(D_1) + \frac{|D_2|}{|D|} Gini(D_2)$$

The algorithm selects the split that maximizes the **Gini Gain**:

$$\Delta Gini = Gini(D) - Gini_{split}(D)$$

### Connection to Unit 5: Combining Models
Single decision trees can have high variance. In **Unit 5 – Combining Models**, CART serves as the base learner for powerful ensemble algorithms:
1. **Bagging (Bootstrap Aggregation):** Trains multiple parallel CART trees on bootstrap samples to reduce variance.
2. **Random Forests:** Builds upon Bagging by selecting random feature subsets at each node split.
3. **Boosting (AdaBoost & Gradient Boosting):** Sequentially fits CART trees where each subsequent tree corrects errors made by the previous trees.

---

## 📊 Dataset & Features
The dataset contains **250 student records** with realistic academic distributions:

| Feature Name | Type | Range | Description |
| :--- | :--- | :--- | :--- |
| **`attendance`** | Numeric (Float) | 0% – 100% | Classroom lecture attendance percentage |
| **`study_hours`** | Numeric (Float) | 0 – 12 hrs/day | Average daily self-study and revision time |
| **`internal_marks`** | Numeric (Float) | 0 – 100 | Midterm and continuous internal assessment tests |
| **`assignment_score`** | Numeric (Float) | 0 – 100 | Homework, lab assignments, and project submissions |
| **`result`** *(Target)* | Categorical | `Pass` / `Fail` | Semester final outcome |

---

## 🏗 Directory Structure

```text
student-performance-cart/
│
├── app.py                     # Flask web server, routing, validation & ML inference
├── train_model.py             # Dataset generation, CART training & plot exports
├── dataset.csv                # Student academic dataset (250 samples)
├── model.pkl                  # Serialized Scikit-learn DecisionTreeClassifier
├── metrics.json               # Saved evaluation statistics (accuracy, matrix, report)
├── requirements.txt           # Python library dependencies
├── test_app.py                # Automated unit test suite (9 tests)
├── README.md                  # Detailed documentation and college viva guide
│
├── templates/
│   ├── index.html             # Home page: Title, CART summary, model accuracy
│   ├── predict.html           # Prediction form: 4 feature inputs, presets & validation
│   ├── result.html            # Prediction result: PASS/FAIL, confidence & explanation
│   └── insights.html          # Analytics: Tree diagram, confusion matrix, CART formulas
│
└── static/
    ├── style.css              # Custom responsive stylesheet (Navy Blue & Purple)
    └── images/
        ├── decision_tree.png      # High-res plot of trained CART Decision Tree
        ├── confusion_matrix.png   # Heatmap with counts and percentages
        └── feature_importance.png # Bar chart of feature Gini contributions
```

---

## ⚙️ Installation & Setup

### Step 1: Open Terminal / Command Prompt
Navigate to the project directory:
```bash
cd "c:\Users\Admin\Downloads\CART\student-performance-cart"
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
```bash
python -m venv venv
```
Activate on Windows:
```bash
.\venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running in VS Code

1. Open **VS Code**.
2. Go to **File -> Open Folder...** and select `student-performance-cart` (or `CART`).
3. Open a new terminal in VS Code (`Ctrl + ~` or **Terminal -> New Terminal**).
4. Run the training script (creates `dataset.csv`, `model.pkl`, and all plot images):
   ```bash
   python train_model.py
   ```
5. Run the web application:
   ```bash
   python app.py
   ```
6. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

---

## 🌐 Application Walkthrough

### 1. Home Page (`/`)
- Highlights the project title: **Student Performance Prediction Using Classification and Regression Trees (CART)**.
- Explains CART fundamentals and its role in Machine Learning Unit 5.
- Displays live model accuracy badges (Training: 94.00%, Testing: 88.00%).
- Provides quick CTA buttons: "Start Prediction" and "View Model Insights".

### 2. Prediction Page (`/predict`)
- Clean input form with numeric inputs:
  - Attendance Percentage (`0 - 100%`)
  - Study Hours per Day (`0 - 12 hrs`)
  - Internal Marks (`0 - 100`)
  - Assignment Score (`0 - 100`)
- **Quick Test Presets:**
  - *High Achiever:* Attendance: 92%, Study: 6.5h, Internals: 85, Assignments: 90
  - *Average / Borderline:* Attendance: 68%, Study: 3.5h, Internals: 54, Assignments: 62
  - *At-Risk Student:* Attendance: 48%, Study: 1.2h, Internals: 34, Assignments: 42
- Robust client-side (JavaScript) and server-side (Flask) validation with clear error messages.

### 3. Prediction Result (`/predict` POST)
- Displays a prominent **PASS** (emerald green) or **FAIL** (crimson red) outcome badge.
- Shows model **Confidence Percentage** (e.g., 93.3% confidence).
- Visual probability progress bar comparing Pass vs Fail odds.
- Pedagogical decision tree rationale explaining which inputs drove the classification.
- Full summary table of submitted student metrics.
- Buttons to "Predict Another Student" or "View Decision Tree in Insights".

### 4. Model Insights Page (`/insights`)
- **High-Resolution Decision Tree Diagram:** Visualizes every decision node with sample counts, Gini values, and class distributions. Includes an "Expand High-Res View" zoom modal.
- **Confusion Matrix:** Heatmap showing True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP).
- **Feature Importance Chart:** Bar chart showing the relative percentage importance of each academic feature.
- **Classification Report:** Detailed table of Precision, Recall, and F1-score.
- **CART Theory Section:** Complete mathematical breakdown of Gini impurity and recursive splitting for exams/viva.

---

## 🧪 Automated Testing
To run the automated test suite verifying all 9 unit tests:
```bash
python test_app.py
```
Expected output:
```text
Ran 9 tests in 1.2s
OK
```

---

## 🎓 College Viva & Examination Questions

**Q1: Why did we choose CART over ID3?**
> *Answer:* ID3 uses Information Gain based on Shannon Entropy and only handles categorical splits. CART supports continuous numeric attributes directly, produces binary trees, and uses Gini Impurity, which is computationally faster because it avoids logarithmic operations.

**Q2: What does Gini = 0 indicate?**
> *Answer:* A Gini value of 0 indicates a completely pure node where 100% of the training instances belong to a single class (either all Pass or all Fail).

**Q3: How does CART avoid overfitting?**
> *Answer:* In our implementation, we constrained `max_depth=4`, `min_samples_split=6`, and `min_samples_leaf=3`. These pre-pruning stopping conditions prevent the tree from growing leaves that memorize single noisy training samples.

---

**Developed for Academic Machine Learning Demonstrations.**

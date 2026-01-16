# In-Class Activity: Build Your Own Linear Regression App

## 🎯 Learning Objectives

By the end of this activity, you will:
1. Build a complete linear regression application from scratch
2. Implement gradient descent optimization manually
3. Create interactive visualizations with Plotly
4. Deploy a functional Streamlit web application
5. Analyze model performance and interpret results
6. Practice modular code organization and documentation

---

## 📋 Activity Overview

**Time**: 90-120 minutes
**Type**: Individual or Pair Programming
**Difficulty**: Intermediate

You will build a **Student Performance Prediction System** that predicts final exam scores based on study hours using linear regression. Your app will include data generation, model training with gradient descent, visualizations, and interactive predictions.

---

## 🎬 Scenario

You are a data scientist hired by a university to build a tool that helps students understand the relationship between study time and exam performance. The tool should:

- Generate realistic student performance data
- Train a linear regression model using gradient descent (coded by you!)
- Visualize the learning process
- Allow students to predict their exam scores based on planned study hours
- Show model reliability metrics

---

## 📦 Setup

### 1. Create Project Structure

```bash
mkdir student-performance-predictor
cd student-performance-predictor
```

Create the following structure:
```
student-performance-predictor/
├── app.py                 # Main Streamlit app (you'll create this)
├── model.py              # Linear regression implementation (you'll create this)
├── data_generator.py     # Data generation utilities (you'll create this)
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

### 2. Install Dependencies

Create `requirements.txt`:
```
streamlit==1.32.0
numpy==1.24.3
pandas==2.0.3
plotly==5.18.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## 🔨 Implementation Tasks

### Part 1: Data Generator (20 minutes)

Create `data_generator.py` with a class that generates synthetic student data.

**Requirements**:
- Function to generate study hours (2-10 hours per day)
- Function to generate exam scores based on the formula:
  ```
  score = 40 + 6 * study_hours + noise
  ```
  Where noise has standard deviation of 5-10 points
- Include 5-10% outliers (students who performed unexpectedly well/poorly)
- Return data as numpy arrays

**Hints**:
- Use `np.random.uniform()` for study hours
- Use `np.random.normal()` for noise
- Ensure scores are between 0-100

<details>
<summary>Click for starter code structure</summary>

```python
import numpy as np

class StudentDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)

    def generate_data(self, n_samples=100, noise_std=8, outlier_pct=5):
        """
        Generate synthetic student performance data.

        Args:
            n_samples: Number of students
            noise_std: Standard deviation of noise
            outlier_pct: Percentage of outliers (0-100)

        Returns:
            study_hours: Array of study hours
            exam_scores: Array of exam scores
        """
        # TODO: Implement this
        pass
```
</details>

---

### Part 2: Linear Regression Model (30 minutes)

Create `model.py` with a `LinearRegressionGD` class that implements gradient descent.

**Requirements**:
- `__init__`: Initialize with learning rate and max iterations
- `fit(X, y)`: Train the model using gradient descent
  - Initialize slope (m) and intercept (b) to zero
  - For each iteration:
    - Compute predictions: `y_pred = m * X + b`
    - Compute gradients:
      ```
      grad_m = (1/n) * sum((y_pred - y) * X)
      grad_b = (1/n) * sum(y_pred - y)
      ```
    - Update parameters:
      ```
      m = m - learning_rate * grad_m
      b = b - learning_rate * grad_b
      ```
    - Store history (cost, m, b, gradients)
- `predict(X)`: Make predictions using trained parameters
- `calculate_metrics(X, y)`: Return R², MSE, RMSE, MAE

**Hints**:
- Store training history in lists/dictionaries
- Compute cost as: `cost = (1/(2*n)) * sum((y_pred - y)²)`
- Remember to handle numpy arrays properly

<details>
<summary>Click for starter code structure</summary>

```python
import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.slope = 0
        self.intercept = 0
        self.history = {
            'cost': [],
            'slope': [],
            'intercept': [],
            'grad_slope': [],
            'grad_intercept': []
        }

    def fit(self, X, y):
        """Train the model using gradient descent."""
        # TODO: Implement gradient descent
        pass

    def predict(self, X):
        """Make predictions."""
        return self.slope * X + self.intercept

    def calculate_metrics(self, X, y):
        """Calculate performance metrics."""
        # TODO: Implement R², MSE, RMSE, MAE
        pass
```
</details>

---

### Part 3: Streamlit Application (40 minutes)

Create `app.py` with the following sections:

#### Section 1: Configuration Sidebar
- Sliders for:
  - Number of students (50-200)
  - Noise level (0-15)
  - Outlier percentage (0-20%)
  - Learning rate (0.001-0.1)
  - Max iterations (100-2000)
  - Random seed
- Button to regenerate data

#### Section 2: Data Visualization
- Title and description
- Generate and display data
- Show statistics (mean study hours, mean score, correlation)
- Scatter plot of study hours vs exam scores
- Table showing first 10 rows

#### Section 3: Model Training
- Train the model with user's hyperparameters
- Display final parameters (slope, intercept)
- Display model equation: `Score = m × hours + b`
- Show interpretation (e.g., "Each additional hour of study increases expected score by X points")
- Display metrics (R², MSE, RMSE, MAE)

#### Section 4: Visualizations
Create at least 4 plots:
1. **Regression fit**: Scatter plot with fitted line
2. **Cost convergence**: Cost vs iterations
3. **Residual plot**: Residuals vs predictions (to check for patterns)
4. **Prediction intervals**: Show confidence in predictions

#### Section 5: Interactive Predictions
- Number input for study hours
- Display predicted exam score
- Show prediction with context (e.g., "With 7 hours of study, expect ~82 points")
- Add a fun message based on score range

**Hints**:
- Use `st.title()`, `st.header()`, `st.subheader()` for structure
- Use `st.columns()` for side-by-side metrics
- Use `st.plotly_chart()` for interactive plots
- Use `st.markdown()` for formatted text with LaTeX

<details>
<summary>Click for app structure template</summary>

```python
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from data_generator import StudentDataGenerator
from model import LinearRegressionGD

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.title("📚 Student Performance Prediction System")
st.markdown("Predict exam scores based on study hours using linear regression!")

# Sidebar configuration
st.sidebar.header("Configuration")
# TODO: Add all configuration sliders

# Generate data
# TODO: Implement data generation

# Display data
st.header("1️⃣ Student Data")
# TODO: Show statistics and scatter plot

# Train model
st.header("2️⃣ Model Training")
# TODO: Train and display results

# Visualizations
st.header("3️⃣ Training Visualizations")
# TODO: Create plots

# Interactive predictions
st.header("4️⃣ Make Predictions")
# TODO: Add prediction interface
```
</details>

---

### Part 4: Testing and Validation (10 minutes)

Test your app with different scenarios:

1. **Low learning rate** (0.001): Should converge slowly
2. **High learning rate** (0.5): Might oscillate or diverge
3. **High noise** (15): Lower R² score
4. **Many outliers** (20%): Line pulled away from main trend
5. **Edge cases**:
   - Predict with 0 hours (should give intercept)
   - Predict with 10 hours (maximum reasonable study time)

---

## 🎨 Bonus Challenges (Optional)

### Level 1: Enhanced Features
- [ ] Add train/test split functionality
- [ ] Compare gradient descent with closed-form solution
- [ ] Add animation slider showing training progress
- [ ] Include downloadable CSV of predictions

### Level 2: Advanced Visualizations
- [ ] 3D cost surface plot
- [ ] Learning rate comparison (train with multiple learning rates simultaneously)
- [ ] Add confidence intervals to predictions
- [ ] Create a gradient magnitude plot

### Level 3: Extended Functionality
- [ ] Add multiple features (study hours, attendance, sleep)
- [ ] Implement polynomial regression
- [ ] Add regularization (Ridge/Lasso)
- [ ] Compare with sklearn's LinearRegression

### Level 4: Production Ready
- [ ] Add input validation and error handling
- [ ] Include comprehensive documentation
- [ ] Add unit tests for model functions
- [ ] Deploy to Streamlit Cloud

---

## 📊 Assessment Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| **Data Generator** | 20 | Correctly generates data with noise and outliers |
| **Gradient Descent** | 30 | Properly implements gradient descent with history tracking |
| **Metrics Calculation** | 15 | Accurately computes R², MSE, RMSE, MAE |
| **Visualizations** | 20 | Creates clear, labeled plots with appropriate scales |
| **Interactivity** | 10 | Sidebar controls work correctly and update app |
| **Code Quality** | 5 | Clean, commented, modular code |
| **Bonus** | +10 | Extra credit for bonus challenges |

**Total: 100 points (+10 bonus)**

---

## 🚀 Deployment Instructions

Once your app is working locally, deploy it to Streamlit Cloud:

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Student Performance Predictor"
   git remote add origin https://github.com/YOUR_USERNAME/student-performance-predictor.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to https://share.streamlit.io
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Share your app URL** with the class!

---

## 💡 Tips for Success

1. **Start simple**: Get basic functionality working before adding features
2. **Test incrementally**: Test each function as you write it
3. **Use print statements**: Debug by printing values during development
4. **Read error messages**: Streamlit shows helpful error messages
5. **Check dimensions**: Ensure numpy arrays have correct shapes
6. **Ask for help**: Collaborate with classmates when stuck

---

## 🎓 Learning Checkpoints

After completing this activity, you should be able to answer:

- [ ] How does the learning rate affect convergence speed?
- [ ] What causes the cost function to increase instead of decrease?
- [ ] How do you interpret the R² score in the context of your model?
- [ ] What is the relationship between gradients and parameter updates?
- [ ] How do outliers affect the regression line?
- [ ] When would you use gradient descent vs closed-form solution?

---

## 📚 Resources

- **Streamlit Documentation**: https://docs.streamlit.io
- **Plotly Python**: https://plotly.com/python/
- **NumPy Documentation**: https://numpy.org/doc/
- **Demo App** (for reference): The app you just explored in class

---

## 🤝 Collaboration Guidelines

- **Pair Programming**: Work in pairs if desired, but both students must understand all code
- **Code Sharing**: You may discuss approaches, but write your own implementation
- **Getting Unstuck**: Ask classmates/instructor for conceptual help, not direct code solutions
- **Attribution**: If you use code from online resources, cite the source in comments

---

## 📝 Submission Instructions

Submit the following:

1. **GitHub repository link** with your complete code
2. **Deployed Streamlit app URL**
3. **README.md** including:
   - Description of your app
   - How to run it locally
   - Findings from testing different hyperparameters
   - Answers to learning checkpoint questions
   - Any challenges faced and how you solved them

**Due Date**: [Instructor to fill in]

**Submission Method**: [Instructor to fill in]

---

## 🎉 Example Output

Your final app should allow users to:
1. Generate realistic student performance data
2. Adjust hyperparameters via sidebar
3. Train a model and see convergence
4. Visualize the fitted line and training process
5. Make predictions for new study hour values
6. Understand model performance through metrics

---

## ❓ FAQ

**Q: My cost is increasing instead of decreasing. What's wrong?**
A: Your learning rate is too high. Decrease it and try again.

**Q: The gradient descent is too slow. How can I speed it up?**
A: Increase the learning rate, but not too much! Also check your max iterations.

**Q: My R² is negative. Is that possible?**
A: Yes! It means your model is worse than just predicting the mean. Check your implementation.

**Q: How do I add LaTeX equations in Streamlit?**
A: Use `st.latex()` or include LaTeX in `st.markdown()` like `$y = mx + b$`

**Q: Can I use sklearn instead of implementing gradient descent?**
A: For this activity, you must implement gradient descent yourself to understand the algorithm. You can compare with sklearn as a bonus!

---

## 🌟 Instructor Notes

**Time Management**:
- 10 min: Introduction and setup
- 20 min: Part 1 (Data Generator)
- 30 min: Part 2 (Model)
- 40 min: Part 3 (Streamlit App)
- 10 min: Testing
- 10 min: Deployment walkthrough

**Common Issues**:
- Students forget to flatten arrays or handle dimensions
- Learning rate too high causing divergence
- Forgetting to update history during training
- Not normalizing features (optional, but can help)

**Assessment Focus**:
- Correct implementation of gradient descent algorithm
- Understanding of partial derivatives and updates
- Ability to interpret visualizations
- Code organization and documentation

**Extensions for Advanced Students**:
- Feature scaling/normalization
- Batch gradient descent vs mini-batch
- Momentum or adaptive learning rates
- Cross-validation

Good luck! 🚀

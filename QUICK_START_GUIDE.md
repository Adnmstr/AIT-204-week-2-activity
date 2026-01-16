# Quick Start Guide for Students

## 🚀 Getting Started in 5 Minutes

Follow this guide to set up your environment and start building your app quickly!

---

## Step 1: Set Up Your Project (2 minutes)

### Create Project Directory
```bash
# Navigate to where you want your project
cd ~/Desktop  # or wherever you prefer

# Create project folder
mkdir student-performance-predictor
cd student-performance-predictor
```

### Create Project Structure
```bash
# Create the necessary files
touch app.py
touch model.py
touch data_generator.py
touch requirements.txt
touch README.md
```

Your directory should now look like:
```
student-performance-predictor/
├── app.py
├── model.py
├── data_generator.py
├── requirements.txt
└── README.md
```

---

## Step 2: Set Up Dependencies (1 minute)

### Create requirements.txt
Open `requirements.txt` and add:
```
streamlit==1.32.0
numpy==1.24.3
pandas==2.0.3
plotly==5.18.0
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Step 3: Test Streamlit (1 minute)

Create a simple test to make sure Streamlit works:

**Open `app.py` and add:**
```python
import streamlit as st

st.title("📚 Student Performance Predictor")
st.write("Hello! This is my app.")
```

**Run it:**
```bash
streamlit run app.py
```

If a browser opens with your message, you're ready! Press `Ctrl+C` to stop.

---

## Step 4: Start Building! (60-90 minutes)

Now follow the **IN_CLASS_ACTIVITY.md** instructions to build each component:

### 🗂️ Recommended Order:

1. **Start with `data_generator.py`** (20 min)
   - Just get basic data generation working first
   - Test it in a Python terminal:
   ```python
   from data_generator import StudentDataGenerator
   gen = StudentDataGenerator()
   hours, scores = gen.generate_data()
   print(hours[:5], scores[:5])  # Should print arrays
   ```

2. **Then `model.py`** (30 min)
   - Implement `fit()` method first
   - Then `predict()`
   - Finally `calculate_metrics()`
   - Test each method before moving on!

3. **Finally `app.py`** (40 min)
   - Start with Section 1 (data display)
   - Add Section 2 (training)
   - Add Section 3 (visualizations)
   - Add Section 4 (predictions)

---

## 💡 Pro Tips

### Tip 1: Test as You Go
Don't write everything at once! Test each function:

```python
# Example: Test your gradient descent
from model import LinearRegressionGD
import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegressionGD(learning_rate=0.01, max_iterations=100)
model.fit(X, y)
print(f"Slope: {model.slope}, Intercept: {model.intercept}")
# Should be close to: Slope: 2, Intercept: 0
```

### Tip 2: Use Print Statements for Debugging
```python
# Inside your gradient descent loop
print(f"Iteration {iteration}, Cost: {cost:.2f}, m: {m:.2f}, b: {b:.2f}")
```

### Tip 3: Start Simple, Add Complexity
```python
# Start with this:
study_hours = np.random.uniform(2, 10, 100)
exam_scores = 40 + 6 * study_hours + np.random.normal(0, 8, 100)

# THEN add outliers, clipping, etc.
```

### Tip 4: Check Array Shapes
```python
# Always verify your arrays are 1D
X = X.flatten()
y = y.flatten()
print(f"X shape: {X.shape}, y shape: {y.shape}")  # Should be (n,)
```

### Tip 5: Copy-Paste Template Code
The activity document has code templates in dropdown sections. Start with those!

---

## 🐛 Common Errors & Quick Fixes

### Error: "operands could not be broadcast"
**Fix**: Flatten your arrays
```python
X = np.array(X).flatten()
y = np.array(y).flatten()
```

### Error: Cost is increasing
**Fix**: Lower your learning rate
```python
# Change from:
learning_rate = 0.1
# To:
learning_rate = 0.01
```

### Error: "ModuleNotFoundError: No module named 'streamlit'"
**Fix**: Install requirements
```bash
pip install -r requirements.txt
```

### Error: R² is negative
**Fix**: Check your gradient calculations. They should be:
```python
grad_m = (1/n) * np.sum((y_pred - y) * X)
grad_b = (1/n) * np.sum(y_pred - y)
```

### Error: Nothing happens when I move sliders
**Fix**: Make sure you're regenerating data when sliders change. Streamlit reruns automatically!

---

## 📋 Development Checklist

Use this to track your progress:

### Data Generator
- [ ] Generate random study hours (2-10)
- [ ] Calculate scores with formula
- [ ] Add noise
- [ ] Add outliers
- [ ] Clip to [0, 100]
- [ ] Test: `python -c "from data_generator import *; print('Works!')"`

### Model
- [ ] Initialize parameters to zero
- [ ] Implement forward pass (predictions)
- [ ] Calculate cost
- [ ] Calculate gradients
- [ ] Update parameters
- [ ] Store history
- [ ] Implement predict()
- [ ] Implement calculate_metrics()
- [ ] Test: Cost should decrease!

### Streamlit App
- [ ] Import libraries
- [ ] Add title and description
- [ ] Create sidebar with sliders
- [ ] Generate and display data
- [ ] Train model
- [ ] Display parameters and metrics
- [ ] Create scatter plot
- [ ] Create cost convergence plot
- [ ] Create regression line plot
- [ ] Add prediction interface
- [ ] Test: App runs without errors!

---

## ⏱️ Time Check-ins

### After 20 minutes:
✅ You should have: Data generation working

### After 50 minutes:
✅ You should have: Model training with decreasing cost

### After 90 minutes:
✅ You should have: Basic app with plots working

### After 120 minutes:
✅ You should have: Complete app with predictions

---

## 🆘 Getting Help

1. **Check the error message** - Streamlit shows errors clearly
2. **Print debug info** - Add print statements to see values
3. **Compare with demo** - Reference the app you explored in class
4. **Ask a classmate** - Discuss approaches (but write your own code!)
5. **Ask instructor** - Get unstuck, but try first!

---

## 🎯 Minimum Viable Product (MVP)

If you're short on time, aim for this first:

**Must Have:**
- Data generation (even without outliers)
- Gradient descent that decreases cost
- One plot showing the data
- One plot showing the fitted line
- Basic Streamlit app structure

**Then Add:**
- Metrics (R², MSE, etc.)
- More plots
- Prediction interface
- Polish and explanations

---

## 🚢 Deployment (If Time Permits)

### Push to GitHub:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Deploy on Streamlit Cloud:
1. Go to https://share.streamlit.io
2. Connect GitHub
3. Select your repo
4. Set main file: `app.py`
5. Deploy!

---

## 📚 Helpful Resources

- **NumPy Basics**: https://numpy.org/doc/stable/user/quickstart.html
- **Streamlit Docs**: https://docs.streamlit.io
- **Plotly Examples**: https://plotly.com/python/
- **Gradient Descent Tutorial**: Review the demo app explanations!

---

## ✨ Final Tips

1. **Don't overthink it** - Start simple, iterate
2. **Test frequently** - Don't write 100 lines before testing
3. **Read error messages** - They usually tell you exactly what's wrong
4. **Have fun!** - You're building a real ML app!

---

**Ready? Let's build! 🚀**

Open `IN_CLASS_ACTIVITY.md` and start with Part 1!

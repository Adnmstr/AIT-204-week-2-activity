# Student Performance Predictor - In-Class Activity

Build your own linear regression application from scratch! This hands-on activity teaches gradient descent, model evaluation, and Streamlit app development.

## 🎯 What You'll Build

A **Student Performance Prediction System** that:
- Predicts exam scores based on study hours
- Implements gradient descent optimization **manually** (no sklearn!)
- Creates an interactive web application with Streamlit
- Visualizes the learning process with multiple plots
- Allows users to make predictions

**Live Demo**: [See what you'll create](#) *(deploy your own!)*

## 📚 Learning Objectives

By completing this activity, you will:
- ✅ Implement gradient descent from scratch
- ✅ Understand how partial derivatives guide learning
- ✅ Calculate and interpret model metrics (R², MSE, RMSE, MAE)
- ✅ Build interactive visualizations with Plotly
- ✅ Create and deploy a Streamlit web application
- ✅ Practice modular Python code organization

## ⏱️ Time Required

**90-120 minutes** (in-class)

## 📋 Prerequisites

- Basic Python programming
- Understanding of functions and loops
- Familiarity with NumPy arrays
- Basic calculus (derivatives) - will be reinforced during activity

## 🚀 Quick Start

**New to this activity?** Start here:

1. **Read** [`QUICK_START_GUIDE.md`](QUICK_START_GUIDE.md) - 5-minute setup
2. **Follow** [`IN_CLASS_ACTIVITY.md`](IN_CLASS_ACTIVITY.md) - Complete instructions
3. **Reference** `solution_reference/` - Only if stuck! (Instructor-guided)

## 📁 What's Included

```
student-performance-activity/
├── README.md                    # This file - overview
├── IN_CLASS_ACTIVITY.md         # 📘 Detailed activity instructions
├── QUICK_START_GUIDE.md         # 🚀 Fast setup guide
├── INSTRUCTOR_OVERVIEW.md       # 👨‍🏫 Instructor guide (for teachers)
└── solution_reference/          # ⚠️ Reference solutions (instructor-guided)
    ├── app.py                   # Complete Streamlit app
    ├── model.py                 # Gradient descent implementation
    ├── data_generator.py        # Data generation utilities
    ├── requirements.txt         # Dependencies
    └── README.md                # Grading guide for instructors
```

## 🎬 Getting Started

### Step 1: Setup (5 minutes)

```bash
# Create your project directory
mkdir student-performance-predictor
cd student-performance-predictor

# Create project files
touch app.py model.py data_generator.py requirements.txt
```

### Step 2: Install Dependencies (2 minutes)

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

### Step 3: Build Your App (90 minutes)

Follow the instructions in [`IN_CLASS_ACTIVITY.md`](IN_CLASS_ACTIVITY.md)

You'll implement three main components:
1. **Data Generator** (20 min) - Generate synthetic student data
2. **Model** (30 min) - Implement gradient descent
3. **Streamlit App** (40 min) - Create interactive interface

## 📊 What You'll Learn

### Part 1: Data Generation
- Creating synthetic datasets
- Adding realistic noise and outliers
- Understanding data distributions

### Part 2: Gradient Descent
Learn the **core of machine learning** by implementing:
```python
# The fundamental update rule you'll code:
m_new = m_old - learning_rate * gradient_m
b_new = b_old - learning_rate * gradient_b
```

### Part 3: Visualizations
Create plots that show:
- Data distribution
- Model fit quality
- Training progress (cost decreasing!)
- Parameter evolution

### Part 4: Model Evaluation
Calculate and interpret:
- **R²**: How well does the model explain variance?
- **MSE/RMSE**: Average prediction error
- **MAE**: Absolute prediction accuracy

## 🎯 Expected Outcomes

### Minimum Viable Product (MVP)
- Data generation working
- Gradient descent decreasing cost
- Basic Streamlit app with 2-3 plots
- Prediction interface

### Full Implementation
- All 4 required visualizations
- Comprehensive metrics display
- Interactive controls
- Polished explanations

### Bonus (Extra Credit)
- Train/test split
- Comparison with closed-form solution
- Animation of training process
- Deployment to Streamlit Cloud

## 📈 Assessment

**100 points total** (+10 bonus)

| Component | Points |
|-----------|--------|
| Data Generator | 20 |
| Gradient Descent | 30 |
| Metrics Calculation | 15 |
| Visualizations | 20 |
| Interactivity | 10 |
| Code Quality | 5 |
| **Bonus Features** | +10 |

See [`IN_CLASS_ACTIVITY.md`](IN_CLASS_ACTIVITY.md) for detailed rubric.

## 💡 Success Tips

1. **Test as you go** - Don't write everything before testing
2. **Start simple** - Get basic functionality working first
3. **Use print statements** - Debug by seeing values
4. **Check array shapes** - Common source of errors
5. **Ask for help** - Collaborate on concepts, code individually

## 🐛 Common Issues & Solutions

### "Cost is increasing!"
**Problem**: Learning rate too high
**Solution**: Decrease to 0.01 or 0.001

### "R² is negative!"
**Problem**: Gradient calculation error
**Solution**: Check your gradient formulas:
```python
grad_m = (1/n) * np.sum((y_pred - y) * X)
grad_b = (1/n) * np.sum(y_pred - y)
```

### "Array dimension error"
**Problem**: Not handling array shapes
**Solution**: Flatten arrays:
```python
X = np.array(X).flatten()
y = np.array(y).flatten()
```

More in [`QUICK_START_GUIDE.md`](QUICK_START_GUIDE.md)

## 🚀 Deployment

Once your app works locally, deploy it!

### Deploy to Streamlit Cloud

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Student Performance Predictor"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Deploy**:
   - Go to https://share.streamlit.io
   - Connect GitHub
   - Select your repo
   - Deploy!

3. **Share** your live app URL!

## 📚 Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **NumPy Tutorial**: https://numpy.org/doc/stable/user/quickstart.html
- **Plotly Examples**: https://plotly.com/python/
- **Demo App**: See the companion demo for reference

## 🎓 For Instructors

See [`INSTRUCTOR_OVERVIEW.md`](INSTRUCTOR_OVERVIEW.md) for:
- Teaching guide and timing
- Common student issues
- Grading rubric details
- Solution reference notes
- Assessment strategies

## ❓ FAQ

**Q: Can we use sklearn?**
A: For comparison only. Core gradient descent must be manual.

**Q: Can we work in pairs?**
A: Yes! But both must understand all code.

**Q: What if we don't finish?**
A: Submit what you have. Partial credit given.

**Q: How do we know it's working?**
A: Your cost should steadily decrease!

## 🎉 Student Showcase

After completing this activity, you'll have:
- ✅ A deployed web application
- ✅ Understanding of gradient descent internals
- ✅ Portfolio project to share
- ✅ Real ML implementation experience

## 📝 Submission

Submit:
1. GitHub repository link
2. Deployed app URL (optional but encouraged)
3. README with:
   - How to run your app
   - Findings from experiments
   - Answers to checkpoint questions
   - Challenges faced and solutions

## 🌟 Next Steps

After completing this activity:
- Extend to multiple features
- Try different scenarios (housing prices, sales forecasting)
- Add advanced features (regularization, feature scaling)
- Apply to real datasets

---

## 🚀 Ready to Build?

1. Open [`QUICK_START_GUIDE.md`](QUICK_START_GUIDE.md) for setup
2. Follow [`IN_CLASS_ACTIVITY.md`](IN_CLASS_ACTIVITY.md) for instructions
3. Build your app!
4. Deploy and share!

**Let's learn by doing! 💪**

---

*Part of AIT-204: AI & Machine Learning course materials*

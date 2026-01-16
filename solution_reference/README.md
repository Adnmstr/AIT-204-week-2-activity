# Solution Reference - Student Performance Predictor

## ⚠️ INSTRUCTOR USE ONLY

This directory contains the complete reference solution for the in-class activity. **Do not share these files with students before they complete the activity.**

## Purpose

Use these files to:
- Grade student submissions
- Help students who are stuck (show relevant parts, not the whole solution)
- Verify expected behavior
- Prepare for class (test the activity yourself)

## Files

- `data_generator.py`: Complete implementation of data generation with outliers
- `model.py`: Full gradient descent implementation with all methods
- `app.py`: Complete Streamlit application with all sections and visualizations
- `requirements.txt`: Python dependencies

## Running the Solution

```bash
# From the solution_reference directory
pip install -r requirements.txt
streamlit run app.py
```

The app should run on `http://localhost:8501`

## Key Implementation Details

### Data Generator (`data_generator.py`)
- Generates study hours uniformly between 2-10 hours
- Uses relationship: Score = 40 + 6 × hours + noise
- Adds three types of outliers: high, low, extreme
- Clips scores to valid range [0, 100]

### Model (`model.py`)
- Implements gradient descent from scratch
- Properly handles numpy array dimensions
- Stores complete training history
- Calculates standard metrics: R², MSE, RMSE, MAE

### Streamlit App (`app.py`)
- 4 main sections: Data, Training, Visualizations, Predictions
- 6 interactive plots
- Proper use of Streamlit components
- Clear explanations and LaTeX formatting

## Common Student Issues

### Issue 1: Diverging Cost Function
**Symptom**: Cost increases instead of decreasing
**Cause**: Learning rate too high
**Solution**: Decrease learning rate to 0.01 or lower

### Issue 2: Negative R²
**Symptom**: R² score is negative
**Cause**: Model implementation error (usually in gradient computation)
**Solution**: Check gradient formulas match the mathematical equations

### Issue 3: Array Dimension Errors
**Symptom**: "operands could not be broadcast together" errors
**Cause**: Not flattening arrays or incorrect dimension handling
**Solution**: Use `.flatten()` on input arrays in fit() and predict()

### Issue 4: History Not Updating
**Symptom**: Plots don't show progression
**Cause**: Forgot to append to history lists in training loop
**Solution**: Ensure history is updated at every iteration

### Issue 5: Slow Convergence
**Symptom**: Model doesn't converge in max_iterations
**Cause**: Learning rate too small or max_iterations too low
**Solution**: Increase learning rate to 0.01-0.05 or increase max_iterations

## Grading Tips

### Data Generator (20 points)
- ✓ Correct data generation (5 pts)
- ✓ Noise implementation (5 pts)
- ✓ Outlier injection (5 pts)
- ✓ Proper clipping to [0, 100] (5 pts)

### Gradient Descent Implementation (30 points)
- ✓ Correct gradient computation (10 pts)
- ✓ Proper parameter updates (10 pts)
- ✓ History tracking (5 pts)
- ✓ Convergence (cost decreases) (5 pts)

### Metrics (15 points)
- ✓ R² calculation (5 pts)
- ✓ MSE/RMSE calculation (5 pts)
- ✓ MAE calculation (5 pts)

### Visualizations (20 points)
- ✓ Scatter plot with data (5 pts)
- ✓ Regression line plot (5 pts)
- ✓ Cost convergence plot (5 pts)
- ✓ At least one additional plot (5 pts)

### Interactivity (10 points)
- ✓ Sidebar controls (5 pts)
- ✓ Prediction interface (5 pts)

### Code Quality (5 points)
- ✓ Clean, readable code (2 pts)
- ✓ Comments explaining logic (2 pts)
- ✓ Proper function organization (1 pt)

## Expected Behavior

With default settings (n=100, noise=8, outliers=5%, lr=0.01, iter=500):
- **R²**: ~0.75-0.85 (varies due to random seed)
- **RMSE**: ~8-12 points
- **Cost**: Decreases smoothly from ~1500-2000 to ~300-500
- **Parameters**: Slope ~5.5-6.5, Intercept ~38-42
- **Convergence**: Within 200-300 iterations typically

## Testing Checklist

Use this to verify student submissions:

- [ ] App runs without errors
- [ ] Data generates correctly with outliers visible
- [ ] Cost function decreases (doesn't increase or oscillate)
- [ ] Regression line fits the data reasonably
- [ ] R² is positive and reasonable (0.5-0.9)
- [ ] Sidebar controls update the app
- [ ] Predictions work for different input values
- [ ] All plots are labeled and clear
- [ ] Code is organized in separate files

## Advanced Extensions to Look For

Students who complete bonus challenges:
- Train/test split with performance comparison
- Closed-form solution comparison
- Animation slider showing training progress
- Multiple learning rate comparison
- Feature normalization/scaling
- Batch/mini-batch gradient descent variants

## Deployment Verification

If students deploy to Streamlit Cloud:
- [ ] URL is publicly accessible
- [ ] App loads without errors
- [ ] All functionality works in deployed version
- [ ] requirements.txt is correct
- [ ] GitHub repository is properly structured

## Questions Students Might Ask

**Q: Can we use sklearn?**
A: Only for comparison/validation. Core implementation must be manual.

**Q: How do we know if our gradient descent is correct?**
A: Check if: (1) cost decreases, (2) parameters converge to reasonable values, (3) R² is positive

**Q: What if our R² is 0.4?**
A: That's fine with high noise/outliers. The relationship is still valid.

**Q: Should we normalize features?**
A: Not required, but bonus points if they implement it!

**Q: Can we add more features?**
A: Bonus challenge! Must extend to multiple linear regression.

## Time Management

If students are struggling with time:
- **Priority 1** (must have): Working gradient descent and basic plot
- **Priority 2** (should have): Metrics and regression line plot
- **Priority 3** (nice to have): Additional visualizations and predictions
- **Priority 4** (bonus): Advanced features and deployment

## Support Strategy

When students ask for help:
1. **First**: Ask them to explain what they're trying to do
2. **Then**: Point them to relevant documentation/demos
3. **If stuck**: Show them the *specific function* they need (not the whole solution)
4. **Last resort**: Work through the logic together, but make them type

## Assessment Notes

- Focus on **understanding over perfection**
- Partial credit for attempted implementations
- Bonus points for creative extensions
- Negative points only for plagiarism or not following requirements

---

Good luck facilitating this activity! 🚀

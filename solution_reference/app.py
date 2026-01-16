"""
Reference Solution: Student Performance Predictor Streamlit App
Instructor Use Only

This is a complete working example that students should build themselves.
Use this for grading reference and helping stuck students.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data_generator import StudentDataGenerator
from model import LinearRegressionGD


# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="wide"
)

# Title
st.title("📚 Student Performance Prediction System")
st.markdown("""
This application predicts final exam scores based on daily study hours using linear regression.
The model learns the relationship between study time and performance using gradient descent optimization.
""")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("Data Settings")
n_students = st.sidebar.slider("Number of students", 50, 200, 100, 10)
noise_level = st.sidebar.slider("Noise level (std)", 0, 15, 8, 1)
outlier_pct = st.sidebar.slider("Outlier percentage (%)", 0, 20, 5, 1)
seed = st.sidebar.number_input("Random seed", 0, 1000, 42)

st.sidebar.subheader("Training Settings")
learning_rate = st.sidebar.slider("Learning rate", 0.001, 0.1, 0.01, 0.001, format="%.3f")
max_iterations = st.sidebar.slider("Max iterations", 100, 2000, 500, 100)

if st.sidebar.button("🔄 Regenerate Data"):
    st.rerun()


# ============================================================================
# SECTION 1: Generate and Display Data
# ============================================================================

st.header("1️⃣ Student Performance Data")

# Generate data
data_gen = StudentDataGenerator(seed=seed)
study_hours, exam_scores = data_gen.generate_data(
    n_samples=n_students,
    noise_std=noise_level,
    outlier_pct=outlier_pct
)

# Get statistics
stats = data_gen.get_statistics(study_hours, exam_scores)

# Display statistics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Students", stats['n_students'])
with col2:
    st.metric("Avg Study Hours", f"{stats['mean_study_hours']:.1f}h")
with col3:
    st.metric("Avg Exam Score", f"{stats['mean_exam_score']:.1f}")
with col4:
    st.metric("Correlation", f"{stats['correlation']:.3f}")

# Display data table
df = data_gen.create_dataframe(study_hours, exam_scores)

with st.expander("📊 View Raw Data"):
    st.dataframe(df.head(10), use_container_width=True)
    st.download_button(
        "Download Full Dataset (CSV)",
        df.to_csv(index=False),
        "student_performance.csv",
        "text/csv"
    )

# Scatter plot
fig_scatter = px.scatter(
    df,
    x='Study Hours',
    y='Exam Score',
    title='Study Hours vs Exam Score',
    labels={'Study Hours': 'Daily Study Hours', 'Exam Score': 'Final Exam Score'}
)
fig_scatter.update_traces(marker=dict(size=10, opacity=0.6))
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown(f"""
**Data Insights**:
- **Correlation**: {stats['correlation']:.3f} indicates a {'strong' if abs(stats['correlation']) > 0.7 else 'moderate' if abs(stats['correlation']) > 0.4 else 'weak'} positive relationship
- **Score range**: {stats['min_score']:.1f} to {stats['max_score']:.1f} points
- **Outliers**: ~{outlier_pct}% of students performed unexpectedly (notice points far from the trend)
""")


# ============================================================================
# SECTION 2: Train Model
# ============================================================================

st.header("2️⃣ Model Training")

st.markdown(f"""
Training a linear regression model with gradient descent to find the best line that fits the data.

**Current Settings**:
- Learning Rate: `{learning_rate}`
- Max Iterations: `{max_iterations}`
""")

# Train the model
model = LinearRegressionGD(learning_rate=learning_rate, max_iterations=max_iterations)

with st.spinner("Training model..."):
    model.fit(study_hours, exam_scores)

# Get results
params = model.get_parameters()
metrics = model.calculate_metrics(study_hours, exam_scores)
history = model.get_history()

# Display results
st.subheader("📐 Trained Model")

col1, col2 = st.columns(2)
with col1:
    st.metric(
        "Slope (m)",
        f"{params['slope']:.3f}",
        help="Points gained per additional study hour"
    )
with col2:
    st.metric(
        "Intercept (b)",
        f"{params['intercept']:.2f}",
        help="Expected score with zero study hours"
    )

st.markdown(f"""
### Model Equation

$$\\text{{Exam Score}} = {params['slope']:.3f} \\times \\text{{Study Hours}} + {params['intercept']:.2f}$$

**Interpretation**: Each additional hour of daily study increases the expected exam score by **{params['slope']:.2f} points**.
""")

# Display metrics
st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("R² Score", f"{metrics['r2']:.4f}")
with col2:
    st.metric("MSE", f"{metrics['mse']:.2f}")
with col3:
    st.metric("RMSE", f"{metrics['rmse']:.2f}")
with col4:
    st.metric("MAE", f"{metrics['mae']:.2f}")

st.markdown(f"""
**Performance Analysis**:
- **R² = {metrics['r2']:.4f}**: The model explains **{metrics['r2']*100:.1f}%** of the variance in exam scores
- **RMSE = {metrics['rmse']:.2f}**: Average prediction error is about ±{metrics['rmse']:.1f} points
- {'✅ Excellent fit!' if metrics['r2'] > 0.8 else '✅ Good fit!' if metrics['r2'] > 0.6 else '⚠️ Moderate fit - try adjusting hyperparameters'}
""")


# ============================================================================
# SECTION 3: Visualizations
# ============================================================================

st.header("3️⃣ Training Visualizations")

# Plot 1: Regression fit
st.subheader("Fitted Regression Line")

predictions = model.predict(study_hours)

fig_fit = go.Figure()
fig_fit.add_trace(go.Scatter(
    x=study_hours,
    y=exam_scores,
    mode='markers',
    name='Actual Data',
    marker=dict(size=8, opacity=0.6, color='blue')
))
fig_fit.add_trace(go.Scatter(
    x=study_hours,
    y=predictions,
    mode='markers',
    name='Predictions',
    marker=dict(size=6, opacity=0.8, color='red', symbol='x')
))

# Add best fit line
x_line = np.linspace(study_hours.min(), study_hours.max(), 100)
y_line = model.predict(x_line)
fig_fit.add_trace(go.Scatter(
    x=x_line,
    y=y_line,
    mode='lines',
    name='Best Fit Line',
    line=dict(color='red', width=3)
))

fig_fit.update_layout(
    xaxis_title='Study Hours',
    yaxis_title='Exam Score',
    hovermode='closest'
)
st.plotly_chart(fig_fit, use_container_width=True)

# Plot 2 & 3: Cost convergence and residuals
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cost Function Convergence")
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(
        x=history['iteration'],
        y=history['cost'],
        mode='lines',
        line=dict(color='blue', width=2)
    ))
    fig_cost.update_layout(
        xaxis_title='Iteration',
        yaxis_title='Cost (MSE/2)',
        height=400
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    st.markdown(f"""
    **Final Cost**: {history['cost'][-1]:.2f}

    The cost decreases smoothly, indicating successful convergence.
    """)

with col2:
    st.subheader("Residual Plot")
    residuals = model.calculate_residuals(study_hours, exam_scores)

    fig_residuals = go.Figure()
    fig_residuals.add_trace(go.Scatter(
        x=predictions,
        y=residuals,
        mode='markers',
        marker=dict(size=8, opacity=0.6, color='purple')
    ))
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
    fig_residuals.update_layout(
        xaxis_title='Predicted Score',
        yaxis_title='Residual (Actual - Predicted)',
        height=400
    )
    st.plotly_chart(fig_residuals, use_container_width=True)

    st.markdown("""
    **Residuals** should be randomly scattered around zero.
    Patterns indicate model limitations.
    """)

# Plot 4: Parameter evolution
st.subheader("Parameter Evolution During Training")

fig_params = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Slope Evolution", "Intercept Evolution")
)

fig_params.add_trace(
    go.Scatter(
        x=history['iteration'],
        y=history['slope'],
        mode='lines',
        name='Slope',
        line=dict(color='green', width=2)
    ),
    row=1, col=1
)

fig_params.add_trace(
    go.Scatter(
        x=history['iteration'],
        y=history['intercept'],
        mode='lines',
        name='Intercept',
        line=dict(color='orange', width=2)
    ),
    row=1, col=2
)

fig_params.update_xaxes(title_text="Iteration", row=1, col=1)
fig_params.update_xaxes(title_text="Iteration", row=1, col=2)
fig_params.update_yaxes(title_text="Slope Value", row=1, col=1)
fig_params.update_yaxes(title_text="Intercept Value", row=1, col=2)
fig_params.update_layout(height=400, showlegend=False)

st.plotly_chart(fig_params, use_container_width=True)

st.markdown("""
Watch how parameters start from zero and converge to optimal values. The curves flatten when gradients approach zero.
""")


# ============================================================================
# SECTION 4: Interactive Predictions
# ============================================================================

st.header("4️⃣ Make Predictions")

st.markdown("""
Use the trained model to predict exam scores for different study hour values.
""")

col1, col2 = st.columns([1, 2])

with col1:
    study_input = st.number_input(
        "Daily Study Hours",
        min_value=0.0,
        max_value=12.0,
        value=5.0,
        step=0.5,
        help="Enter hours of daily study (0-12)"
    )

    predicted_score = model.predict(np.array([study_input]))[0]

    st.metric("Predicted Exam Score", f"{predicted_score:.1f}")

with col2:
    # Determine message based on score
    if predicted_score >= 90:
        message = "🌟 Excellent! This study schedule should lead to outstanding performance!"
        color = "green"
    elif predicted_score >= 80:
        message = "✅ Great! This amount of study should result in a strong grade."
        color = "blue"
    elif predicted_score >= 70:
        message = "👍 Good! Decent performance expected with this study schedule."
        color = "orange"
    elif predicted_score >= 60:
        message = "⚠️ Passing, but consider increasing study time for better results."
        color = "orange"
    else:
        message = "❌ Warning: This may result in a failing grade. More study time recommended!"
        color = "red"

    st.markdown(f"**Prediction**: With **{study_input:.1f} hours** of daily study, "
                f"you can expect approximately **{predicted_score:.1f} points** on the exam.")

    st.info(message)

# What-if analysis
st.subheader("📈 What-If Analysis")

study_range = np.linspace(0, 12, 25)
predicted_range = model.predict(study_range)

fig_whatif = go.Figure()
fig_whatif.add_trace(go.Scatter(
    x=study_range,
    y=predicted_range,
    mode='lines',
    name='Predicted Score',
    line=dict(color='green', width=3)
))

# Add confidence interval (±RMSE)
fig_whatif.add_trace(go.Scatter(
    x=study_range,
    y=predicted_range + metrics['rmse'],
    mode='lines',
    name='Upper Bound (+RMSE)',
    line=dict(color='lightgreen', width=1, dash='dash'),
    showlegend=False
))
fig_whatif.add_trace(go.Scatter(
    x=study_range,
    y=predicted_range - metrics['rmse'],
    mode='lines',
    name='Lower Bound (-RMSE)',
    line=dict(color='lightgreen', width=1, dash='dash'),
    fill='tonexty',
    fillcolor='rgba(0,255,0,0.1)',
    showlegend=False
))

# Highlight current prediction
fig_whatif.add_trace(go.Scatter(
    x=[study_input],
    y=[predicted_score],
    mode='markers',
    name='Your Prediction',
    marker=dict(size=15, color='red', symbol='star')
))

fig_whatif.update_layout(
    title='Expected Score vs Study Hours',
    xaxis_title='Daily Study Hours',
    yaxis_title='Predicted Exam Score',
    yaxis_range=[0, 100]
)
st.plotly_chart(fig_whatif, use_container_width=True)

st.markdown(f"""
The shaded region represents the uncertainty in predictions (±{metrics['rmse']:.1f} points).
Your prediction is marked with a red star.
""")


# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
### 🎓 Key Takeaways

This application demonstrates:
- **Linear Regression**: Modeling relationships between variables
- **Gradient Descent**: Iterative optimization to find best parameters
- **Model Evaluation**: Using metrics (R², MSE, RMSE, MAE) to assess performance
- **Interpretability**: Understanding what the model tells us about study and performance

**Remember**: This is a simplified model. Real-world academic performance depends on many factors
beyond just study time (quality of study, prior knowledge, sleep, nutrition, stress, etc.).
""")

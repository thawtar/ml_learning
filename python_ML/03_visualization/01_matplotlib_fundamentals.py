"""
Module 3 — Lesson 1: Matplotlib Fundamentals
==============================================
Matplotlib is the base plotting library in Python. Everything else
(Seaborn, Pandas plots) is built on top of it. Understanding the
Figure/Axes API gives you full control.

Key idea: Figure contains Axes. Axes is where you plot.
    fig, ax = plt.subplots()   ← this is the canonical pattern
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")    # non-interactive backend (for scripts)
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════
# 1. THE FIGURE/AXES API
# ══════════════════════════════════════════════════════════════════════

# Always use the object-oriented API (not plt.plot())
fig, ax = plt.subplots(figsize=(8, 5))

x = np.linspace(0, 2 * np.pi, 100)
ax.plot(x, np.sin(x), label="sin(x)", color="steelblue", linewidth=2)
ax.plot(x, np.cos(x), label="cos(x)", color="coral", linewidth=2, linestyle="--")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Sine and Cosine Waves")
ax.legend()
ax.grid(True, alpha=0.3)

fig.savefig("01_line_plot.png", dpi=150, bbox_inches="tight")
print("Saved: 01_line_plot.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 2. COMMON PLOT TYPES
# ══════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(42)

# ── Scatter plot ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
x = rng.normal(0, 1, 200)
y = 0.5 * x + rng.normal(0, 0.3, 200)
colors = rng.uniform(0, 1, 200)

scatter = ax.scatter(x, y, c=colors, cmap="viridis", alpha=0.7, s=30)
ax.set_xlabel("Feature X")
ax.set_ylabel("Feature Y")
ax.set_title("Scatter Plot with Color Mapping")
fig.colorbar(scatter, ax=ax, label="Value")
fig.savefig("02_scatter_plot.png", dpi=150, bbox_inches="tight")
print("Saved: 02_scatter_plot.png")
plt.close()


# ── Bar chart ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
models = ["LogReg", "RF", "XGB", "SVM", "KNN"]
accuracies = [0.82, 0.89, 0.91, 0.85, 0.78]
colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

bars = ax.bar(models, accuracies, color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Accuracy")
ax.set_title("Model Comparison")
ax.set_ylim(0.7, 1.0)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{acc:.2f}", ha="center", va="bottom", fontsize=10)

fig.savefig("03_bar_chart.png", dpi=150, bbox_inches="tight")
print("Saved: 03_bar_chart.png")
plt.close()


# ── Histogram ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
data = rng.normal(100, 15, 1000)

ax.hist(data, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
ax.axvline(data.mean(), color="red", linestyle="--", label=f"Mean: {data.mean():.1f}")
ax.axvline(np.median(data), color="orange", linestyle="--", label=f"Median: {np.median(data):.1f}")
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Scores")
ax.legend()
fig.savefig("04_histogram.png", dpi=150, bbox_inches="tight")
print("Saved: 04_histogram.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 3. SUBPLOTS
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Line
axes[0, 0].plot(np.cumsum(rng.standard_normal(100)))
axes[0, 0].set_title("Random Walk")

# Scatter
axes[0, 1].scatter(rng.normal(0, 1, 50), rng.normal(0, 1, 50), alpha=0.6)
axes[0, 1].set_title("Random Points")

# Bar
axes[1, 0].barh(["A", "B", "C", "D"], [25, 40, 30, 55], color="teal")
axes[1, 0].set_title("Horizontal Bars")

# Histogram
axes[1, 1].hist(rng.exponential(2, 500), bins=25, color="salmon", edgecolor="white")
axes[1, 1].set_title("Exponential Distribution")

fig.suptitle("Four Plot Types", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig("05_subplots.png", dpi=150, bbox_inches="tight")
print("Saved: 05_subplots.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 4. STYLING & CUSTOMIZATION
# ══════════════════════════════════════════════════════════════════════

# Available styles
print("\nAvailable styles:", plt.style.available[:10], "...")

# Use a style
with plt.style.context("seaborn-v0_8-whitegrid"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(5):
        ax.plot(np.cumsum(rng.standard_normal(100)), label=f"Series {i+1}")
    ax.legend(loc="upper left")
    ax.set_title("Styled Line Chart")
    fig.savefig("06_styled.png", dpi=150, bbox_inches="tight")
    print("Saved: 06_styled.png")
    plt.close()

# Custom colors and annotations
fig, ax = plt.subplots(figsize=(8, 5))
x = np.linspace(0, 10, 100)
y = np.exp(-0.3 * x) * np.sin(2 * x)
ax.plot(x, y, color="#2c3e50", linewidth=2)
ax.fill_between(x, y, alpha=0.2, color="#3498db")

# Annotate a point
peak_idx = np.argmax(y)
ax.annotate("Peak", xy=(x[peak_idx], y[peak_idx]),
            xytext=(x[peak_idx] + 1.5, y[peak_idx] + 0.2),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=12, color="red")

ax.set_title("Damped Oscillation")
fig.savefig("07_annotated.png", dpi=150, bbox_inches="tight")
print("Saved: 07_annotated.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 5. SAVING FIGURES
# ══════════════════════════════════════════════════════════════════════
# Formats: png (raster), pdf (vector), svg (vector), jpg
# Key parameters:
#   dpi=150        — resolution (150 is good for screen, 300 for print)
#   bbox_inches="tight"  — trim white space
#   transparent=True      — transparent background

print("\nAll plots saved to current directory.")
print("Tip: Use fig.savefig() not plt.savefig() for explicit control.")


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 1.1: Create a figure with 3 subplots (1 row, 3 cols) showing:
    - Normal distribution (μ=0, σ=1)
    - Uniform distribution [0, 1]
    - Exponential distribution (λ=1)
    Each should be a histogram with 50 bins, different colors, and titles.

Exercise 1.2: Plot a "training curve": x = epochs (1-100),
    - y1 = training loss (starts high, decreases with noise)
    - y2 = validation loss (decreases then increases = overfitting)
    Include legend, grid, labels. Mark the "best epoch" with a vertical line.

Exercise 1.3: Create a grouped bar chart comparing 4 models across
    3 metrics (accuracy, precision, recall). Use side-by-side bars.

Exercise 1.4: Recreate a simple version of the Anscombe's quartet
    (4 scatter plots with same statistics but different patterns).
"""

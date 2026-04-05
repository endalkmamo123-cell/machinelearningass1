import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 6))

# Hide axes
ax.axis('off')

# Make boxes
def draw_box(ax, text, xy, width=0.25, height=0.15, facecolor='lightblue'):
    rect = patches.FancyBboxPatch((xy[0]-width/2, xy[1]-height/2), width, height, 
                                  boxstyle="round,pad=0.05", facecolor=facecolor, edgecolor='black')
    ax.add_patch(rect)
    ax.text(xy[0], xy[1], text, ha='center', va='center', fontsize=10, weight='bold')

# Positions
pos_data = (0.2, 0.8)
pos_prep = (0.5, 0.8)
pos_split = (0.8, 0.8)

pos_model_train = (0.5, 0.5)
pos_eval = (0.8, 0.5)

pos_deploy = (0.8, 0.2)
pos_monitor = (0.8, -0.1)  # slightly below

# Draw boxes
draw_box(ax, "1. Data Collection\n(Logs & Flow Data)", pos_data, facecolor='#ffb3ba')
draw_box(ax, "2. Data Preparation\n(Cleaning, Encoding, Scaling)", pos_prep, facecolor='#ffdfba')
draw_box(ax, "3. Data Splitting\n(80% Train, 20% Test)", pos_split, facecolor='#ffffba')

draw_box(ax, "4. Model Train & Tune\n(GridSearchCV on RF & XGBoost)", pos_model_train, facecolor='#baffc9')
draw_box(ax, "5. Model Evaluation\n(ROC-AUC, Confusion Matrix)", pos_eval, facecolor='#bae1ff')

draw_box(ax, "6. Streamlit Deployment\n(Live Intrusion Detection)", pos_deploy, facecolor='#e6b3ff')

# Draw arrows
kwargs = dict(arrowprops=dict(facecolor='black', width=1, headwidth=8, headlength=10, shrink=0.01))

ax.annotate('', xy=(pos_prep[0]-0.15, pos_prep[1]), xytext=(pos_data[0]+0.15, pos_data[1]), **kwargs)
ax.annotate('', xy=(pos_split[0]-0.15, pos_split[1]), xytext=(pos_prep[0]+0.15, pos_prep[1]), **kwargs)
ax.annotate('', xy=(pos_model_train[0], pos_model_train[1]+0.1), xytext=(pos_split[0]-0.15, pos_split[1]-0.1), **kwargs)
ax.annotate('', xy=(pos_eval[0]-0.15, pos_eval[1]), xytext=(pos_model_train[0]+0.15, pos_model_train[1]), **kwargs)
ax.annotate('', xy=(pos_deploy[0], pos_deploy[1]+0.1), xytext=(pos_eval[0], pos_eval[1]-0.1), **kwargs)

plt.title("Cybersecurity Intrusion Detection Pipeline Workflow", fontsize=14, weight='bold', y=0.95)
plt.tight_layout()
plt.savefig('workflow_diagram.png', dpi=300)
print("Saved workflow_diagram.png successfully.")

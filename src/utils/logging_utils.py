import os, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def try_wandb_init(project="cv8501-adni", run_name=None, config=None):
    use = os.environ.get("WANDB", "1") != "0"  # set WANDB=0 to disable
    if not use:
        return None
    try:
        import wandb
        run = wandb.init(project=project, name=run_name, config=config, reinit=True)
        return run
    except Exception as e:
        print(f"[wandb] disabled: {e}")
        return None

def wandb_log(run, metrics: dict, step=None):
    if run is None: return
    try:
        run.log(metrics, step=step)
    except Exception as e:
        print(f"[wandb] log failed: {e}")

def plot_conf_mat(y_true, y_pred, labels=("CN","MCI","AD")):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(4,4), dpi=120)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    return fig, cm

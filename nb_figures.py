"""
Nature Biotechnology-standard figures for genome-ft project.
One chart per PNG. 300 DPI. No overlapping elements.
Run: python nb_figures.py
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.fontsize": 9.5,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Palette ───────────────────────────────────────────────────────────────────
C_ENH   = "#C0392B"   # deep red
C_PRO   = "#1A7A5E"   # deep teal
C_BASE  = "#7F8C8D"   # grey (base model)
C_REF   = "#2C3E50"   # dark (published refs)
C_BEST  = "#E67E22"   # amber (best epoch marker)
C_LIGHT = "#F5F5F5"

# ── Shared data ───────────────────────────────────────────────────────────────
def load(path):
    with open(path) as f: return json.load(f)

enh_s42  = load("metrics_enh_s42.json")
enh_s0   = load("metrics_enh_s0.json")
enh_s123 = load("metrics_enh_s123.json")
pro_s42  = load("metrics_pro_s42.json")

enh_epochs  = enh_s42["epochs"]
pro_epochs  = pro_s42["epochs"]

# Final test results (3 seeds enhancers)
enh_tests = [
    load("metrics_enh_s42.json")["final_test"],
    load("metrics_enh_s0.json")["final_test"],
    load("metrics_enh_s123.json")["final_test"],
]
pro_test = load("metrics_pro_s42.json")["final_test"]

# Base model numbers (from eval report)
try:
    er_enh = load("eval_report_enh.json")
    base_enh = er_enh["base_model"]
    cm_enh = er_enh.get("confusion_matrix", None)
except: base_enh = None; cm_enh = None
try:
    er_pro = load("eval_report_pro.json")
    base_pro = er_pro["base_model"]
    cm_pro = er_pro.get("confusion_matrix", None)
except: base_pro = None; cm_pro = None

print("Data loaded.")

# ── Helper ────────────────────────────────────────────────────────────────────
def save(fig, name):
    fig.savefig(name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {name}")

def styled_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_title(title, pad=14, fontweight="bold")
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

# ── FIG 1 — Enhancers: validation MCC curve (seed 42) ────────────────────────
def fig01():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = [d["epoch"] for d in enh_epochs]
    ys = [d["mcc"]   for d in enh_epochs]
    best_i = int(np.argmax(ys))

    ax.plot(xs, ys, color=C_ENH, linewidth=2, marker="o", markersize=6, zorder=4, label="NT-v2 50M (val MCC)")
    ax.scatter([xs[best_i]], [ys[best_i]], s=80, color=C_BEST, zorder=6, edgecolors="white", linewidths=1.2)
    ax.annotate(f"Best epoch {xs[best_i]}\nMCC = {ys[best_i]:.3f}",
                xy=(xs[best_i], ys[best_i]),
                xytext=(xs[best_i] + 0.3, ys[best_i] - 0.015),
                fontsize=9, color=C_BEST,
                arrowprops=dict(arrowstyle="-", color=C_BEST, lw=0.8))

    ax2 = ax.twinx()
    losses = [d["train_loss"] for d in enh_epochs]
    ax2.plot(xs, losses, color="#95A5A6", linewidth=1.5, linestyle="--", marker="s", markersize=4, label="Train loss")
    ax2.set_ylabel("Training loss", color="#7F8C8D", labelpad=8)
    ax2.tick_params(axis="y", colors="#7F8C8D")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#cccccc")

    styled_ax(ax, "Epoch", "Validation MCC",
              "Enhancers: validation MCC per epoch\nhuman_enhancers_cohn · NT-v2 50M · seed 42")
    ax.set_xticks(xs)
    ax.set_ylim(0.35, 0.55)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    fig.tight_layout()
    save(fig, "nb_fig01_enh_mcc_curve.png")

# ── FIG 2 — Promoters: validation MCC curve ───────────────────────────────────
def fig02():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = [d["epoch"] for d in pro_epochs]
    ys = [d["mcc"]   for d in pro_epochs]
    best_i = int(np.argmax(ys))

    ax.plot(xs, ys, color=C_PRO, linewidth=2, marker="o", markersize=6, zorder=4, label="NT-v2 50M (val MCC)")
    ax.scatter([xs[best_i]], [ys[best_i]], s=80, color=C_BEST, zorder=6, edgecolors="white", linewidths=1.2)
    ax.annotate(f"Best epoch {xs[best_i]}\nMCC = {ys[best_i]:.3f}",
                xy=(xs[best_i], ys[best_i]),
                xytext=(xs[best_i] + 0.3, ys[best_i] - 0.004),
                fontsize=9, color=C_BEST,
                arrowprops=dict(arrowstyle="-", color=C_BEST, lw=0.8))

    ax2 = ax.twinx()
    losses = [d["train_loss"] for d in pro_epochs]
    ax2.plot(xs, losses, color="#95A5A6", linewidth=1.5, linestyle="--", marker="s", markersize=4, label="Train loss")
    ax2.set_ylabel("Training loss", color="#7F8C8D", labelpad=8)
    ax2.tick_params(axis="y", colors="#7F8C8D")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#cccccc")

    styled_ax(ax, "Epoch", "Validation MCC",
              "Promoters: validation MCC per epoch\nhuman_nontata_promoters · NT-v2 50M · seed 42")
    ax.set_xticks(xs)
    ax.set_ylim(0.68, 0.78)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    fig.tight_layout()
    save(fig, "nb_fig02_pro_mcc_curve.png")


# ── FIG 3 — Enhancers vs SotA (bar chart) ────────────────────────────────────
def fig03():
    models = [
        ("Random baseline",           0.500, "#DADADA"),
        ("GB-CNN\n(Gresova 2023)",    0.690, "#B0BEC5"),
        ("DNABERT\n(Ji 2021)",        0.706, "#90A4AE"),
        ("HyenaDNA tiny-1k\n(Nguyen 2023)", 0.740, "#78909C"),
        ("NT-v2 500M\n(Dalla-Torre 2023)",  0.776, "#546E7A"),
        ("DNABERT-2\n(Zhou 2023)",    0.785, "#455A64"),
        ("NT-v2 50M\n(this work)",    np.mean([t["accuracy"] for t in enh_tests]), C_ENH),
    ]
    names  = [m[0] for m in models]
    vals   = [m[1] for m in models]
    colors = [m[2] for m in models]
    std_val = np.std([t["accuracy"] for t in enh_tests])

    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(names))
    bars = ax.barh(y, vals, color=colors, height=0.55, edgecolor="white", linewidth=0.5)

    # Error bar only on our model (last bar)
    ax.barh([y[-1]], [vals[-1]], xerr=[std_val], height=0.55,
            color="none", ecolor="#333333", capsize=4, linewidth=1.2)

    for i, (bar, val, name) in enumerate(zip(bars, vals, names)):
        fw = "bold" if i == len(models)-1 else "normal"
        ax.text(val + 0.004, i, f"{val:.3f}", va="center", fontsize=9.5, fontweight=fw, color="#2C3E50")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0.44, 0.84)
    ax.axvline(0.5, color="#CCCCCC", linewidth=0.8, linestyle=":")
    styled_ax(ax, "Test accuracy", "",
              "Enhancers: NT-v2 50M vs published models\nhuman_enhancers_cohn · 6,948 test sequences")

    note = mpatches.Patch(color=C_ENH, label="This work (mean ± std, n=3 seeds)")
    ref  = mpatches.Patch(color="#78909C", label="Published references (source papers)")
    ax.legend(handles=[note, ref], loc="lower right", fontsize=9)
    ax.text(0.99, -0.06, "Published values not recomputed here",
            transform=ax.transAxes, ha="right", fontsize=7.5, color="#888888", style="italic")
    fig.tight_layout()
    save(fig, "nb_fig03_enh_vs_sota.png")

# ── FIG 4 — Promoters vs SotA (bar chart) ─────────────────────────────────────
def fig04():
    models = [
        ("Random baseline",           0.500, "#DADADA"),
        ("GB-CNN\n(Gresova 2023)",    0.850, "#B0BEC5"),
        ("HyenaDNA tiny-1k\n(Nguyen 2023)", 0.960, "#78909C"),
        ("NT-v2 50M\n(this work)",    pro_test["accuracy"], C_PRO),
    ]
    names  = [m[0] for m in models]
    vals   = [m[1] for m in models]
    colors = [m[2] for m in models]

    fig, ax = plt.subplots(figsize=(6, 4))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, height=0.50, edgecolor="white", linewidth=0.5)
    for i, (val, name) in enumerate(zip(vals, names)):
        fw = "bold" if i == len(models)-1 else "normal"
        ax.text(val + 0.004, i, f"{val:.3f}", va="center", fontsize=9.5, fontweight=fw, color="#2C3E50")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0.44, 1.02)
    ax.axvline(0.5, color="#CCCCCC", linewidth=0.8, linestyle=":")
    styled_ax(ax, "Test accuracy", "",
              "Promoters: NT-v2 50M vs published models\nhuman_nontata_promoters · 9,034 test sequences")

    note = mpatches.Patch(color=C_PRO, label="This work (seed 42)")
    ref  = mpatches.Patch(color="#78909C", label="Published references (source papers)")
    ax.legend(handles=[note, ref], loc="lower right", fontsize=9)
    ax.text(0.99, -0.07, "Published values not recomputed here",
            transform=ax.transAxes, ha="right", fontsize=7.5, color="#888888", style="italic")
    fig.tight_layout()
    save(fig, "nb_fig04_pro_vs_sota.png")


# ── FIG 5 — Multi-seed MCC reproducibility (enhancers) ───────────────────────
def fig05():
    seeds = [42, 0, 123]
    mccs  = [t["mcc"] for t in enh_tests]
    mean_mcc = np.mean(mccs)
    std_mcc  = np.std(mccs)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    x = np.arange(len(seeds))
    bars = ax.bar(x, mccs, color=[C_ENH]*3, width=0.45,
                  edgecolor="white", linewidth=0.5, zorder=3)
    for bar, val in zip(bars, mccs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=C_ENH)

    ax.axhline(mean_mcc, color="#2C3E50", linewidth=1.5, linestyle="--", zorder=4, label=f"Mean = {mean_mcc:.3f}")
    ax.fill_between([-0.5, 2.5], mean_mcc - std_mcc, mean_mcc + std_mcc,
                    alpha=0.12, color="#2C3E50", zorder=2, label=f"±1 SD = {std_mcc:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=11)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.44, 0.52)
    styled_ax(ax, "", "Test MCC",
              "Reproducibility: test MCC across three seeds\nhuman_enhancers_cohn · NT-v2 50M")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save(fig, "nb_fig05_multiseed_mcc.png")

# ── FIG 6 — Base vs fine-tuned lift (both tasks) ─────────────────────────────
def fig06():
    tasks = ["Enhancers\n(Cohn)", "Promoters\n(non-TATA)"]
    base_accs = [
        base_enh["accuracy"] if base_enh else 0.505,
        base_pro["accuracy"] if base_pro else 0.501,
    ]
    ft_accs = [
        np.mean([t["accuracy"] for t in enh_tests]),
        pro_test["accuracy"],
    ]
    ft_stds = [np.std([t["accuracy"] for t in enh_tests]), 0]
    colors  = [C_ENH, C_PRO]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(tasks))
    w = 0.32
    b1 = ax.bar(x - w/2, base_accs, width=w, color="#B0BEC5", label="Base model\n(pretrained, untrained head)",
                edgecolor="white")
    b2 = ax.bar(x + w/2, ft_accs, width=w, color=colors, label="Fine-tuned\n(NT-v2 50M, full fine-tune)",
                edgecolor="white",
                yerr=[ft_stds[0], 0], capsize=4, error_kw=dict(lw=1.2, color="#333"))

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.006,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    for i, (ba, fa) in enumerate(zip(base_accs, ft_accs)):
        lift = fa - ba
        ax.annotate(f"+{lift:.3f}", xy=(x[i] + w/2, fa + 0.018),
                    ha="center", fontsize=9.5, fontweight="bold",
                    color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_ylim(0.0, 1.02)
    styled_ax(ax, "", "Test accuracy",
              "Fine-tuning lift: base model vs fine-tuned\nNT-v2 50M · GenomicBenchmarks")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, "nb_fig06_base_vs_finetuned.png")


# ── FIG 7 — Enhancers confusion matrix ───────────────────────────────────────
def fig07():
    if cm_enh is None:
        print("Skipping fig07: eval_report_enh.json not ready")
        return
    cm = np.array(cm_enh)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    total = cm.sum()
    pct   = cm / total * 100
    im = ax.imshow(pct, cmap="Reds", vmin=0, vmax=50)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}\n({pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=11,
                    color="white" if pct[i,j] > 30 else "#2C3E50", fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nNegative", "Predicted\nPositive"], fontsize=10)
    ax.set_yticklabels(["True\nNegative", "True\nPositive"], fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("% of all test examples", fontsize=9)
    tn,fp,fn,tp = cm[0,0],cm[0,1],cm[1,0],cm[1,1]
    prec = tp/(tp+fp) if tp+fp else 0; rec = tp/(tp+fn) if tp+fn else 0
    spec = tn/(tn+fp) if tn+fp else 0
    ax.set_title(
        f"Enhancers: confusion matrix (best checkpoint)\n"
        f"Precision {prec:.3f}  ·  Recall {rec:.3f}  ·  Specificity {spec:.3f}",
        pad=14, fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "nb_fig07_enh_confusion.png")

# ── FIG 8 — Promoters confusion matrix ───────────────────────────────────────
def fig08():
    if cm_pro is None:
        print("Skipping fig08: eval_report_pro.json not ready")
        return
    cm = np.array(cm_pro)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    total = cm.sum()
    pct   = cm / total * 100
    im = ax.imshow(pct, cmap="Greens", vmin=0, vmax=55)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}\n({pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=11,
                    color="white" if pct[i,j] > 35 else "#2C3E50", fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nNegative", "Predicted\nPositive"], fontsize=10)
    ax.set_yticklabels(["True\nNegative", "True\nPositive"], fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("% of all test examples", fontsize=9)
    tn,fp,fn,tp = cm[0,0],cm[0,1],cm[1,0],cm[1,1]
    prec = tp/(tp+fp) if tp+fp else 0; rec = tp/(tp+fn) if tp+fn else 0
    spec = tn/(tn+fp) if tn+fp else 0
    ax.set_title(
        f"Promoters: confusion matrix (best checkpoint)\n"
        f"Precision {prec:.3f}  ·  Recall {rec:.3f}  ·  Specificity {spec:.3f}",
        pad=14, fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "nb_fig08_pro_confusion.png")


# ── FIG 9 — Pipeline workflow ─────────────────────────────────────────────────
def fig09():
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")

    steps = [
        (1.1, "1. Raw Data",     "GenomicBenchmarks\nHugging Face Hub\n27k–20k sequences",   "#EBF5FB", "#1A5276"),
        (3.3, "2. Preprocessing","Tokenization (6-mer)\nRC augmentation\nTrain/val/test split","#E8F8F5", "#1A7A5E"),
        (5.5, "3. Base Model",   "NT-v2 50M\n(InstaDeep)\nPretrained on\nmulti-species DNA",  "#FEF9E7", "#7D6608"),
        (7.7, "4. Fine-Tuning",  "Full fine-tune\nAdamW lr=1e-5\nWarmup+cosine LR\nGrad clip",  "#FDEDEC", "#922B21"),
        (9.9, "5. Evaluation",   "Val MCC\ncheckpoint selection\nHonest test eval\n(once only)","#F5EEF8", "#6C3483"),
        (12.1,"6. Results",      "Acc 0.735±0.004\nMCC 0.478±0.003\n(enhancers, 3 seeds)",    "#EAFAF1", "#1A7A5E"),
    ]

    box_w, box_h = 1.85, 2.6
    for (cx, title, detail, fc, ec) in steps:
        from matplotlib.patches import FancyBboxPatch
        rect = FancyBboxPatch((cx - box_w/2, 0.7), box_w, box_h,
                               boxstyle="round,pad=0.12", facecolor=fc,
                               edgecolor=ec, linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(cx, 0.7 + box_h - 0.25, title, ha="center", va="top",
                fontsize=10, fontweight="bold", color=ec, zorder=4)
        ax.text(cx, 0.7 + box_h - 0.65, detail, ha="center", va="top",
                fontsize=8, color="#2C3E50", zorder=4, linespacing=1.5)

    arrow_xs = [(2.03, 2.37), (4.23, 4.57), (6.43, 6.77), (8.63, 8.97), (10.83, 11.17)]
    for x0, x1 in arrow_xs:
        ax.annotate("", xy=(x1, 2.0), xytext=(x0, 2.0),
                    arrowprops=dict(arrowstyle="-|>", color="#7F8C8D", lw=1.5, mutation_scale=14), zorder=5)

    ax.set_title("End-to-end fine-tuning pipeline · NT-v2 50M on GenomicBenchmarks",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    save(fig, "nb_fig09_pipeline.png")

# ── FIG 10 — Architecture overview ───────────────────────────────────────────
def fig10():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")

    from matplotlib.patches import FancyBboxPatch

    def box(cx, cy, w, h, fc, ec, title, detail, title_size=9, detail_size=7.5):
        rect = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                               boxstyle="round,pad=0.10", facecolor=fc,
                               edgecolor=ec, linewidth=1.4, zorder=3)
        ax.add_patch(rect)
        ax.text(cx, cy + h/2 - 0.18, title, ha="center", va="top",
                fontsize=title_size, fontweight="bold", color=ec, zorder=4)
        ax.text(cx, cy - 0.05, detail, ha="center", va="center",
                fontsize=detail_size, color="#2C3E50", zorder=4, linespacing=1.4)

    def arr(x0, y0, x1, y1):
        ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                    arrowprops=dict(arrowstyle="-|>", color="#7F8C8D", lw=1.3, mutation_scale=12), zorder=5)

    # Top row: architecture components
    box(1.0, 3.6, 1.7, 1.5, "#EBF5FB","#1A5276",  "DNA Sequence",    "500 bp\nACGT...TGCA", 9)
    arr(1.85, 3.6, 2.25, 3.6)
    box(3.1, 3.6, 1.7, 1.5, "#E8F8F5","#1A7A5E",  "6-mer Tokenizer", "ACGTAC→1042\n~83 tokens/seq", 9)
    arr(3.95, 3.6, 4.35, 3.6)
    box(5.2, 3.6, 1.7, 1.5, "#FEF9E7","#7D6608",  "Token Embeddings","dim 512\n+ position", 9)
    arr(6.05, 3.6, 6.45, 3.6)
    box(7.5, 3.6, 2.0, 1.5, "#FDEDEC","#922B21",  "20× Transformer\nBlocks",  "Self-attention\n20 heads · dim 512\n~52M params", 9)
    arr(8.5, 3.6, 8.9, 3.6)
    box(9.9, 3.6, 1.7, 1.5, "#F5EEF8","#6C3483",  "[CLS] Pooling",  "[1 × 512]\nrepresentation", 9)
    arr(10.75, 3.6, 11.15, 3.6)
    box(12.0, 3.6, 1.7, 1.5, "#FDEDEC","#922B21",  "Classifier Head", "Linear 512→512\nReLU · Linear 512→2\n~1.5k params", 9)

    # Bottom info bar
    info_items = [
        (2.0,  1.2, "#EBF5FB","#1A5276", "Total params", "53.8M"),
        (5.0,  1.2, "#E8F8F5","#1A7A5E", "Fine-tune method", "Full (all params)"),
        (8.0,  1.2, "#FEF9E7","#7D6608", "Optimizer", "AdamW · lr=1e-5\nweight_decay=0.01"),
        (11.0, 1.2, "#F5EEF8","#6C3483", "Hardware", "Apple M4 Pro\n48 GB · MPS"),
    ]
    for (cx, cy, fc, ec, t, d) in info_items:
        box(cx, cy, 2.6, 1.2, fc, ec, t, d, 9, 8.5)

    ax.set_title("Model architecture: Nucleotide Transformer v2 50M · Full fine-tuning",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    save(fig, "nb_fig10_architecture.png")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig01(); fig02(); fig03(); fig04(); fig05()
    fig06(); fig07(); fig08(); fig09(); fig10()
    print("\nAll 10 figures complete.")

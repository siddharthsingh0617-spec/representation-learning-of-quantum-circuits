
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, os, numpy as np

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})
C = {"product": "#2563EB", "entangled": "#DC2626",
     "rbf_svm": "#16A34A",  "mlp": "#9333EA"}
RUN_IDS    = ["iris_q4_l2", "wine_q4_l2"]
DS_LABELS  = ["Iris", "Wine"]
os.makedirs("figures", exist_ok=True)


def load(run_id, name):
    with open(f"results/{run_id}/{name}.json") as f:
        return json.load(f)

def merge(run_id, prefix):
  
    out = {"product": [], "entangled": []}
    for fname in sorted(os.listdir(f"results/{run_id}")):
        if fname.startswith(prefix + "_seeds_"):
            d = load(run_id, fname[:-5])
            for k in out:
                out[k].extend(d[k])
    return out

def mn(lst, key):
    return float(np.mean([r[key] for r in lst]))

def sd(lst, key):
    return float(np.std([r[key] for r in lst]))


def fig1():
    bar_map = {"feature_map_product": "FM-Product",
               "feature_map_entangled": "FM-Entangled",
               "ansatz_product": "Ansatz-Product",
               "ansatz_entangled": "Ansatz-Entangled"}
    bar_cols = ["#93C5FD", "#2563EB", "#FCA5A5", "#DC2626"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, run_id, ds in zip(axes, RUN_IDS, DS_LABELS):
        m = load(run_id, "manip")
        keys = list(bar_map.keys())
        means = [m[k][0] for k in keys]
        stds  = [m[k][1] for k in keys]
        bars  = ax.bar(range(4), means, yerr=stds, capsize=5,
                       color=bar_cols, edgecolor="white", alpha=0.9)
        ax.set_xticks(range(4))
        ax.set_xticklabels([bar_map[k] for k in keys],
                            rotation=18, ha="right", fontsize=9)
        ax.set_ylabel("VN Entropy (nats)")
        ax.set_title(f"Entanglement Check – {ds}")
        for b, v in zip(bars, means):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Figure 1: Manipulation Check – Circuit Entanglement Entropy",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/fig1_manip.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig1 saved")


def fig2():
    model_names = ["QK-SVM\nProduct", "QK-SVM\nEntangled",
                   "QNN\nProduct",    "QNN\nEntangled",
                   "RBF-SVM", "MLP"]
    mcols = [C["product"], C["entangled"],
             C["product"], C["entangled"],
             C["rbf_svm"], C["mlp"]]
    x = np.arange(len(model_names))
    w = 0.35
    offsets = [-w/2, w/2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax_i, (metric, qk_key, qn_key, mlabel) in enumerate([
        ("accuracy", "accuracy", "test_accuracy", "Accuracy"),
        ("auc",      "auc",      "test_auc",      "ROC-AUC"),
    ]):
        ax = axes[ax_i]
        for ds_i, (run_id, ds) in enumerate(zip(RUN_IDS, DS_LABELS)):
            ksvm = merge(run_id, "kernel_svm")
            qnn  = merge(run_id, "qnn")
            cls  = load(run_id, "classical_baselines")
            means = [mn(ksvm["product"],   qk_key),
                     mn(ksvm["entangled"], qk_key),
                     mn(qnn["product"],    qn_key),
                     mn(qnn["entangled"],  qn_key),
                     cls["rbf_svm"][metric],
                     cls["mlp"][metric]]
            stds  = [sd(ksvm["product"],   qk_key),
                     sd(ksvm["entangled"], qk_key),
                     sd(qnn["product"],    qn_key),
                     sd(qnn["entangled"],  qn_key),
                     0, 0]
            hatch = "//" if ds_i == 1 else ""
            for i, (mean, std, col) in enumerate(zip(means, stds, mcols)):
                ax.bar(x[i]+offsets[ds_i], mean, w*0.88,
                       yerr=std, capsize=4, color=col, alpha=0.85,
                       hatch=hatch, edgecolor="white",
                       label=ds if i == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=9)
        ax.set_ylabel(mlabel)
        ax.set_ylim(0, 1.22)
        ax.set_title(f"Test {mlabel}")
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.4)
        if ax_i == 0:
            handles = [mpatches.Patch(facecolor="white", edgecolor="black",
                                       hatch="",  label="Iris"),
                       mpatches.Patch(facecolor="white", edgecolor="black",
                                       hatch="//", label="Wine")]
            ax.legend(handles=handles, loc="lower right")
    fig.suptitle("Figure 2: Performance Comparison – All Models & Datasets",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/fig2_perf_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig2 saved")


def fig3():
    N = 5  # n_seeds
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for row, (run_id, ds) in enumerate(zip(RUN_IDS, DS_LABELS)):
        ss = load(run_id, "cka_self")
        for col, (cond, label) in enumerate([
            ("product",   "Non-Entangled (Product)"),
            ("entangled", "Entangled"),
        ]):
            ax = axes[row][col]
            mat = np.array(ss[cond]["cka_matrix_per_layer"])[-1]
            mean_cka = np.array(ss[cond]["mean_self_cka_per_layer"])
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
            ax.set_xticks(range(N)); ax.set_yticks(range(N))
            ax.set_xticklabels([f"S{i}" for i in range(N)])
            ax.set_yticklabels([f"S{i}" for i in range(N)])
            for i in range(N):
                for j in range(N):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center",
                            va="center", fontsize=8,
                            color="white" if mat[i,j] > 0.6 else "black")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{ds} – {label}\n(Final-layer CKA, seed×seed)")
            # inset: mean CKA by layer
            ins = ax.inset_axes([0.60, 0.04, 0.37, 0.35])
            ins.plot(range(len(mean_cka)), mean_cka, "ko-", markersize=4)
            ins.set_ylim(0, 1.05); ins.set_xticks(range(len(mean_cka)))
            ins.set_xticklabels([f"L{i}" for i in range(len(mean_cka))],
                                  fontsize=6)
            ins.set_ylabel("CKA", fontsize=6); ins.tick_params(labelsize=5)
            ins.set_title("per layer", fontsize=6)
    fig.suptitle("Figure 3: Self-Similarity Sanity Check\n"
                 "(CKA between same-architecture models across random seeds)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/fig3_self_sim.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig3 saved")

def fig4():
    metrics_cfg = [
        ("linear_cka",    "Linear CKA",  "#2563EB"),
        ("rbf_cka",       "RBF CKA",     "#DC2626"),
        ("linear_reg_r2", "Linear R²",   "#16A34A"),
        ("cca_mean_corr", "CCA ρ̄",       "#9333EA"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, run_id, ds in zip(axes, RUN_IDS, DS_LABELS):
        records = load(run_id, "cka_cross")
        layers = sorted(set(r["layer"] for r in records))
        for mkey, mlab, col in metrics_cfg:
            layer_vals = {l: [] for l in layers}
            for r in records:
                layer_vals[r["layer"]].append(r[mkey])
            means = [np.mean(layer_vals[l]) for l in layers]
            stds  = [np.std(layer_vals[l])  for l in layers]
            ax.plot(layers, means, "o-", color=col, linewidth=2,
                    label=mlab)
            ax.fill_between(layers,
                             np.subtract(means, stds),
                             np.add(means, stds),
                             alpha=0.12, color=col)
        ax.set_xlabel("Layer Depth")
        ax.set_ylabel("Similarity Score")
        ax.set_title(f"{ds}: Product vs. Entangled Similarity")
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_ylim(-0.05, 1.1)
        ax.legend(loc="lower left")
    fig.suptitle("Figure 4: Cross-Condition Representation Similarity\n"
                 "(mean ± std across 5 random seeds)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/fig4_cross_cond.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig4 saved")


def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, run_id, ds in zip(axes, RUN_IDS, DS_LABELS):
        tvr = load(run_id, "cka_trained_random")
        for cond, label, ls, mk in [
            ("product",   "Non-Entangled", "--", "o"),
            ("entangled", "Entangled",     "-",  "s"),
        ]:
            data  = tvr[cond]
            means = [d[0] for d in data]
            stds  = [d[1] for d in data]
            layers = list(range(len(means)))
            ax.plot(layers, means, f"{mk}{ls}", color=C[cond],
                    linewidth=2.5, markersize=7, label=label)
            ax.fill_between(layers,
                             np.subtract(means, stds),
                             np.add(means, stds),
                             alpha=0.14, color=C[cond])
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Layer Depth")
        ax.set_ylabel("Linear CKA (trained vs. random)")
        ax.set_title(f"{ds}: Trained vs. Random Representations")
        ax.set_ylim(-0.05, 1.15)
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.legend()
    fig.suptitle("Figure 5: How Much Does Training Reorganise Representations?\n"
                 "(CKA between trained model and random-parameter circuits)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/fig5_trained_vs_random.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig5 saved")


def fig6():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, run_id, ds in zip(axes, RUN_IDS, DS_LABELS):
        qnn = merge(run_id, "qnn")
        for cond, label in [("product","Non-Entangled"),("entangled","Entangled")]:
            curves = [r["loss_curve"] for r in qnn[cond] if r.get("loss_curve")]
            if not curves: continue
            L = max(len(c) for c in curves)
            arr = np.array([c + [c[-1]]*(L-len(c)) for c in curves])
            mean = arr.mean(0); std = arr.std(0)
            ep = np.arange(L)
            ax.plot(ep, mean, linewidth=2, label=label, color=C[cond])
            ax.fill_between(ep, mean-std, mean+std, alpha=0.15, color=C[cond])
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.set_title(f"{ds} – QNN Training Loss")
        ax.set_yscale("log"); ax.legend()
    fig.suptitle("Figure 6: QNN Training Loss (mean ± std across 5 seeds)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/fig6_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig6 saved")


def fig7():
    rows = []
    for run_id, ds in zip(RUN_IDS, DS_LABELS):
        ksvm = merge(run_id, "kernel_svm")
        qnn  = merge(run_id, "qnn")
        cls  = load(run_id, "classical_baselines")
        def row(name, vals, ak, uk):
            return [ds, name, mn(vals,ak), sd(vals,ak), mn(vals,uk), sd(vals,uk)]
        rows += [
            row("QK-SVM Product",   ksvm["product"],   "accuracy","auc"),
            row("QK-SVM Entangled", ksvm["entangled"], "accuracy","auc"),
            row("QNN Product",      qnn["product"],    "test_accuracy","test_auc"),
            row("QNN Entangled",    qnn["entangled"],  "test_accuracy","test_auc"),
            [ds,"RBF-SVM",cls["rbf_svm"]["accuracy"],0,cls["rbf_svm"]["auc"],0],
            [ds,"MLP",    cls["mlp"]["accuracy"],0,    cls["mlp"]["auc"],0],
        ]
    cell = [[r[0],r[1],f"{r[2]:.3f}",f"±{r[3]:.3f}",f"{r[4]:.3f}",f"±{r[5]:.3f}"]
            for r in rows]
    cols = ["Dataset","Model","Acc (mean)","Acc (±std)","AUC (mean)","AUC (±std)"]
    row_colors = {"QK-SVM Product":"#DBEAFE","QK-SVM Entangled":"#FEE2E2",
                  "QNN Product":"#DBEAFE","QNN Entangled":"#FEE2E2",
                  "RBF-SVM":"#F0FDF4","MLP":"#F5F3FF"}
    fig, ax = plt.subplots(figsize=(13, 0.5*len(rows)+1.8))
    ax.axis("off")
    tbl = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.65)
    for i,r in enumerate(cell):
        bg = row_colors.get(r[1], "white")
        for j in range(len(cols)):
            tbl[i+1,j].set_facecolor(bg)
    for j in range(len(cols)):
        tbl[0,j].set_facecolor("#1E3A5F")
        tbl[0,j].set_text_props(color="white", fontweight="bold")
    fig.suptitle("Figure 7: Full Results Table", fontweight="bold", y=0.97)
    plt.tight_layout()
    plt.savefig("figures/fig7_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig7 saved")
    print("\n=== Numerical Results ===")
    print(f"{'Dataset':<7} {'Model':<20} {'Acc':>6} {'±':>5}  {'AUC':>6} {'±':>5}")
    for r in rows:
        print(f"{r[0]:<7} {r[1]:<20} {r[2]:>6.3f} {r[3]:>5.3f}  {r[4]:>6.3f} {r[5]:>5.3f}")

# ── Run all ──────────────────────────────────────────────────────────────
fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
print("\nAll 7 figures saved to figures/")

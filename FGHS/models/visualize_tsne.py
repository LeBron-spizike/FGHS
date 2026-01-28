import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib

from meta_learner_2 import MetaLearner
from models import MAML

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from torch_geometric.loader import DataLoader

from models.model import KRGTS
from dataset.dataset import FewshotMolDataset
from sklearn.preprocessing import MinMaxScaler



# ===================== MODIFIED =====================
from torch_geometric.loader import DataLoader
# ===================================================

from sklearn.neighbors import NearestNeighbors

# =====================================================
# t-SNE visualization
# =====================================================

def draw_tsne_fast(features, labels, save_path, title):
    """
    features: np.ndarray or torch.Tensor, shape (N, D)
    labels:   np.ndarray or torch.Tensor, shape (N,)
    """

    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    # ---------- 1. 类型与形状统一 ----------
    if torch.is_tensor(features):
        features = features.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    labels = labels.reshape(-1)
    assert features.shape[0] == labels.shape[0], "features / labels size mismatch"

    N = features.shape[0]

    # ---------- 2. 轻量标准化（推荐） ----------
    # features = StandardScaler().fit_transform(features)
    perplexity = min(30, max(5, (N - 1) // 3))

    # ---------- 3. t-SNE ----------
    tsne = TSNE(
        n_components=2,
        perplexity=10,
        init="pca",
        learning_rate="auto",
        n_iter=1000,
        random_state=42,
        verbose=1
    )
    Z = tsne.fit_transform(features)   # (N, 2)

    # ---------- 4. 绘图 ----------
    plt.figure(figsize=(6, 6))

    for lab, name, color in [(0, "Negative", "#1f77b4"),
                             (1, "Positive", "#da0808")]:
        idx = labels == lab
        if idx.sum() == 0:
            continue
        plt.scatter(
            Z[idx, 0], Z[idx, 1],
            s=6, alpha=0.6,
            c=color, label=name
        )

    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title(title)
    plt.legend()
    # plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] t-SNE figure -> {save_path}")


def tsne_feature_dict_to_csv(tsne_feature_dict, save_path):
    rows = []

    for task_id, tensor_list in tsne_feature_dict.items():
        for tensor_idx, feat in enumerate(tensor_list):
            # 确保在 CPU
            feat = feat.detach().cpu()

            # (N, D)
            N, D = feat.shape

            for i in range(N):
                row = {
                    "task_id": task_id,
                    "sample_id": f"{tensor_idx}_{i}"
                }
                # 每一维展开
                for d in range(D):
                    row[f"dim_{d}"] = feat[i, d].item()
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")

# =====================================================
# Main
# =====================================================
def main(args):
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ---- Dataset ----
    print("[Loading dataset]")
    dataset = FewshotMolDataset(
        root=args.data_root,
        name=args.dataset,
        device=device,
        workers=args.workers,
        chunk_size=args.chunk_size
    )

    # ---- Model ----
    print("[Loading model]")
    model = KRGTS(
        args,
        task_num=args.task_num,
        train_task_num=args.task_num,
        device=device
    )

    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and ("model" in ckpt or "state_dict" in ckpt):
        state_dict = ckpt.get("model", ckpt.get("state_dict"))
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Checkpoint] missing={len(missing)}, unexpected={len(unexpected)}")

    model.to(device)
    model.eval()


    metaLearner = MetaLearner(args, device)
    metaLearner.auxiliary_selector.load_state_dict(
        torch.load('/models/sider/auxiliary_selector_param.pkl')
    )
    # score, tsne_task_features, tsne_labels = metaLearner.test_step()
    score, tsne_before, tsne_after, tsne_labels = metaLearner.test_step()

    # ---- Visualization ----
    os.makedirs(args.out_dir, exist_ok=True)

    for task_i in tsne_before:
        feats = torch.cat(tsne_before[task_i], dim=0).numpy()
        # normalize
        feats = normalize(feats, axis=1)

        labels = torch.cat(tsne_labels[task_i], dim=0).numpy().reshape(-1)

        print(f"{task_i} feats: {feats.shape}")
        print(f"{task_i} labels: {labels.shape}")

        draw_tsne_fast(
            feats,
            labels,
            save_path=os.path.join(args.out_dir, f"tsne_task_sider_{task_i}_before.png"),
            title=f"t-SNE (SIDER_before)"
        )
    for task_i in tsne_after:
        feats = torch.cat(tsne_after[task_i], dim=0).numpy()
        # normalize
        feats = normalize(feats, axis=1)

        labels = torch.cat(tsne_labels[task_i], dim=0).numpy().reshape(-1)

        print(f"{task_i} feats: {feats.shape}")
        print(f"{task_i} labels: {labels.shape}")

        draw_tsne_fast(
            feats,
            labels,
            save_path=os.path.join(args.out_dir, f"tsne_task_sider_{task_i}.png"),
            title=f"t-SNE (SIDER)"
        )


    print("[All done]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic paths
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--dataset", type=str, default="tox21")
    parser.add_argument("--ckpt", type=str, default="models/sider/model.pkl")
    parser.add_argument("--out_dir", type=str, default="tsne_results")

    # Data loading
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=1)

    # Task selection (for multi-task datasets)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--task_num", type=int, default=1)

    # KRGTS arguments (must match training)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--mol_num_layer", type=int, default=5)
    parser.add_argument("--mol_dropout", type=float, default=0.1)
    parser.add_argument("--mol_graph_pooling", type=str, default="mean")
    parser.add_argument("--mol_gnn_type", type=str, default="gin")
    parser.add_argument("--mol_batch_norm", action="store_true")
    parser.add_argument("--mol_pretrain_load_path", type=str, default=None)
    parser.add_argument("--JK", type=str, default="last")

    parser.add_argument("--rel_layer", type=int, default=2)
    parser.add_argument("--rel_top_k", type=int, default=3)
    parser.add_argument("--rel_dropout", type=float, default=0.2)
    parser.add_argument("--rel_pre_dropout", type=float, default=0.1)
    parser.add_argument("--num_relation_attr", type=int, default=4)
    parser.add_argument("--rel_norm", type=str, default="batch")
    parser.add_argument("--rel_hidden_dim", type=int, default=128)
    parser.add_argument("--rel_batch_norm", default=1, type=int)
   
    # t-SNE visualization parameters
    parser.add_argument("--tsne_perplexity", type=int, default=20)
    parser.add_argument("--tsne_lr", type=float, default=200)
    parser.add_argument("--tsne_pca_dim", type=int, default=50)
    parser.add_argument("--tsne_max_points", type=int, default=1500)
    
    parser.add_argument("--task_aware", action="store_true",
                    help="Use task-aware (scheme B) t-SNE on query set")
    parser.add_argument("--n_support", type=int, default=10)
    parser.add_argument("--n_query", type=int, default=16)
    parser.add_argument("--inner_steps", type=int, default=5)
    
    parser.add_argument("--rel_nan_w", type=float, default=0.0)
    parser.add_argument("--task_batch_size", default=256, type=int)

    # maml
    parser.add_argument("--inner_lr", default=5e-1, type=float)
    parser.add_argument("--meta_lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=5e-5, type=float)
    parser.add_argument("--second_order", default=1, type=int)
    parser.add_argument("--inner_update_step", default=1, type=int)
    parser.add_argument("--inner_tasks", default=10, type=int)

    # few-shot
    parser.add_argument("--episode", default=2000, type=int)
    parser.add_argument("--eval_step", default=10, type=int)
    parser.add_argument("--test_batch_size", default=128, type=int)
    parser.add_argument("--train_auxi_task_num", default=None, type=int)
    parser.add_argument("--test_auxi_task_num", default=None, type=int)

    # contrastive
    parser.add_argument("--nce_t", default=0.08, type=float)
    parser.add_argument("--contr_w", default=0.05, type=float)

    # meta training selector
    parser.add_argument("--pool_num", default=10, type=int)
    parser.add_argument("--task_lr", default=5e-4, type=float)

    # auxiliary selector
    parser.add_argument("--auxi_lr", default=5e-4, type=float)
    parser.add_argument("--auxi_norm", default=0, type=int)
    parser.add_argument("--s_weight", default=0.3, type=float)
    parser.add_argument("--q_weight", default=0.7, type=float)
    parser.add_argument("--auxi_gamma", default=0.95, type=float)

    # modify-element
    parser.add_argument("--use_element_view", default=1, type=int,
                        help="Whether to use the element view (1: use, 0: skip)")

    args = parser.parse_args()
    main(args)

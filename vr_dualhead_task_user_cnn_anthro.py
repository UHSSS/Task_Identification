from __future__ import annotations
import os, re, glob, json, pickle, argparse, random, math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable
from collections import defaultdict
from time import perf_counter
from pathlib import Path
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, precision_recall_fscore_support, roc_auc_score, average_precision_score
from datetime import datetime

import re
import numpy as np
import pandas as pd
import tensorflow as tf


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ===== Helpers for exports =====
def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _export_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[EXPORT] {path}")

# ----- Timing callback for per-epoch durations -----
class EpochTimer(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []  # seconds

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = perf_counter()
    def on_epoch_end(self, epoch, logs=None):
        try:
            dt = perf_counter() - self._t0
            self.epoch_times.append(float(dt))
            print(f"[EPOCH {epoch:02d}] {dt:.2f}s")
        except Exception:
            pass

# -------------------- Dataclasses --------------------
@dataclass
class StreamConfig:
    fs: float = 90.0
    win_ms: int = 200
    stride_ms: int = 75
    stability_K: int = 15
    random_state: int = 42

    user_col: str = "user"
    trial_col: str = "trial"
    idle_label: str = "idle"

    use_cols_patterns: Tuple[str, ...] = (
        "imu_pos_", "imu_rot_",
        "rh_pos_", "rh_rot_", "rh_*_curl","rh_*_knuckle_*"
    )

# -------------------- Utils --------------------
def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass

def _norm_path(p: str) -> str:
    return os.path.normpath(os.path.expanduser(p))

def _select_columns(df: pd.DataFrame, patterns: Tuple[str, ...]) -> List[str]:
    cols = []
    for p in patterns:
        if p == "rh_*_curl":
            cols.extend([c for c in df.columns if c.startswith("rh_") and c.endswith("_curl")])
        else:
            cols.extend([c for c in df.columns if c.startswith(p)])
    cols = [c for c in cols if not c.startswith("lh_")]
    out, seen = [], set()
    for c in cols:
        if c in df.columns and c not in seen:
            out.append(c); seen.add(c)
    return out

def _window_indices(start: int, end: int, W: int, S: int) -> List[Tuple[int,int]]:
    idx, t = [], start
    while t + W <= end:
        idx.append((t, t+W)); t += S
    return idx

def _infer_user_trial_from_path(path: str, df: Optional[pd.DataFrame]=None, user_col="user", trial_col="trial") -> Tuple[str,str]:
    if df is not None and user_col in df.columns and trial_col in df.columns:
        return str(df[user_col].iloc[0]), str(df[trial_col].iloc[0])
    parts = os.path.normpath(path).split(os.sep)
    nums = [p for p in parts if re.fullmatch(r'\d+', p)]
    if len(nums) >= 2:
        return nums[-2], nums[-1]
    dirs = [p for p in parts[:-1]]  # exclude filename
    user = dirs[-2] if len(dirs) >= 2 else "user_1"
    trial = dirs[-1] if len(dirs) >= 1 else "trial_1"
    return str(user), str(trial)

def _infer_task_from_filename(path: str, idle_label: str) -> Optional[str]:
    stem = Path(path).stem
    token = re.split(r'[_\-\s]+', stem, maxsplit=1)[0]
    token = re.sub(r'\d+$', '', token)
    if not token:
        return None
    t = token
    if t.lower().startswith("projectile"):
        return "Projectile"
    if idle_label and t.lower() == idle_label.lower():
        return None
    return t

def build_paths(root: str, filename: str, pattern: Optional[str] = None) -> List[str]:
    root = _norm_path(root)
    ptn = os.path.join(root, pattern) if pattern else os.path.join(root, "*", "*", filename)
    paths = sorted(glob.glob(ptn))
    return paths

def _load_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return None

def discover_feature_columns(paths: List[str], cfg: StreamConfig) -> List[str]:
    seen=set(); ordered=[]
    for p in paths:
        df=_load_csv(p)
        if df is None: continue
        cols=_select_columns(df, cfg.use_cols_patterns)
        for c in cols:
            if c not in seen:
                seen.add(c); ordered.append(c)
    if not ordered:
        raise SystemExit("No candidate feature columns discovered. Check files/headers.")
    return ordered

def df_to_matrix_TxC(df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
    X = df.reindex(columns=feature_columns, fill_value=0.0).to_numpy(dtype=float, copy=False)
    return X

def _make_trial_split(paths: List[str], cfg: StreamConfig, train_counts: Tuple[int,int], W: int, test_count: int = 0) -> Dict[Tuple[str,str], int]:
    rng = np.random.default_rng(cfg.random_state)
    trials_by_user = defaultdict(list)
    for p in paths:
        df = _load_csv(p)
        if df is None or len(df) < W: continue
        u,t = _infer_user_trial_from_path(p, df, cfg.user_col, cfg.trial_col)
        if t not in trials_by_user[u]: trials_by_user[u].append(t)
    for u in trials_by_user: trials_by_user[u] = sorted(trials_by_user[u])
    a,b = train_counts; split={}
    for u,trials in trials_by_user.items():
        n=len(trials); order=np.arange(n); rng.shuffle(order)
        n_tr=min(a,n)
        n_va=min(b, max(0, n - n_tr))
        n_te=min(test_count, max(0, n - n_tr - n_va))
        tr_idx = order[:n_tr]
        va_idx = order[n_tr:n_tr+n_va]
        te_idx = order[n_tr+n_va:n_tr+n_va+n_te]
        tr=set(trials[i] for i in tr_idx)
        va=set(trials[i] for i in va_idx)
        te=set(trials[i] for i in te_idx)
        for t in trials:
            split[(u,t)] = 0 if t in tr else (1 if t in va else (2 if t in te else -1))
    return split

def discover_from_train(paths: List[str], cfg: StreamConfig, split: Dict[Tuple[str,str], int]) -> List[str]:
    train_paths=[]
    for p in paths:
        df=_load_csv(p)
        if df is None: continue
        u,t=_infer_user_trial_from_path(p, df, cfg.user_col, cfg.trial_col)
        if split.get((u,t),-1)==0: train_paths.append(p)
    return discover_feature_columns(train_paths, cfg)

# -------------------- Feature/Label Builders --------------------
def build_windows(paths: List[str], cfg: StreamConfig, feature_columns: List[str], split: Dict[Tuple[str,str], int], max_wins_per_file:int=0, for_train:bool=True):
    rng = np.random.default_rng(cfg.random_state)
    W = int(round(cfg.win_ms * cfg.fs / 1000.0))
    S = int(round(cfg.stride_ms * cfg.fs / 1000.0))
    X_list=[]; yt_list=[]; yu_list=[]; tags=[]; tasks_set=set(); users_set=set()
    for p in paths:
        df = _load_csv(p)
        if df is None: continue
        user,trial = _infer_user_trial_from_path(p, df, cfg.user_col, cfg.trial_col)
        tag = split.get((user,trial), -1)
        task = _infer_task_from_filename(p, cfg.idle_label)
        if task is None: continue
        tasks_set.add(task); users_set.add(user)
        Xall = df_to_matrix_TxC(df, feature_columns)
        if len(Xall) < W: continue
        idx = _window_indices(0, len(Xall), W, S)
        if for_train and max_wins_per_file and len(idx) > max_wins_per_file:
            sel = rng.choice(len(idx), size=max_wins_per_file, replace=False)
            idx = list(np.array(idx)[sel])
        for (i0,i1) in idx:
            Xwin = Xall[i0:i1]
            X_list.append(Xwin)
            yt_list.append((task,)) ; yu_list.append((user,))
            tags.append(tag)
    if not X_list: raise SystemExit("No windows found. Check files/headers/splits.")
    return np.array(X_list, dtype=np.float32), np.array(yt_list, dtype=object), np.array(yu_list, dtype=object), np.array(tags, dtype=int), sorted(tasks_set), sorted(users_set), W

def compute_channel_norm(train_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_X.reshape(-1, train_X.shape[-1]).mean(axis=0)
    std  = train_X.reshape(-1, train_X.shape[-1]).std(axis=0)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)

def apply_channel_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean.reshape(1,1,-1)) / std.reshape(1,1,-1)

# -------------------- Anthropometric Normalization --------------------
def _anthro_feature_indices(feature_columns):
    """Return indices for rh_pos_* (x,y,z) and imu_pos_* (x,y,z if available).
    Falls back gracefully if columns are missing."""
    rh_pos_idx = [i for i,c in enumerate(feature_columns) if c.startswith("rh_pos_")]
    imu_pos_idx = [i for i,c in enumerate(feature_columns) if c.startswith("imu_pos_")]
    # Prefer exactly x,y,z if present for L2 norms
    def pick_xyz(prefix):
        m = {}
        for axis in ("x","y","z"):
            name = f"{prefix}{axis}"
            if name in feature_columns:
                m[axis] = feature_columns.index(name)
        return [m.get("x"), m.get("y"), m.get("z")]
    rh_xyz = pick_xyz("rh_pos_")
    imu_xyz = pick_xyz("imu_pos_")
    return {"rh_pos": rh_pos_idx, "imu_pos": imu_pos_idx, "rh_xyz": rh_xyz, "imu_xyz": imu_xyz}

def compute_anthro_scales(X, yu_raw, tags, feature_columns):
    """Compute per-user arm-length and height proxies using only TRAIN windows (tags==0).
    - arm: 95th percentile of ||rh_pos_xyz|| over time and windows
    - height: (95th - 5th) percentile of imu_pos_z if available, else L2 range of imu_pos_xyz
    Returns dict with per-user scales and global medians for fallback.
    """
    # import numpy as np
    idxs = _anthro_feature_indices(feature_columns)
    rh_xyz = [i for i in idxs["rh_xyz"] if i is not None]
    imu_xyz = [i for i in idxs["imu_xyz"] if i is not None]
    have_rh = len(rh_xyz) >= 2  # need at least 2 axes for meaningful norm
    have_imu = len(idxs["imu_pos"]) > 0

    users = [u[0] for u in yu_raw]
    users_unique = sorted(set(users))

    per_user = {}
    arms_all, heights_all = [], []
    # iterate users
    for u in users_unique:
        m_user = np.array([uu == u for uu in users], dtype=bool)
        m_train = (tags == 0)
        m = m_user & m_train
        if not np.any(m):
            continue
        Xu = X[m]  # (n_win, T, C)
        # Arm proxy
        arm_scale = np.nan
        if have_rh:
            rh = Xu[:,:,rh_xyz]  # (..., 2-3)
            # L2 norm per timestep
            l2 = np.linalg.norm(rh, axis=-1).reshape(-1)
            if l2.size:
                arm_scale = float(np.percentile(l2, 95))
        # Height proxy
        height_scale = np.nan
        if "imu_pos_z" in feature_columns:
            z_idx = feature_columns.index("imu_pos_z")
            seq = Xu[:,:,z_idx].reshape(-1)
            if seq.size:
                p95 = float(np.percentile(seq, 95))
                p05 = float(np.percentile(seq, 5))
                height_scale = max(1e-6, p95 - p05)
        elif imu_xyz:
            imu = Xu[:,:,imu_xyz]
            l2 = np.linalg.norm(imu, axis=-1).reshape(-1)
            if l2.size:
                height_scale = float(np.percentile(l2, 95) - np.percentile(l2, 5))

        # Finalize with safe minima
        if not (arm_scale and arm_scale > 1e-6):
            arm_scale = float("nan")
        if not (height_scale and height_scale > 1e-6):
            height_scale = float("nan")

        per_user[u] = {"arm": arm_scale, "height": height_scale}
        if not np.isnan(arm_scale): arms_all.append(arm_scale)
        if not np.isnan(height_scale): heights_all.append(height_scale)

    # Global fallbacks
    # import numpy as np
    global_arm = float(np.nanmedian(arms_all)) if arms_all else 1.0
    global_height = float(np.nanmedian(heights_all)) if heights_all else 1.0

    return {"per_user": per_user, "global": {"arm": max(1e-6, global_arm),
                                             "height": max(1e-6, global_height)},
            "feature_indices": idxs}

def apply_anthro_norm(X, yu_raw, scales, feature_columns):
    """Divide rh_pos_* by per-user arm scale and imu_pos_* by height scale, inplace-like."""
    # import numpy as np
    out = X.copy()
    idxs = scales.get("feature_indices") or _anthro_feature_indices(feature_columns)
    rh_idx = idxs["rh_pos"]
    imu_idx = idxs["imu_pos"]
    if not rh_idx and not imu_idx:
        return out  # nothing to do

    users = [u[0] for u in yu_raw]
    per_user = scales.get("per_user", {})
    g = scales.get("global", {"arm":1.0, "height":1.0})
    for i,u in enumerate(users):
        arm = per_user.get(u, {}).get("arm", g["arm"]) or g["arm"]
        height = per_user.get(u, {}).get("height", g["height"]) or g["height"]
        if rh_idx:
            out[i,:,rh_idx] = out[i,:,rh_idx] / max(1e-6, arm)
        if imu_idx:
            out[i,:,imu_idx] = out[i,:,imu_idx] / max(1e-6, height)
    return out


def to_indices(labels: np.ndarray, vocab: List[str]) -> np.ndarray:
    m = {s:i for i,s in enumerate(vocab)}
    return np.array([m[x[0]] for x in labels], dtype=np.int32)

def one_hot(y: np.ndarray, K: int) -> np.ndarray:
    oh = np.zeros((len(y), K), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh

# -------------------- Metrics/Calibration --------------------
def _equal_error_rate(fpr, tpr, thr):
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0), float(thr[i])

def binary_like_eer(scores_pos: np.ndarray, scores_neg: np.ndarray) -> Tuple[float,float]:
    if len(scores_pos)==0 or len(scores_neg)==0:
        return float('nan'), float('nan')
    scores = np.concatenate([scores_pos, scores_neg])
    labels = np.concatenate([np.ones_like(scores_pos), np.zeros_like(scores_neg)])
    fpr, tpr, thr = roc_curve(labels, scores)
    return _equal_error_rate(fpr, tpr, thr)

def identification_eer(probs: np.ndarray, y_true: np.ndarray):
    n = len(y_true)
    if n == 0: return float('nan'), float('nan')
    p_true = probs[np.arange(n), y_true]
    impost = probs.copy(); impost[np.arange(n), y_true] = -1.0
    p_imp = impost.max(axis=1)
    return binary_like_eer(p_true, p_imp)

def choose_theta_from_validation(probs_val: np.ndarray, y_val: np.ndarray, target_precision: float=0.98) -> float:
    if probs_val.size == 0:
        return 0.8
    maxp = probs_val.max(axis=1)
    yhat = probs_val.argmax(axis=1)
    correct = (yhat == y_val)
    uniq = np.unique(maxp)[::-1]
    best = None
    for t in uniq:
        mask = (maxp >= t)
        if not mask.any(): continue
        prec = correct[mask].mean()
        if prec >= target_precision: best = t
    if best is None:
        best = np.quantile(maxp[correct], 0.80) if correct.any() else 0.8
    return float(best)

def choose_theta_safe(model, X_val, y_val, X_tr, y_tr, head: str, target_precision: float=0.98):
    try:
        if X_val is not None and len(X_val):
            p = model.predict(X_val, verbose=0)[head]
            return choose_theta_from_validation(p, y_val, target_precision), "val"
        elif X_tr is not None and len(X_tr):
            p = model.predict(X_tr, verbose=0)[head]
            tgt = min(0.995, max(0.90, target_precision))
            return choose_theta_from_validation(p, y_tr, tgt), "train"
        else:
            return 0.8, "default"
    except Exception as e:
        print(f"[WARN] theta selection failed for {head}: {e}")
        return 0.8, "default"

# -------------------- Model --------------------
def build_cnn(T: int, C: int, n_tasks: int, n_users: int, dropout: float=0.2) -> Model:
    inp = layers.Input(shape=(T, C))
    x = layers.Conv1D(32, 5, padding="same")(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv1D(64, 5, padding="same", dilation_rate=2)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(64, 3, padding="same", dilation_rate=4)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    head_task = layers.Dense(n_tasks, activation="softmax", name="task")(x)
    head_user = layers.Dense(n_users, activation="softmax", name="user")(x)
    model = Model(inp, {"task": head_task, "user": head_user})
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"task": "categorical_crossentropy", "user": "categorical_crossentropy"},
        metrics={"task": ["accuracy"], "user": ["accuracy"]}
    )
    return model

# -------------------- Stages --------------------
def stage_train(args):
    t0 = perf_counter()  # total stage start
    set_seed(args.seed)
    cfg = StreamConfig(fs=args.fs, win_ms=args.win_ms, stride_ms=args.stride_ms,
                       stability_K=args.stability_k, random_state=args.seed, idle_label=args.idle)

    paths = build_paths(_norm_path(args.root), args.filename, pattern=args.pattern)
    if args.list_only:
        print("[TRAIN] Listing matched files:"); [print("  ", p) for p in paths]; return
    if not paths: raise SystemExit("No CSV files.")

    W = int(round(cfg.win_ms * cfg.fs / 1000.0))

    split = _make_trial_split(paths, cfg, args.train_counts, W, test_count=args.test_counts)
    feat_cols = discover_from_train(paths, cfg, split)

    X, yt_raw, yu_raw, tags, tasks, users, T = build_windows(

        paths, cfg, feat_cols, split, max_wins_per_file=args.max_wins_per_file, for_train=True
    )


    # --- Restrict vocabularies to TRAIN-only to avoid unseen-class softmax units ---


    _train_idx = np.where(tags == 0)[0]


    _train_tasks = [yt_raw[i][0] for i in _train_idx]


    _train_users = [yu_raw[i][0] for i in _train_idx]


    tasks = sorted(set(_train_tasks))
    # --- Anthropometric normalization (per-user arm/height) ---
    anthro_scales = compute_anthro_scales(X, yu_raw, tags, feat_cols)
    X = apply_anthro_norm(X, yu_raw, anthro_scales, feat_cols)



    users = sorted(set(_train_users))
    C = X.shape[-1]
    y_task = to_indices(yt_raw, tasks); y_user = to_indices(yu_raw, users)

    m_tr = (tags == 0); m_va = (tags == 1)

    mean, std = compute_channel_norm(X[m_tr])
    Xn = apply_channel_norm(X, mean, std).astype(np.float32)

    model = build_cnn(T, C, n_tasks=len(tasks), n_users=len(users), dropout=args.dropout)

    ytr_task_oh = one_hot(y_task[m_tr], len(tasks))
    ytr_user_oh = one_hot(y_user[m_tr], len(users))
    Xtr = Xn[m_tr]

    Xva = Xn[m_va]; yva_task = y_task[m_va]; yva_user = y_user[m_va]
    yva_task_oh = one_hot(yva_task, len(tasks)) if len(yva_task) else None
    yva_user_oh = one_hot(yva_user, len(users)) if len(yva_user) else None

    # callbacks
    callbacks = []
    has_val = len(yva_task) > 0
    epoch_timer = EpochTimer()  # capture per-epoch durations
    if has_val:
        callbacks = [
            epoch_timer,
            EarlyStopping(monitor="val_task_accuracy", mode="max", patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_task_accuracy", mode="max", factor=0.5, patience=4, min_lr=1e-5),
        ]
    else:
        callbacks = [epoch_timer]

    # ---- timing: prep, fit, total ----
    prep_s = perf_counter() - t0

    t_fit = perf_counter()
    hist = model.fit(
        Xtr, {"task": ytr_task_oh, "user": ytr_user_oh},
        validation_data=(Xva, {"task": yva_task_oh, "user": yva_user_oh}) if has_val else None,
        epochs=args.epochs, batch_size=args.batch_size, verbose=1, callbacks=callbacks
    )
    fit_s = perf_counter() - t_fit
    total_s = perf_counter() - t0
    print(f"[TIME] prep={prep_s:.1f}s | fit={fit_s:.1f}s | total={total_s:.1f}s")

    theta_task, src_t = choose_theta_safe(model, Xva, yva_task, Xtr, y_task[m_tr], "task",
                                          target_precision=args.precision_target)
    theta_user, src_u = choose_theta_safe(model, Xva, yva_user, Xtr, y_user[m_tr], "user",
                                          target_precision=args.precision_target)
    print(f"[TRAIN] θ_task={theta_task:.3f} (from {src_t}), θ_user={theta_user:.3f} (from {src_u})")

    model_out = _norm_path(args.model_out)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    keras_path = os.path.splitext(model_out)[0] + "_cnn.keras"
    model.save(keras_path)

    artifact = {
        "version": 1,
        "keras_model_path": keras_path,
        "tasks": tasks, "users": users,
        "theta": {"task": float(theta_task), "user": float(theta_user)},
        "cfg": dict(
            fs=cfg.fs, win_ms=cfg.win_ms, stride_ms=cfg.stride_ms,
            stability_K=cfg.stability_K, idle_label=cfg.idle_label,
            use_cols_patterns=list(cfg.use_cols_patterns)
        ),
        "feature_columns": feat_cols,
        "trial_split": {f"{u}/{t}": tag for (u, t), tag in split.items()},
        "channel_mean": mean, "channel_std": std,
        "window_T": int(T),
        "timing": {"prep_s": float(prep_s), "fit_s": float(fit_s), "total_s": float(total_s)},
        "anthro_scales": anthro_scales
    }


    # === Export timing data ===
    export_dir = os.path.join(os.path.dirname(model_out), "exports")
    _ensure_dir(export_dir)

    # Add a deterministic split fingerprint summary
    try:
        artifact["split_fingerprint"] = {
            "seed": int(args.seed) if hasattr(args, "seed") else None,
            "counts": {
                "train": int(sum(1 for v in artifact.get("trial_split", {}).values() if v == 0)),
                "val":   int(sum(1 for v in artifact.get("trial_split", {}).values() if v == 1)),
                "test":  int(sum(1 for v in artifact.get("trial_split", {}).values() if v == 2)),
                "unused":int(sum(1 for v in artifact.get("trial_split", {}).values() if v == -1)),
            },
        }
    except Exception:
        pass
        # === Export timing data ===
    # Per-epoch durations
    if epoch_timer.epoch_times:
        df_epochs = pd.DataFrame({
            "epoch": list(range(len(epoch_timer.epoch_times))),
            "seconds": epoch_timer.epoch_times
        })
        _export_csv(df_epochs, os.path.join(export_dir, "train_epoch_times.csv"))

    # Summary row
    df_summary = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prep_s": float(prep_s),
        "fit_s": float(fit_s),
        "total_s": float(total_s),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "train_windows": int(len(Xtr)),
        "val_windows": int(len(Xva)) if has_val else 0,
        "window_T": int(T),
        "channels": int(C)
    }])
    _export_csv(df_summary, os.path.join(export_dir, "train_summary.csv"))

    with open(model_out, "wb") as f:
        pickle.dump(artifact, f)
    print(f"[SAVE] model -> {keras_path}")
    print(f"[SAVE] artifact -> {model_out}")
    print(f"[INFO] Tasks({len(tasks)}): {tasks}")
    print(f"[INFO] Users({len(users)}): {users}")

def _filter_by_split(artifact: dict, use_splits: str, paths: List[str], cfg: StreamConfig) -> List[str]:
    if use_splits == "all": return paths
    split = artifact.get("trial_split", {})
    kept=[]
    sel=set()
    if use_splits == "train": sel = {k for k,v in split.items() if v==0}
    elif use_splits == "val": sel = {k for k,v in split.items() if v==1}
    elif use_splits == "test": sel = {k for k,v in split.items() if v==2}
    elif use_splits in ("unused","not-trainval"): sel = {k for k,v in split.items() if v==-1}
    elif use_splits == "not-train": sel = {k for k,v in split.items() if v!=0}
    else: sel = set(split.keys())
    for p in paths:
        df=_load_csv(p)
        u,t=_infer_user_trial_from_path(p, df, cfg.user_col, cfg.trial_col)
        if f"{u}/{t}" in sel: kept.append(p)
    return kept

def build_eval_windows(paths: List[str], cfg: StreamConfig, feat_cols: List[str],
                       tasks: List[str], users: List[str], allowed_keys: Optional[Iterable[str]] = None) -> Tuple[np.ndarray,np.ndarray,np.ndarray,int]:
    W = int(round(cfg.win_ms * cfg.fs / 1000.0))
    S = int(round(cfg.stride_ms * cfg.fs / 1000.0))
    X_list=[]; yt_list=[]; yu_list=[]
    tset=set(tasks); uset=set(users)
    allowed = set(allowed_keys) if allowed_keys is not None else None
    for p in paths:
        df=_load_csv(p)
        if df is None: continue
        u,t=_infer_user_trial_from_path(p, df, cfg.user_col, cfg.trial_col)
        key=f"{u}/{t}"
        if allowed is not None and key not in allowed: continue
        task=_infer_task_from_filename(p, cfg.idle_label)
        if (not task) or (task not in tset) or (u not in uset): continue
        Xall=df_to_matrix_TxC(df, feat_cols)
        if len(Xall) < W: continue
        for i0,i1 in _window_indices(0, len(Xall), W, S):
            X_list.append(Xall[i0:i1])
            yt_list.append(tasks.index(task))
            yu_list.append(users.index(u))
    if not X_list: raise SystemExit("No eval windows found.")
    return np.array(X_list, dtype=np.float32), np.array(yt_list, dtype=np.int32), np.array(yu_list, dtype=np.int32), W

def brier_score(probs, y_true):
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - onehot)**2, axis=1)))

def expected_calibration_error(probs, y_true, n_bins: int = 10):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i+1] if i < n_bins-1 else conf <= bins[i+1])
        if not np.any(m): continue
        gap = abs(acc[m].mean() - conf[m].mean())
        ece += (m.mean()) * gap
    return float(ece)

def normalize_cm(cm):
    with np.errstate(invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row_sums, where=row_sums>0)
    return norm

def top_confusions(cm, labels, topn=5):
    out = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i==j: continue
            if cm[i,j] > 0:
                out.append((labels[i], labels[j], int(cm[i,j])))
    out.sort(key=lambda x: x[2], reverse=True)
    return out[:topn]

def _shorten_labels(labels, maxlen=10):
    out = []
    for s in labels:
        s = str(s)
        out.append(s if len(s) <= maxlen else s[:maxlen-1] + "…")
    return out

def _print_rule(title=None, char='─', width=80):
    if title:
        pad = max(0, width - len(title) - 2)
        print(f"{char*3} {title} {char*pad}")
    else:
        print(char*width)

def _fmt_row(cells, widths, align='>'):
    parts = []
    for c,w in zip(cells, widths):
        s = str(c)
        parts.append(s.rjust(w) if align=='>' else s.ljust(w))
    return " | ".join(parts)

def _print_kv(pairs, key_w=24, val_w=52):
    for k,v in pairs:
        print(f"{str(k).ljust(key_w)}: {v}")

def _print_confusion(cm, row_labels, col_labels, normalize=False, max_label=10, width=90):
    rlab = _shorten_labels(row_labels, max_label)
    clab = _shorten_labels(col_labels, max_label)
    rows, cols = cm.shape
    rlw = max(max(len(x) for x in rlab), 5)
    cw = max(5, max(len(x) for x in clab))
    widths = [rlw] + [cw]*cols
    header = ["true\\pred"] + clab
    print(_fmt_row(header, widths, align='<'))
    print('-'*min(width, rlw + (cw+3)*cols))
    for i in range(rows):
        row = [rlab[i]]
        for j in range(cols):
            val = cm[i,j]
            row.append(f"{val:.3f}" if normalize else str(int(val)))
        print(_fmt_row(row, widths))


def stage_test(args):
    # --- Load artifact ---
    art_path = Path(_norm_path(args.model_in)).resolve()
    with open(art_path, "rb") as f:
        art = pickle.load(f)
    art_dir = art_path.parent

    # --- Resolve model path ---
    keras_path = art.get("keras_model_path", "")
    kp = Path(keras_path)
    if not kp.is_absolute():
        kp = (art_dir / kp).resolve()

    # --- Silent fallback search ---
    if not kp.exists():
        candidates = list(art_dir.glob("*.keras")) + list((art_dir / "cvmodels").glob("*.keras"))
        if not candidates:
            raise FileNotFoundError(f"Model file not found: {kp}")
        # pick the first match deterministically
        kp = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    # --- Load model quietly ---
    try:
        model = keras.models.load_model(str(kp), compile=False)
    except Exception:
        model = tf.keras.models.load_model(str(kp), compile=False)
        
    tasks = art["tasks"]; users = art["users"]
    theta = art["theta"]
    cfgd = art["cfg"]
    feat_cols = art["feature_columns"]
    mean = art["channel_mean"]; std = art["channel_std"]
    cfg = StreamConfig(fs=cfgd["fs"], win_ms=cfgd["win_ms"], stride_ms=cfgd["stride_ms"],
                       stability_K=cfgd["stability_K"], idle_label=cfgd["idle_label"])

    root = _norm_path(args.root)
    paths = build_paths(root, args.filename, pattern=args.pattern)
    if args.list_only:
        print("[TEST] Listing matched files:"); [print("  ", p) for p in paths]; return
    if not paths: raise SystemExit("No CSV files.")

    # Export directory resolution
    export_dir = args.export_dir
    if not export_dir:
        export_dir = os.path.join(os.path.dirname(_norm_path(args.model_in)), "exports")
    _ensure_dir(export_dir)

    paths = _filter_by_split(art, args.use_splits, paths, cfg)
    if not paths: raise SystemExit("[TEST] After --use-splits, no files remain.")

    split = art.get("trial_split", {})
    if args.use_splits == "all": allowed_keys=None
    else:
        if args.use_splits == "train": allowed_keys={k for k,v in split.items() if v==0}
        elif args.use_splits == "val": allowed_keys={k for k,v in split.items() if v==1}
        elif args.use_splits == "test": allowed_keys={k for k,v in split.items() if v==2}
        elif args.use_splits in ("unused","not-trainval"): allowed_keys={k for k,v in split.items() if v==-1}
        elif args.use_splits == "not-train": allowed_keys={k for k,v in split.items() if v!=0}
        else: allowed_keys=None

    X, y_task, y_user, T = build_eval_windows(paths, cfg, feat_cols, tasks, users, allowed_keys=allowed_keys)

    # Apply same anthropometric normalization used at train time
    anthro_scales = art.get("anthro_scales", None)
    if anthro_scales is not None:
        yu_raw_test = np.array([[users[idx]] for idx in y_user], dtype=object)
        X = apply_anthro_norm(X, yu_raw_test, anthro_scales, feat_cols)
    Xn = apply_channel_norm(X, mean, std).astype(np.float32)

    # ===== Inference timing + predictions =====
    infer_times = []
    if args.time_per_window:
        for i in range(len(Xn)):
            x = Xn[i:i+1]
            t0 = perf_counter()
            _ = model(x, training=False)
            infer_times.append(float(perf_counter() - t0))
        preds = model.predict(Xn, verbose=0)
        pt = preds["task"]; pu = preds["user"]
    else:
        t0 = perf_counter()
        preds = model.predict(Xn, verbose=0)
        total_predict_s = float(perf_counter() - t0)
        avg = total_predict_s / max(1, len(Xn))
        infer_times = [avg] * len(Xn)
        pt = preds["task"]; pu = preds["user"]

    # --- WALL CLOCK PATCH: cumulative compute time (seconds) for quick lookup ---
    infer_times = np.asarray(infer_times, dtype=float)
    infer_times_cumsum = np.cumsum(infer_times)

    acc_t = (pt.argmax(1) == y_task).mean()
    acc_u = (pu.argmax(1) == y_user).mean()
    _print_rule("WINDOW-LEVEL ACCURACY")
    print(f"Task acc: {acc_t*100:.2f}% | User acc: {acc_u*100:.2f}%")

    def _eer(p, y):
        e, thr = identification_eer(p, y); return e, thr
    et, thr_t = _eer(pt, y_task); eu, thr_u = _eer(pu, y_user)
    _print_rule("WINDOW-LEVEL EER")
    print(f"Task EER: {et*100:.2f}%  (theta@EER≈{thr_t:.3f})")
    print(f"User EER: {eu*100:.2f}%  (theta@EER≈{thr_u:.3f})")

    def prf(p, y):
        pred = p.argmax(1)
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average=None, zero_division=0)
        pr_macro, rc_macro, f1_macro, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
        pr_weighted, rc_weighted, f1_weighted, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
        return {
            "macro": {"precision": float(pr_macro), "recall": float(rc_macro), "f1": float(f1_macro)},
            "weighted": {"precision": float(pr_weighted), "recall": float(rc_weighted), "f1": float(f1_weighted)}
        }
    prf_task = prf(pt, y_task)
    prf_user = prf(pu, y_user)
    _print_rule("WINDOW-LEVEL PR / REC / F1 (macro, weighted)")
    print(f"TASK  | macro F1={prf_task['macro']['f1']:.3f}  macro P/R=({prf_task['macro']['precision']:.3f}/{prf_task['macro']['recall']:.3f})  weighted F1={prf_task['weighted']['f1']:.3f}")
    print(f"USER  | macro F1={prf_user['macro']['f1']:.3f}  macro P/R=({prf_user['macro']['precision']:.3f}/{prf_user['macro']['recall']:.3f})  weighted F1={prf_user['weighted']['f1']:.3f}")

    try:
        roc_task = float(roc_auc_score(y_task, pt, multi_class="ovr"))
        pr_task  = float(average_precision_score((np.eye(len(set(y_task)))[y_task]), pt, average="macro"))
    except Exception:
        roc_task = float("nan"); pr_task = float("nan")
    try:
        roc_user = float(roc_auc_score(y_user, pu, multi_class="ovr"))
        pr_user  = float(average_precision_score((np.eye(len(set(y_user)))[y_user]), pu, average="macro"))
    except Exception:
        roc_user = float("nan"); pr_user = float("nan")

    brier_t = brier_score(pt, y_task); ece_t = expected_calibration_error(pt, y_task)
    brier_u = brier_score(pu, y_user); ece_u = expected_calibration_error(pu, y_user)
    _print_rule("CALIBRATION")
    print(f"Task: Brier={brier_t:.4f}, ECE={ece_t:.4f}, ROC-AUC={roc_task:.3f}, PR-AUC={pr_task:.3f}")
    print(f"User: Brier={brier_u:.4f}, ECE={ece_u:.4f}, ROC-AUC={roc_user:.3f}, PR-AUC={pr_user:.3f}")

    cm_t = confusion_matrix(y_task, pt.argmax(1), labels=list(range(len(tasks))))
    cm_u = confusion_matrix(y_user, pu.argmax(1), labels=list(range(len(users))))
    cm_tn = (cm_t / np.clip(cm_t.sum(axis=1, keepdims=True), 1, None))
    cm_un = (cm_u / np.clip(cm_u.sum(axis=1, keepdims=True), 1, None))
    top_conf_t = top_confusions(cm_t, tasks, topn=5)

    _print_rule("CONFUSION MATRIX — TASK (raw)")
    _print_confusion(cm_t, tasks, tasks, normalize=False)
    _print_rule("CONFUSION MATRIX — TASK (row-normalized)")
    _print_confusion(cm_tn, tasks, tasks, normalize=True)

    _print_rule("CONFUSION MATRIX — USER (raw)")
    _print_confusion(cm_u, users, users, normalize=False)
    _print_rule("CONFUSION MATRIX — USER (row-normalized)")
    _print_confusion(cm_un, users, users, normalize=True)

    _print_rule("TOP CONFUSIONS (TASK)")
    for a,b,cnt in top_conf_t:
        print(f"{a} → {b}: {cnt}")

    require = args.require.lower()
    K = int(args.k_override) if args.k_override is not None else cfg.stability_K
    K_task = int(args.k_task) if getattr(args, "k_task", None) is not None else K
    K_user = int(args.k_user) if getattr(args, "k_user", None) is not None else K
    K2 = int(args.k2_fallback) if args.k2_fallback is not None else 0
    theta_bump = float(args.theta_bump)

    # hard-force after N strides if no θ/K decision
    FORCE_STRIDES = 20

    data_lat = []
    decisions=0; c_task=0; c_user=0; c_both=0
    per_trial_rows = []

    decision_events = []
    missed_events = []
    forced_events = []

    file_to_windows = []
    W = T; S = int(round(cfg.stride_ms * cfg.fs / 1000.0))
    idx = 0
    for p in paths:
        df=_load_csv(p)
        if df is None: continue
        u,t=_infer_user_trial_from_path(p, df, cfg.user_col, cfg.trial_col)
        task=_infer_task_from_filename(p, cfg.idle_label)
        if (task not in tasks) or (u not in users): continue
        Xall=df_to_matrix_TxC(df, feat_cols)
        if len(Xall) < W: continue
        nwin = len(_window_indices(0, len(Xall), W, S))
        file_to_windows.append((p, u, t, task, idx, idx+nwin))
        idx += nwin

    _print_rule("STREAM SETUP")
    _print_kv([
        ("require", args.require),
        ("theta_task", f"{theta['task']:.3f}"),
        ("theta_user", f"{theta['user']:.3f}"),
        ("theta_bump", f"{theta_bump:+.3f}"),
        ("K_task", K_task),
        ("K_user", K_user),
        ("K2_fallback", K2),
        ("force_after_strides/end", FORCE_STRIDES),
        ("windows_per_stride_ms", cfg.stride_ms),
    ])

    for (p,u,t,task, i0,i1) in file_to_windows:
        true_t = tasks.index(task); true_u = users.index(u)
        consec_t = consec_u = 0; decision_idx = None

        last_t, last_u = [], []


        for i in range(i0, i1):
            pt_i = pt[i]; pu_i = pu[i]
            pred_t = int(np.argmax(pt_i)); pred_u = int(np.argmax(pu_i))
            maxpt = float(np.max(pt_i)); maxpu = float(np.max(pu_i))

            # --- CONSENSUS-BASED (non-leaky) decision logic ---
            # Maintain rolling buffers of last K predictions
            last_t.append(pred_t)
            if len(last_t) > K_task: last_t.pop(0)
            last_u.append(pred_u)
            if len(last_u) > K_user: last_u.pop(0)

            # Confidence check
            conf_t = (maxpt >= (theta['task'] + theta_bump))
            conf_u = (maxpu >= (theta['user'] + theta_bump))
            # Consistency (same label for K consecutive windows)
            cons_t = (len(last_t) == K_task and len(set(last_t)) == 1)
            cons_u = (len(last_u) == K_user and len(set(last_u)) == 1)

            hit_t = conf_t and cons_t
            hit_u = conf_u and cons_u


            consec_t = consec_t + 1 if hit_t else 0
            consec_u = consec_u + 1 if hit_u else 0

            ok = (consec_t >= K_task) if require == 'task' else \
                 (consec_u >= K_user) if require == 'user' else \
                 ((consec_t >= K_task) and (consec_u >= K_user))
            if ok:
                decision_idx = i - i0
                decisions += 1
                c_t = (pred_t == true_t); c_u = (pred_u == true_u)
                if c_t: c_task += 1
                if c_u: c_user += 1
                if c_t and c_u: c_both += 1

                i_eval = i
                data_ms = (decision_idx + 1) * cfg.stride_ms

                # latency components
                window_fill_ms = int(cfg.win_ms)
                wait_ms = int(max(0, decision_idx) * cfg.stride_ms)

                # compute
                if len(infer_times_cumsum) == len(pt):
                    cum_start = infer_times_cumsum[i0 - 1] if i0 > 0 else 0.0
                    compute_ms_until_decision = int(round((infer_times_cumsum[i_eval] - cum_start) * 1000.0))
                else:
                    compute_ms_until_decision = 0

                total_decision_latency_ms = int(window_fill_ms + wait_ms + compute_ms_until_decision)

                maxpt_i = float(np.max(pt[i_eval]))
                maxpu_i = float(np.max(pu[i_eval]))
                decision_events.append({
                    "file_path": p,
                    "user": u,
                    "trial": t,
                    "true_task": tasks[true_t],
                    "true_user": users[true_u],
                    "pred_task": tasks[int(np.argmax(pt[i_eval]))],
                    "pred_user": users[int(np.argmax(pu[i_eval]))],
                    "t_decide_stride_ms": int(data_ms),
                    "window_fill_ms": int(window_fill_ms),
                    "wait_ms": int(wait_ms),
                    "compute_ms_until_decision": int(compute_ms_until_decision),
                    "total_decision_latency_ms": int(total_decision_latency_ms),
                    "require": args.require,
                    "K_task": int(K_task),
                    "K_user": int(K_user),
                    "theta_task": float(theta["task"]),
                    "theta_user": float(theta["user"]),
                    "theta_bump": float(theta_bump),
                    "maxp_task_at_decision": maxpt_i,
                    "maxp_user_at_decision": maxpu_i,
                    "consec_task_required": int(K_task),
                    "consec_user_required": int(K_user),
                    "correct_task": int(c_t),
                    "correct_user": int(c_u),
                    "correct_both": int(c_t and c_u),
                    "decision_source": "thetaK"
                })
                break

        if decision_idx is not None:
            data_ms = (decision_idx + 1) * cfg.stride_ms
            data_lat.append(total_decision_latency_ms)
            i_eval = i0 + decision_idx
            pred_t = int(np.argmax(pt[i_eval])); pred_u = int(np.argmax(pu[i_eval]))
            per_trial_rows.append([u, t, task, True, int(total_decision_latency_ms), tasks[pred_t], users[pred_u], int(pred_t==true_t and pred_u==true_u)])
        else:
            # fallback
            nwin = (i1 - i0)
            decision_idx = None
            decision_source = None

            if nwin <= 0:
                per_trial_rows.append([u, t, task, False, None, None, None, 0])
            else:
                if K2 and nwin >= K2:
                    decision_idx = K2 - 1
                    decision_source = "K2"
                elif nwin >= FORCE_STRIDES:
                    decision_idx = FORCE_STRIDES - 1
                    decision_source = "force_5strides"
                else:
                    decision_idx = nwin - 1
                    decision_source = "force_end"

                i_eval = i0 + decision_idx
                data_ms = (decision_idx + 1) * cfg.stride_ms

                decisions += 1
                pred_t = int(np.argmax(pt[i_eval])); pred_u = int(np.argmax(pu[i_eval]))
                c_t = (pred_t == true_t); c_u = (pred_u == true_u)
                if c_t: c_task += 1
                if c_u: c_user += 1
                if c_t and c_u: c_both += 1

                window_fill_ms = int(cfg.win_ms)
                wait_ms = int(max(0, decision_idx) * cfg.stride_ms)
                if len(infer_times_cumsum) == len(pt):
                    cum_start = infer_times_cumsum[i0 - 1] if i0 > 0 else 0.0
                    compute_ms_until_decision = int(round((infer_times_cumsum[i_eval] - cum_start) * 1000.0))
                else:
                    compute_ms_until_decision = 0
                total_decision_latency_ms = int(window_fill_ms + wait_ms + compute_ms_until_decision)

                per_trial_rows.append([u, t, task, True, int(total_decision_latency_ms), tasks[pred_t], users[pred_u],
                                       int(pred_t==true_t and pred_u==true_u)])

                maxpt_i = float(np.max(pt[i_eval]))
                maxpu_i = float(np.max(pu[i_eval]))
                data_lat.append(total_decision_latency_ms)
                decision_events.append({
                    "file_path": p,
                    "user": u,
                    "trial": t,
                    "true_task": tasks[true_t],
                    "true_user": users[true_u],
                    "pred_task": tasks[pred_t],
                    "pred_user": users[pred_u],
                    "t_decide_stride_ms": int(data_ms),
                    "window_fill_ms": int(window_fill_ms),
                    "wait_ms": int(wait_ms),
                    "compute_ms_until_decision": int(compute_ms_until_decision),
                    "total_decision_latency_ms": int(total_decision_latency_ms),
                    "require": args.require,
                    "K_task": int(K_task),
                    "K_user": int(K_user),
                    "theta_task": float(theta["task"]),
                    "theta_user": float(theta["user"]),
                    "theta_bump": float(theta_bump),
                    "maxp_task_at_decision": maxpt_i,
                    "maxp_user_at_decision": maxpu_i,
                    "consec_task_required": int(K_task),
                    "consec_user_required": int(K_user),
                    "correct_task": int(c_t),
                    "correct_user": int(c_u),
                    "correct_both": int(c_t and c_u),
                    "decision_source": decision_source
                })

    # Latency summary
    _print_rule("TOTAL DECISION LATENCY (ms)")
    wall_lat = [d.get("total_decision_latency_ms", None) for d in decision_events if d.get("total_decision_latency_ms", None) is not None]
    if wall_lat:
        def _p(a, q): return float(np.percentile(a, q)) if len(a) else float('nan')
        def _avg(a): return float(np.mean(a)) if len(a) else float('nan')
        p50, p90, p95, mean = np.median(wall_lat), _p(wall_lat, 90), _p(wall_lat, 95), _avg(wall_lat)
        print(f"Total-decision latency: p50={p50:.1f}  p90={p90:.1f}  p95={p95:.1f}  mean={mean:.1f}  n={len(wall_lat)}")
    else:
        print("No decisions (θ/K too strict?)")

    num_streams = len(file_to_windows)
    missed = max(0, num_streams - decisions)

    _print_rule("STREAM DECISIONS")
    print(f"trials={num_streams}  decisions={decisions}  missed={missed}  decision_rate={decisions/max(1,num_streams):.3f}")

    if decisions > 0:
        print(f"accuracy (task/user/both): {c_task/decisions:.3f} / {c_user/decisions:.3f} / {c_both/decisions:.3f}")
    else:
        print("no decisions")

    # Misclassified trials (decided but wrong)
    _print_rule("STREAM MISCLASSIFIED TRIALS")
    wrong_rows = []
    for row in per_trial_rows:
        u, t, true_task, decided, t_ms, p_task, p_user, correct_both = row
        if not decided:
            continue
        is_task_ok = (str(p_task) == str(true_task))
        is_user_ok = (str(p_user) == str(u))
        if not (is_task_ok and is_user_ok):
            reason = []
            reason.append("task_wrong" if not is_task_ok else "task_ok")
            reason.append("user_wrong" if not is_user_ok else "user_ok")
            wrong_rows.append([u, t, true_task, p_task, p_user, t_ms, "/".join(reason)])
    if not wrong_rows:
        print("None")
    else:
        headers = ["user","trial","true_task","pred_task","pred_user","t_decide_ms","reason"]
        widths  = [8,8,14,14,10,12,16]
        print(" | ".join(h.ljust(w) for h,w in zip(headers,widths)))
        print("-"*sum(widths))
        for r in wrong_rows:
            print(" | ".join(str(c).ljust(w) for c,w in zip(r,widths)))

    # ===== Exports for analysis =====
    df_win_metrics = pd.DataFrame([{
        "task_acc_window": float(acc_t),
        "user_acc_window": float(acc_u),
        "task_eer_window": float(et),
        "task_theta_at_eer_window": float(thr_t) if thr_t == thr_t else None,
        "user_eer_window": float(eu),
        "user_theta_at_eer_window": float(thr_u) if thr_u == thr_u else None
    }])
    _export_csv(df_win_metrics, os.path.join(export_dir, "test_window_metrics.csv"))

    df_trials = pd.DataFrame(per_trial_rows, columns=[
        "user","trial","true_task","decided","total_decision_latency_ms","pred_task","pred_user","correct_both"
    ])
    _export_csv(df_trials, os.path.join(export_dir, "test_per_trial_decisions.csv"))

    if data_lat:
        df_lat = pd.DataFrame({"total_decision_latency_ms": data_lat})
        _export_csv(df_lat, os.path.join(export_dir, "test_stream_latency_ms.csv"))

    if decision_events:
        df_dec = pd.DataFrame(decision_events)
        _export_csv(df_dec.sort_values(["total_decision_latency_ms","t_decide_stride_ms"], ascending=False),
                    os.path.join(export_dir, "test_decision_events.csv"))
        _print_rule("SLOWEST DECISIONS")
        for _, r in df_dec.sort_values("total_decision_latency_ms", ascending=False).head(5).iterrows():
            print(f"{r['file_path']} | Decision Latency={int(r['total_decision_latency_ms'])} "
                  f"(fill={int(r['window_fill_ms'])}, wait={int(r['wait_ms'])}, compute={int(r['compute_ms_until_decision'])}) "
                  f"pred=({r['pred_task']}/{r['pred_user']}) correct_both={r['correct_both']}")
    if missed_events:
        df_miss = pd.DataFrame(missed_events)
        _export_csv(df_miss, os.path.join(export_dir, "test_forced_events.csv"))


def build_argparser():
    ap = argparse.ArgumentParser(prog="vr_dualhead_task_user_cnn.py")
    sp = ap.add_subparsers(dest="cmd", required=True)

    tr = sp.add_parser("train")
    tr.add_argument("--root", type=str, required=True)
    tr.add_argument("--filename", type=str, default="*.csv")
    tr.add_argument("--pattern", type=str, default=None)
    tr.add_argument("--fs", type=float, default=90.0)
    tr.add_argument("--win-ms", type=int, default=200)
    tr.add_argument("--stride-ms", type=int, default=75)
    tr.add_argument("--stability-k", type=int, default=2)
    tr.add_argument("--idle", type=str, default="idle")
    tr.add_argument("--seed", type=int, default=1715)
    tr.add_argument("--train-counts", type=lambda s: tuple(map(int, s.split(','))), default=(12,4))
    tr.add_argument("--test-counts", type=int, default=0)
    tr.add_argument("--precision-target", type=float, default=0.98)
    tr.add_argument("--max-wins-per-file", type=int, default=0)
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--batch-size", type=int, default=128)
    tr.add_argument("--dropout", type=float, default=0.2)
    tr.add_argument("--model-out", type=str, required=True)
    tr.add_argument("--list-only", action="store_true")

    te = sp.add_parser("test")
    te.add_argument("--root", type=str, required=True)
    te.add_argument("--filename", type=str, default="*.csv")
    te.add_argument("--pattern", type=str, default=None)
    te.add_argument("--use-splits", choices=["all","train","val","test","unused","not-train","not-trainval"], default="all")
    te.add_argument("--require", choices=["task","user","both"], default="both")
    te.add_argument("--k-override", type=int, default=None)
    te.add_argument("--k-task", type=int, default=None)
    te.add_argument("--k-user", type=int, default=None)
    te.add_argument("--k2-fallback", type=int, default=None)
    te.add_argument("--theta-bump", type=float, default=0.0)
    te.add_argument("--model-in", type=str, required=True)
    te.add_argument("--dump-per-trial", type=str, default=None)
    te.add_argument("--list-only", action="store_true")
    # exports/timing controls
    te.add_argument("--export-dir", type=str, default=None,
                    help="Folder to put timing/latency CSVs (default: next to model-in).")
    te.add_argument("--time-per-window", action="store_true",
                    help="Measure per-window inference time (slower; one forward pass per window).")

    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()
    if args.cmd == "train": stage_train(args)
    else: stage_test(args)

if __name__ == "__main__":
    main()

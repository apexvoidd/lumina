import os
import threading
import torch
import json
import zipfile
import io
import ssl
from flask import Flask, jsonify, request, send_from_directory, send_file
from tokenizer import CustomTokenizer
from model import GPTLanguageModel
from train import train_model

app = Flask(__name__, static_folder="static")

@app.before_request
def handle_options_preflight():
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        return response

@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

# ── Directory setup ─────────────────────────────────────────────────────────
BASE_DIR             = os.path.dirname(os.path.abspath(__file__))
STANDALONE_FLAG_PATH = os.path.join(BASE_DIR, "standalone.flag")

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "default"), exist_ok=True)

active_model = "default"

def get_model_paths(model_name):
    """Return (weights_path, vocab_path, state_path) for the given model name."""
    m_dir = os.path.join(MODELS_DIR, model_name)
    os.makedirs(m_dir, exist_ok=True)
    return (
        os.path.join(m_dir, "model_weights.pth"),
        os.path.join(m_dir, "vocab.json"),
        os.path.join(m_dir, "training_state.json")
    )

def get_available_datasets():
    if not os.path.exists(DATASETS_DIR):
        return []
    return sorted(f for f in os.listdir(DATASETS_DIR) if f.endswith(".txt"))

# Check standalone mode
is_standalone = os.path.exists(STANDALONE_FLAG_PATH)

# ── Global training / dataset state ─────────────────────────────────────────
training_state = {
    "status":        "idle",
    "iteration":     0,
    "max_iters":     0,
    "loss":          0.0,
    "speed":         0.0,
    "loss_history":  [],
    "error_message": "",
    "hyperparams":   {}
}

dataset_state = {
    "status":        "idle",
    "progress":      0,
    "total":         0,
    "error_message": "",
    "filename":      ""
}

training_lock  = threading.Lock()
stop_requested = False

model_cache = {
    "model":         None,
    "tokenizer":     None,
    "weights_mtime": 0,
    "model_name":    ""
}

# ── State persistence helpers ────────────────────────────────────────────────
def save_state_to_file(model_name=None):
    global training_state, active_model
    if model_name is None:
        model_name = active_model
    _, _, state_path = get_model_paths(model_name)
    try:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(training_state, f, indent=4)
    except Exception as e:
        print(f"[WARN] Could not save training state for '{model_name}': {e}")

def load_state_from_file(model_name="default"):
    global training_state
    _, _, state_path = get_model_paths(model_name)
    training_state.update({
        "status":        "idle",
        "iteration":     0,
        "max_iters":     0,
        "loss":          0.0,
        "speed":         0.0,
        "loss_history":  [],
        "error_message": "",
        "hyperparams":   {},
        "dataset_format": "dialogue"
    })
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if loaded.get("status") == "training":
                loaded["status"] = "interrupted"
            training_state.update(loaded)
            print(f"[INFO] Loaded saved training state for model '{model_name}'.")
        except Exception as e:
            print(f"[WARN] Could not load training state for '{model_name}': {e}")

# ── Static routes ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)

# ── Status endpoint ──────────────────────────────────────────────────────────
@app.route("/api/status", methods=["GET"])
def get_status():
    global active_model
    with training_lock:
        weights_path, _, _ = get_model_paths(active_model)
        has_weights = os.path.exists(weights_path)
        return jsonify({
            **training_state,
            "has_weights":    has_weights,
            "standalone":     is_standalone,
            "active_model":   active_model,
            "dataset_status": dataset_state
        })

# ── Dataset / Model list endpoints ──────────────────────────────────────────
@app.route("/api/datasets", methods=["GET"])
def list_datasets():
    return jsonify({"datasets": get_available_datasets()})

@app.route("/api/models", methods=["GET"])
def list_models():
    folders = []
    if os.path.exists(MODELS_DIR):
        for name in os.listdir(MODELS_DIR):
            if os.path.isdir(os.path.join(MODELS_DIR, name)):
                folders.append(name)
    if not folders:
        folders = ["default"]
    return jsonify({"models": sorted(folders), "active_model": active_model})

@app.route("/api/model/switch", methods=["POST"])
def switch_model():
    global active_model
    data = request.json or {}
    model_name = data.get("model_name", "").strip()
    if not model_name:
        return jsonify({"error": "Model name is required"}), 400

    # Sanitize name
    safe_name = "".join(c for c in model_name if c.isalnum() or c in ("_", "-")).strip()
    if not safe_name:
        return jsonify({"error": "Invalid model name. Use only letters, digits, _ or -"}), 400

    m_dir = os.path.join(MODELS_DIR, safe_name)
    os.makedirs(m_dir, exist_ok=True)

    with training_lock:
        active_model = safe_name
        load_state_from_file(active_model)

    return jsonify({
        "message":        f"Switched to model '{safe_name}'",
        "active_model":   active_model,
        "training_state": training_state
    })

# ── Hugging Face Dataset Endpoints ───────────────────────────────────────────

def _bypass_ssl():
    """Disable SSL certificate verification for HF downloads on restrictive networks."""
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        os.environ["PYTHONHTTPSVERIFY"] = "0"
        os.environ["CURL_CA_BUNDLE"] = ""
    except Exception:
        pass

def _load_hf_dataset_split(dataset_name, config=None, token=None, streaming=False):
    """
    Try to load a Hugging Face dataset split.
    Returns a dataset object. Raises on failure.
    NOTE: trust_remote_code is intentionally omitted (deprecated in datasets >= 2.20).
    """
    import datasets as hf_datasets

    load_kwargs = {"streaming": streaming}
    if token:
        load_kwargs["token"] = token
        os.environ["HF_TOKEN"] = token

    splits_to_try = ["train", "validation", "test"]
    ds = None

    for split in splits_to_try:
        try:
            if config:
                ds = hf_datasets.load_dataset(dataset_name, config, split=split, **load_kwargs)
            else:
                ds = hf_datasets.load_dataset(dataset_name, split=split, **load_kwargs)
            print(f"[INFO] Loaded split='{split}' for dataset '{dataset_name}' (config={config})")
            break
        except Exception as split_err:
            print(f"[DEBUG] Split '{split}' failed: {split_err}")
            continue

    if ds is None:
        # Fallback: load the whole DatasetDict and take the first available split
        try:
            if config:
                ds_dict = hf_datasets.load_dataset(dataset_name, config, **load_kwargs)
            else:
                ds_dict = hf_datasets.load_dataset(dataset_name, **load_kwargs)
            if hasattr(ds_dict, "keys") and len(ds_dict) > 0:
                first_split = list(ds_dict.keys())[0]
                ds = ds_dict[first_split]
                print(f"[INFO] Using fallback split='{first_split}' for '{dataset_name}'")
            else:
                ds = ds_dict
        except Exception as e:
            raise RuntimeError(f"Could not load any split of '{dataset_name}': {e}")

    return ds


@app.route("/api/dataset/hf/configs", methods=["POST"])
def hf_list_configs():
    """
    Return the list of available configs/subsets for a dataset.
    Tries the official API first; falls back to parsing config names from the HF error.
    NOTE: trust_remote_code is intentionally NOT passed to get_dataset_config_names
    because some versions of the datasets library treat unknown kwargs as config names.
    """
    data         = request.json or {}
    dataset_name = data.get("dataset_name", "").strip()
    token        = data.get("token", "").strip()
    if not dataset_name:
        return jsonify({"error": "Dataset name is required"}), 400

    _bypass_ssl()

    try:
        import datasets as hf_datasets
        # Do NOT pass trust_remote_code here – some library versions treat it as a config name
        kw = {}
        if token:
            kw["token"] = token
        configs = hf_datasets.get_dataset_config_names(dataset_name, **kw)
        print(f"[INFO] Found {len(configs)} configs for '{dataset_name}': {configs[:5]}")
        return jsonify({"configs": configs})
    except Exception as e:
        err_str = str(e)
        print(f"[WARN] get_dataset_config_names failed for '{dataset_name}': {err_str}")
        configs = _parse_configs_from_error(err_str, exclude=dataset_name)
        if configs:
            print(f"[INFO] Parsed {len(configs)} configs from error message.")
            return jsonify({"configs": configs})
        return jsonify({"configs": []})



def _parse_configs_from_error(err_str, exclude=None):
    """
    Parse config/subset names from HF error messages. Handles formats:
      - 'Config name is missing. Please pick one among the available configs: [a, b]'
      - 'BuilderConfig X not found. Available: [a, b]'
      - "Available: ['a', 'b']"
    exclude: string to strip from results (typically the dataset name).
    """
    import re
    results = []

    # Strategy 1: parse a bracketed list after 'available' or 'configs' keyword
    # Matches: Available: ['foo', 'bar']  or  configs: ['foo', 'bar']
    list_match = re.search(
        r"(?:available|configs?)\s*:\s*\[([^\]]+)\]",
        err_str, re.IGNORECASE
    )
    if list_match:
        raw = list_match.group(1)
        # Extract all quoted or unquoted tokens from inside the brackets
        items = re.findall(r"['\"]([^'\"]+)['\"]", raw)
        if not items:
            items = [p.strip() for p in raw.split(",") if p.strip()]
        results = [c for c in items if c and len(c) < 120]

    # Strategy 2: fallback – find all quoted strings in the whole message
    if not results:
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", err_str)
        results = [c for c in quoted if c and len(c) < 120]

    # Remove the dataset name itself if it snuck in
    if exclude:
        # exclude could be 'org/repo' – strip both forms
        exclude_parts = {exclude, exclude.split("/")[-1] if "/" in exclude else exclude}
        results = [c for c in results if c not in exclude_parts]

    return results


@app.route("/api/dataset/hf/preview", methods=["POST"])
def hf_preview():
    """
    Preview a single row of a Hugging Face dataset split.
    Checks for required configs BEFORE attempting to load data.
    Returns:
      { columns: [...], preview: {...} }   on success
      { configs: [...] }                   when a subset must be chosen first
      { error: '...' }                     on other failure
    """
    data         = request.json or {}
    dataset_name = data.get("dataset_name", "").strip()
    token        = data.get("token", "").strip()
    config       = data.get("config", "").strip() or None

    if not dataset_name:
        return jsonify({"error": "Dataset name is required"}), 400

    _bypass_ssl()

    import datasets as hf_datasets

    # ────────────────────────────────────────────────────────────────────────
    # STEP 1 – If no config given, proactively check whether this dataset
    #          requires one before we attempt any data loading at all.
    # NOTE: Do NOT pass trust_remote_code to get_dataset_config_names –
    #       some datasets versions treat unknown kwargs as config names.
    # ────────────────────────────────────────────────────────────────────────
    if not config:
        try:
            cfg_kw = {}
            if token:
                cfg_kw["token"] = token
            all_configs = hf_datasets.get_dataset_config_names(dataset_name, **cfg_kw)
            print(f"[INFO] '{dataset_name}' reports {len(all_configs)} config(s): {all_configs[:8]}")
            if len(all_configs) > 1:
                return jsonify({"configs": all_configs})
            elif len(all_configs) == 1:
                config = all_configs[0]
                print(f"[INFO] Auto-selecting sole config: '{config}'")
        except Exception as cfg_err:
            print(f"[WARN] get_dataset_config_names failed for '{dataset_name}': {cfg_err}")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 2 – Load one streaming row for preview
    # ────────────────────────────────────────────────────────────────────────
    try:
        ds        = _load_hf_dataset_split(dataset_name, config=config, token=token or None, streaming=True)
        iterator  = iter(ds)
        first_row = next(iterator)

        columns = list(first_row.keys())
        preview = {}
        for k, v in first_row.items():
            s = str(v)
            preview[k] = s[:200] + ("..." if len(s) > 200 else "")

        return jsonify({"columns": columns, "preview": preview})

    except StopIteration:
        return jsonify({"error": "Dataset is empty."}), 400

    except Exception as e:
        err_str = str(e)

        # Detect ANY error that tells us a config/subset is required
        needs_config = any(phrase in err_str.lower() for phrase in [
            "config name is missing",
            "please pick one",
            "available configs",
            "config_name",
            "builderconfig",      # catches 'BuilderConfig X not found. Available: [...]'
            "available:",         # catches 'Available: [x, y]'
        ])
        if needs_config:
            # Try the API without trust_remote_code first
            try:
                cfg_kw = {}
                if token:
                    cfg_kw["token"] = token
                all_configs = hf_datasets.get_dataset_config_names(dataset_name, **cfg_kw)
            except Exception:
                all_configs = []
            # Fallback: parse from the error text itself
            if not all_configs:
                all_configs = _parse_configs_from_error(err_str, exclude=dataset_name)
            if all_configs:
                print(f"[INFO] Surfacing {len(all_configs)} config(s) from error for '{dataset_name}'.")
                return jsonify({"configs": all_configs})
            return jsonify({
                "error": (
                    "This dataset requires a config/subset name. "
                    "Visit https://huggingface.co/datasets/" + dataset_name +
                    " to find the correct config name, type it in the subset box "
                    "and click Fetch Subset."
                )
            }), 400

        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to preview dataset: {err_str}"}), 500


def run_dataset_load_thread(dataset_name, limit, mapping_config, config=None, token=None):
    global dataset_state
    _bypass_ssl()
    try:
        ds = _load_hf_dataset_split(dataset_name, config=config, token=token, streaming=False)

        # Determine actual row count and clamp limit.
        # Many HF datasets (especially with configs) don't support direct index
        # access, so we convert to a list of dicts upfront.
        print(f"[INFO] Converting dataset to list (limit={limit})...")
        try:
            total_available = len(ds)
            actual_limit = min(limit, total_available)
            # Use the HF Dataset select() if available to avoid loading everything
            try:
                ds_slice = ds.select(range(actual_limit))
                rows = list(ds_slice)
            except Exception:
                rows = [ds[i] for i in range(actual_limit)]
        except Exception:
            # No len() — iterate and collect up to limit rows
            rows = []
            for row in ds:
                rows.append(row)
                if len(rows) >= limit:
                    break
            actual_limit = len(rows)

        with training_lock:
            dataset_state.update({
                "status":        "loading",
                "progress":      0,
                "total":         actual_limit,
                "error_message": ""
            })

        formatted_lines = []
        m_type = mapping_config.get("type", "pairs")

        for i, row in enumerate(rows):
            try:
                if m_type == "pairs":
                    user_col  = mapping_config.get("user_col", "")
                    ai_col    = mapping_config.get("ai_col", "")
                    user_text = " ".join(str(row.get(user_col, "")).split())
                    ai_text   = " ".join(str(row.get(ai_col,   "")).split())
                    if user_text and ai_text:
                        formatted_lines.append(f"User: {user_text}\n")
                        formatted_lines.append(f"AI: {ai_text}\n")

                elif m_type == "list":
                    list_col = mapping_config.get("list_col", "")
                    turns    = row.get(list_col, [])
                    if isinstance(turns, list):
                        for j in range(0, len(turns) - 1, 2):
                            u = " ".join(str(turns[j]).split())
                            a = " ".join(str(turns[j + 1]).split())
                            if u and a:
                                formatted_lines.append(f"User: {u}\n")
                                formatted_lines.append(f"AI: {a}\n")

                elif m_type == "raw":
                    raw_col = mapping_config.get("raw_col", "")
                    text    = str(row.get(raw_col, "")).strip()
                    lines   = [l.strip() for l in text.split("\n") if l.strip()]
                    for j in range(0, len(lines) - 1, 2):
                        u = " ".join(lines[j].split())
                        a = " ".join(lines[j + 1].split())
                        if u and a:
                            formatted_lines.append(f"User: {u}\n")
                            formatted_lines.append(f"AI: {a}\n")

                elif m_type == "plain":
                    raw_col = mapping_config.get("raw_col", "")
                    text    = str(row.get(raw_col, "")).strip()
                    if text:
                        clean_text = " ".join(text.split())
                        formatted_lines.append(f"{clean_text}\n\n")

            except Exception as row_err:
                print(f"[WARN] Skipping row {i}: {row_err}")
                continue

            if i % 100 == 0 or i == actual_limit - 1:
                with training_lock:
                    dataset_state["progress"] = i + 1

        if not formatted_lines:
            # Build a helpful message showing the actual columns found
            sample_cols = list(rows[0].keys()) if rows else []
            sample_preview = {}
            if rows:
                for k, v in rows[0].items():
                    s = str(v)
                    sample_preview[k] = s[:80] + ("..." if len(s) > 80 else "")
            raise ValueError(
                f"No dialogue pairs were extracted from the dataset. "
                f"The dataset has these columns: {sample_cols}. "
                f"Sample row: {sample_preview}. "
                f"Please go back to the Dataset panel, click 'Fetch Columns' again, "
                f"choose the correct column mapping format and column names, "
                f"then click 'Load & Format' again."
            )

        sanitized_name = dataset_name.replace("/", "_").replace("\\", "_")
        if config:
            sanitized_name += f"_{config}"
        filename  = f"hf_{sanitized_name}.txt"
        save_path = os.path.join(DATASETS_DIR, filename)

        with open(save_path, "w", encoding="utf-8") as f:
            f.writelines(formatted_lines)

        with training_lock:
            dataset_state["status"]   = "completed"
            dataset_state["filename"] = filename  # frontend uses this to auto-select
        print(f"[INFO] Dataset saved to '{save_path}' ({len(formatted_lines)} lines).")

    except Exception as e:
        import traceback
        traceback.print_exc()
        with training_lock:
            dataset_state["status"]        = "error"
            dataset_state["error_message"] = str(e)


@app.route("/api/dataset/hf/load", methods=["POST"])
def hf_load():
    with training_lock:
        if dataset_state["status"] == "loading":
            return jsonify({"error": "A dataset is already loading. Please wait."}), 400

    data         = request.json or {}
    dataset_name = data.get("dataset_name", "").strip()
    token        = data.get("token", "").strip()
    config       = data.get("config", "").strip() or None
    limit        = int(data.get("limit", 2000))
    mapping_config = data.get("mapping_config", {})

    if not dataset_name:
        return jsonify({"error": "Dataset name is required"}), 400

    thread = threading.Thread(
        target=run_dataset_load_thread,
        args=(dataset_name, limit, mapping_config, config, token or None),
        daemon=True
    )
    thread.start()

    return jsonify({"message": "Dataset download started."})


# ── Local Dataset Upload ─────────────────────────────────────────────────────
@app.route("/api/dataset/upload", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not file.filename.lower().endswith(".txt"):
        return jsonify({"error": "Only .txt files are supported"}), 400

    try:
        safe_name = "".join(c for c in file.filename if c.isalnum() or c in (".", "_", "-")).strip()
        if not safe_name.lower().endswith(".txt"):
            safe_name += ".txt"
        save_path = os.path.join(DATASETS_DIR, safe_name)
        file.save(save_path)

        with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(1000)

        has_dialogue_format = "User:" in sample and "AI:" in sample

        with training_lock:
            dataset_state.update({"status": "completed", "progress": 100, "total": 100, "error_message": ""})

        return jsonify({
            "message":            "Dataset uploaded successfully",
            "filename":           safe_name,
            "size_bytes":         os.path.getsize(save_path),
            "has_dialogue_format": has_dialogue_format
        })
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500


# ── Model ZIP Export ─────────────────────────────────────────────────────────
@app.route("/api/model/export", methods=["GET"])
def export_model():
    global active_model
    weights_path, vocab_path, state_path = get_model_paths(active_model)

    if not os.path.exists(weights_path) or not os.path.exists(vocab_path):
        return jsonify({"error": "Model has not been trained yet. Train it first!"}), 400

    try:
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(weights_path, "model_weights.pth")
            zf.write(vocab_path,   "vocab.json")
            if os.path.exists(state_path):
                zf.write(state_path, "training_state.json")

            for fname in ("model.py", "tokenizer.py", "train.py", "server.py"):
                full = os.path.join(BASE_DIR, fname)
                if os.path.exists(full):
                    zf.write(full, fname)

            # Activate standalone chat-only mode in the exported zip
            zf.writestr("standalone.flag", "")

            static_dir = os.path.join(BASE_DIR, "static")
            for root, dirs, files in os.walk(static_dir):
                for f in files:
                    fp       = os.path.join(root, f)
                    rel_path = os.path.relpath(fp, BASE_DIR)
                    zf.write(fp, rel_path)

        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"lumina_{active_model}.zip"
        )
    except Exception as e:
        return jsonify({"error": f"Failed to package model: {str(e)}"}), 500


# ── Training Control ─────────────────────────────────────────────────────────
@app.route("/api/train/start", methods=["POST"])
def start_train():
    global stop_requested, active_model

    with training_lock:
        if training_state["status"] == "training":
            return jsonify({"error": "Training is already in progress"}), 400

        data = request.json or {}

        selected_dataset = data.get("dataset", "").strip()
        start_fresh      = bool(data.get("start_fresh", False))

        hyperparams = {
            "n_embd":         int(data.get("n_embd", 64)),
            "n_head":         int(data.get("n_head", 4)),
            "n_layer":        int(data.get("n_layer", 4)),
            "max_iters":      int(data.get("max_iters", 800)),
            "learning_rate":  float(data.get("learning_rate", 0.001)),
            "dropout":        float(data.get("dropout", 0.1)),
            "batch_size":     int(data.get("batch_size", 16)),
            "block_size":     32
        }

        # ── Resolve and validate dataset path BEFORE touching any state ────
        if selected_dataset:
            dataset_filepath = os.path.join(DATASETS_DIR, selected_dataset)
        else:
            available = get_available_datasets()
            if available:
                dataset_filepath = os.path.join(DATASETS_DIR, available[0])
            else:
                return jsonify({"error": "No dataset found. Please load or upload a dataset first!"}), 400

        if not os.path.exists(dataset_filepath):
            available = get_available_datasets()
            if available:
                dataset_filepath = os.path.join(DATASETS_DIR, available[0])
                print(f"[WARN] Requested dataset '{selected_dataset}' not found; falling back to '{available[0]}'.")
            else:
                return jsonify({"error": f"Dataset '{selected_dataset}' not found and no other datasets exist. Please reload it."}), 400

        # Resolve model paths
        weights_path, vocab_path, _ = get_model_paths(active_model)

        # Start fresh: wipe weights and vocab for this model
        if start_fresh:
            print(f"[INFO] Start Fresh requested for model '{active_model}'. Wiping weights.")
            for fpath in (weights_path, vocab_path):
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except Exception as e:
                        print(f"[WARN] Could not remove {fpath}: {e}")

            training_state.update({"iteration": 0, "loss": 0.0, "speed": 0.0, "loss_history": []})
            # Invalidate model cache so next chat reloads
            model_cache["model_name"] = ""

        # Determine resume
        h      = training_state.get("hyperparams", {})
        resume = (
            not start_fresh
            and os.path.exists(weights_path)
            and os.path.exists(vocab_path)
            and h.get("n_embd")  == hyperparams["n_embd"]
            and h.get("n_head")  == hyperparams["n_head"]
            and h.get("n_layer") == hyperparams["n_layer"]
        )
        if resume:
            print(f"[INFO] Hyperparameters match existing weights for '{active_model}'. Resuming...")
        else:
            if not start_fresh:
                training_state.update({"iteration": 0, "loss": 0.0, "speed": 0.0, "loss_history": []})

        stop_requested = False
        training_state.update({
            "status":        "training",
            "error_message": "",
            "hyperparams":   hyperparams
        })

        save_state_to_file()

        # Capture loop variables for the thread closure
        _model_name      = active_model
        _weights_path    = weights_path
        _vocab_path      = vocab_path

        thread = threading.Thread(
            target=run_training_thread,
            args=(_model_name, _weights_path, _vocab_path, hyperparams, dataset_filepath),
            daemon=True
        )
        thread.start()

        return jsonify({"message": "Training started", "hyperparams": hyperparams})


@app.route("/api/train/stop", methods=["POST"])
def stop_train():
    global stop_requested
    with training_lock:
        if training_state["status"] != "training":
            return jsonify({"message": "Training is not running"}), 400
        stop_requested = True
        return jsonify({"message": "Stop requested. Saving weights at next checkpoint."})


def run_training_thread(model_name, weights_path, vocab_path, hyperparams, dataset_filepath):
    """Background thread that drives train.py's train_model()."""
    global stop_requested

    def callback(progress):
        global stop_requested
        with training_lock:
            training_state["iteration"]   = progress["iteration"]
            training_state["max_iters"]   = progress["max_iters"]
            training_state["loss"]        = progress["loss"]
            training_state["speed"]       = progress["speed"]
            if "dataset_format" in progress:
                training_state["dataset_format"] = progress["dataset_format"]
            training_state["loss_history"].append({
                "iter": progress["iteration"],
                "loss": progress["loss"]
            })
            if progress["status"] == "completed":
                training_state["status"] = "completed"

            if progress["iteration"] % 50 == 0:
                save_state_to_file(model_name)

            return stop_requested

    try:
        train_model(
            dataset_path      = dataset_filepath,
            vocab_save_path   = vocab_path,
            weights_save_path = weights_path,
            hyperparams       = hyperparams,
            status_callback   = callback
        )
        with training_lock:
            training_state["status"] = "interrupted" if stop_requested else "completed"
            save_state_to_file(model_name)

        # Invalidate cache so inference reloads the fresh weights
        model_cache["model_name"] = ""

    except Exception as e:
        import traceback
        traceback.print_exc()
        with training_lock:
            training_state["status"]        = "error"
            training_state["error_message"] = str(e)
            save_state_to_file(model_name)


# ── Inference ────────────────────────────────────────────────────────────────
def load_cached_model():
    global active_model
    weights_path, vocab_path, _ = get_model_paths(active_model)

    if not os.path.exists(weights_path) or not os.path.exists(vocab_path):
        return None, None

    mtime = os.path.getmtime(weights_path)

    if (
        model_cache["model"] is None
        or mtime > model_cache["weights_mtime"]
        or model_cache["model_name"] != active_model
    ):
        print(f"[INFO] Loading model '{active_model}' into cache...")
        tokenizer = CustomTokenizer()
        tokenizer.load(vocab_path)

        h          = training_state.get("hyperparams", {})
        n_embd     = h.get("n_embd",     64)
        n_head     = h.get("n_head",      4)
        n_layer    = h.get("n_layer",     4)
        block_size = h.get("block_size", 32)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = GPTLanguageModel(
            vocab_size = tokenizer.vocab_size,
            n_embd     = n_embd,
            n_head     = n_head,
            n_layer    = n_layer,
            block_size = block_size
        ).to(device)

        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.eval()

        model_cache.update({
            "model":         model,
            "tokenizer":     tokenizer,
            "weights_mtime": mtime,
            "model_name":    active_model
        })
        print(f"[INFO] Model '{active_model}' loaded (vocab={tokenizer.vocab_size}).")

    return model_cache["model"], model_cache["tokenizer"]


def determine_mood(text):
    t = text.lower()
    if any(w in t for w in ["haha", "joke", "funny", "cool", "favorite", "!", "awesome"]):
        return "excited"
    if any(w in t for w in ["sorry", "sad", "tear", "deep", "feeling", "care", "heart"]):
        return "empathetic"
    if any(w in t for w in ["think", "meaning", "learn", "question", "how", "why"]):
        return "thoughtful"
    return "calm"


@app.route("/api/chat", methods=["POST"])
def chat():
    if training_state["status"] == "training":
        return jsonify({"response": "I am currently training! Please wait a moment.", "mood": "thoughtful"})

    try:
        model, tokenizer = load_cached_model()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

    if model is None or tokenizer is None:
        return jsonify({
            "response": "Hello! I am not trained yet. Adjust the parameters and click 'Start Training' to teach me!",
            "mood": "calm"
        })

    data         = request.json or {}
    user_message = data.get("message", "").strip()
    temperature  = float(data.get("temperature", 0.8))

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_format = training_state.get("dataset_format", "dialogue")

    if dataset_format == "dialogue":
        user_ids   = tokenizer.encode(user_message)
        prompt_ids = [tokenizer.vocab["<user>"]] + user_ids + [tokenizer.vocab["<ai>"]]
        x          = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        stop_id    = tokenizer.vocab["<end>"]
    else:
        prompt_text = user_message
        if not prompt_text.endswith(" ") and not prompt_text.endswith("\n"):
            prompt_text += " "
        prompt_ids = tokenizer.encode(prompt_text)
        x          = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        stop_id    = None

    y = model.generate(
        x,
        max_new_tokens  = 40,
        temperature     = temperature,
        stop_token_id   = stop_id
    )
    generated_ids = y[0, len(prompt_ids):].tolist()
    response_text = tokenizer.decode(generated_ids)

    if not response_text.strip():
        response_text = "i am still practicing! try asking me something else."

    return jsonify({"response": response_text, "mood": determine_mood(response_text)})


# ── Startup ──────────────────────────────────────────────────────────────────
load_state_from_file()

if __name__ == "__main__":
    os.makedirs(app.static_folder, exist_ok=True)
    print("=" * 60)
    print(" Lumina GPT Trainer — starting on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)

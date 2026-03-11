"""
One-time data preparation for autoresearch experiments.
Downloads data shards and trains a BPE tokenizer.

Usage:
    python prepare.py                  # full prep (download + tokenizer)
    python prepare.py --num-shards 8   # download only 8 shards (for testing)

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import os
import sys
import time
import math
import argparse
import pickle
from multiprocessing import Pool

import requests
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 40 * 524288  # number of tokens for val eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542 # the last datashard is shard_06542.parquet
VAL_SHARD = MAX_SHARD  # pinned validation shard (shard_06542)
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

LANGUAGE_CONFIGS = {
    "en": {
        "name": "English",
        "type": "climbmix",
        "note": "karpathy/climbmix-400b — 400B tokens of web text",
    },
    "fr": {"name": "French",    "type": "wikipedia", "wiki_code": "fr"},
    "es": {"name": "Spanish",   "type": "wikipedia", "wiki_code": "es"},
    "de": {"name": "German",    "type": "wikipedia", "wiki_code": "de"},
    "hi": {"name": "Hindi",     "type": "wikipedia", "wiki_code": "hi"},
    "zh": {"name": "Chinese",   "type": "wikipedia", "wiki_code": "zh"},
    "ja": {"name": "Japanese",  "type": "wikipedia", "wiki_code": "ja"},
    "gu": {"name": "Gujarati",  "type": "wikipedia", "wiki_code": "gu"},
    "nl": {"name": "Dutch",     "type": "wikipedia", "wiki_code": "nl"},
    "or": {"name": "Odia",      "type": "wikipedia", "wiki_code": "or"},
}


def get_lang_dirs(lang="en"):
    """Return (data_dir, tokenizer_dir) for a given language."""
    base = os.path.join(CACHE_DIR, lang)
    return os.path.join(base, "data"), os.path.join(base, "tokenizer")


def get_lang_checkpoint_dir(lang="en"):
    return os.path.join(CACHE_DIR, lang, "checkpoints")


def save_lang_meta(lang, val_filename, num_shards):
    """Save language metadata so train.py can read val_filename."""
    meta_path = os.path.join(CACHE_DIR, lang, "meta.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as f:
        import json
        json.dump({"val_filename": val_filename, "num_shards": num_shards}, f)


def load_lang_meta(lang):
    """Load language metadata. Returns dict or None."""
    meta_path = os.path.join(CACHE_DIR, lang, "meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        import json
        return json.load(f)


def get_wikipedia_shard_list(lang_code):
    """Fetch list of (filename, url) tuples from HuggingFace API for a Wikipedia language."""
    api_url = f"https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.{lang_code}/train"
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()
    files = response.json()
    result = []
    for e in files:
        if isinstance(e, str):
            result.append((e.split("/")[-1].split("?")[0], e))
        else:
            result.append((e["filename"], e["url"]))
    return result

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_single_shard(args):
    """Download one parquet shard. args = (filename, url, data_dir)."""
    filename, url, data_dir = args
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        return True

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_data(lang="en", num_shards=10, download_workers=8):
    """Download training shards for the given language. Returns val_filename."""
    data_dir, _ = get_lang_dirs(lang)
    os.makedirs(data_dir, exist_ok=True)
    cfg = LANGUAGE_CONFIGS[lang]

    if cfg["type"] == "climbmix":
        # English: karpathy/climbmix-400b-shuffle
        num_train = min(num_shards, MAX_SHARD)
        ids = list(range(num_train))
        if VAL_SHARD not in ids:
            ids.append(VAL_SHARD)
        val_filename = VAL_FILENAME
        shard_args = [
            (f"shard_{i:05d}.parquet", f"{BASE_URL}/shard_{i:05d}.parquet", data_dir)
            for i in ids
        ]
    else:
        # Wikipedia language
        wiki_code = cfg["wiki_code"]
        print(f"Data: discovering {cfg['name']} Wikipedia shards...")
        try:
            all_shards = get_wikipedia_shard_list(wiki_code)
        except Exception as e:
            print(f"Data: failed to fetch shard list for {wiki_code}: {e}")
            return VAL_FILENAME
        # Use last shard as val, rest as train (capped by num_shards)
        val_filename = all_shards[-1][0]
        if len(all_shards) == 1:
            # Only 1 shard available (small language) — use it for both train and val
            print(f"Data: only 1 shard available for {cfg['name']}, using it for both train and val.")
            shards_to_download = all_shards
        else:
            train_shards = all_shards[:-1]
            if num_shards > 0:
                train_shards = train_shards[:num_shards]
            shards_to_download = train_shards + [all_shards[-1]]
        shard_args = [(fname, url, data_dir) for fname, url in shards_to_download]

    # Count already downloaded
    existing = sum(1 for fname, _, d in shard_args if os.path.exists(os.path.join(d, fname)))
    if existing == len(shard_args):
        print(f"Data: all {len(shard_args)} shards already downloaded at {data_dir}")
        save_lang_meta(lang, val_filename, len(shard_args))
        return val_filename

    needed = len(shard_args) - existing
    print(f"Data: downloading {needed} shards ({existing} already exist)...")
    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_shard, shard_args)

    ok = sum(1 for r in results if r)
    print(f"Data: {ok}/{len(shard_args)} shards ready at {data_dir}")
    save_lang_meta(lang, val_filename, len(shard_args))
    return val_filename

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def list_parquet_files(data_dir=None):
    """Return sorted list of parquet file paths in the data directory."""
    if data_dir is None:
        data_dir = DATA_DIR
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(data_dir, f) for f in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000, data_dir=None, val_filename=None):
    """Yield documents from training split (all shards except pinned val shard)."""
    if data_dir is None:
        data_dir = DATA_DIR
    if val_filename is None:
        val_filename = VAL_FILENAME
    all_paths = list_parquet_files(data_dir)
    parquet_paths = [p for p in all_paths if not p.endswith(val_filename)] or all_paths
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer(data_dir=None, tokenizer_dir=None, val_filename=None):
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    if data_dir is None:
        data_dir = DATA_DIR
    if tokenizer_dir is None:
        tokenizer_dir = TOKENIZER_DIR
    if val_filename is None:
        val_filename = VAL_FILENAME

    tokenizer_pkl = os.path.join(tokenizer_dir, "tokenizer.pkl")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {tokenizer_dir}")
        return

    os.makedirs(tokenizer_dir, exist_ok=True)

    parquet_files = list_parquet_files(data_dir)
    if len(parquet_files) < 1:
        print("Tokenizer: no data shards found. Download data first.")
        sys.exit(1)

    # --- Train with rustbpe ---
    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(
        text_iterator(data_dir=data_dir, val_filename=val_filename),
        vocab_size_no_special,
        pattern=SPLIT_PATTERN,
    )

    # Build tiktoken encoding from trained merges
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    # Save tokenizer
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # --- Build token_bytes lookup for BPB evaluation ---
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    test = "Hello world! Numbers: 123."
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu", tokenizer_dir=None):
    if tokenizer_dir is None:
        tokenizer_dir = TOKENIZER_DIR
    path = os.path.join(tokenizer_dir, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128, data_dir=None, val_filename=None):
    """Infinite iterator over document batches from parquet files."""
    if data_dir is None:
        data_dir = DATA_DIR
    if val_filename is None:
        val_filename = VAL_FILENAME
    parquet_paths = list_parquet_files(data_dir)
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(data_dir, val_filename)
    if split == "train":
        train_paths = [p for p in parquet_paths if p != val_path]
        parquet_paths = train_paths or parquet_paths  # fallback: single-shard languages
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


def _get_device():
    """Auto-detect the best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def make_dataloader(tokenizer, B, T, split, buffer_size=1000, device=None, data_dir=None, val_filename=None):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
    if device is None:
        device = _get_device()

    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split, data_dir=data_dir, val_filename=val_filename)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    # pin_memory only works for CUDA
    if device == "cuda":
        cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    else:
        cpu_buffer = torch.empty(2 * B * T, dtype=torch.long)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size, device=None, seq_len=None, max_steps=None, data_dir=None, val_filename=None):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    seq_len defaults to MAX_SEQ_LEN (2048). Pass the training seq_len
    when the model was trained on a shorter context (e.g. MPS with 512).
    max_steps caps the number of eval steps (useful for short time budgets).
    """
    if device is None:
        device = _get_device()
    if seq_len is None:
        seq_len = MAX_SEQ_LEN
    # Infer tokenizer_dir from data_dir location if possible
    tokenizer_dir = None
    if data_dir is not None:
        tokenizer_dir = os.path.join(os.path.dirname(data_dir), "tokenizer")
    token_bytes = get_token_bytes(device=device, tokenizer_dir=tokenizer_dir)
    val_loader = make_dataloader(tokenizer, batch_size, seq_len, "val", device=device, data_dir=data_dir, val_filename=val_filename)
    steps = EVAL_TOKENS // (batch_size * seq_len)
    if max_steps is not None:
        steps = min(steps, max_steps)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch")
    parser.add_argument("--language", type=str, default="en", choices=list(LANGUAGE_CONFIGS.keys()),
                        help="Language to prepare data for")
    parser.add_argument("--num-shards", type=int, default=10,
                        help="Number of training shards to download (-1 = all). Val shard is always pinned.")
    parser.add_argument("--download-workers", type=int, default=8,
                        help="Number of parallel download workers")
    args = parser.parse_args()

    lang = args.language
    lang_cfg = LANGUAGE_CONFIGS[lang]
    data_dir, tokenizer_dir = get_lang_dirs(lang)
    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards

    print(f"Language:        {lang_cfg['name']} ({lang})")
    print(f"Cache directory: {os.path.join(CACHE_DIR, lang)}")
    print()

    # Step 1: Download data
    val_filename = download_data(lang=lang, num_shards=num_shards, download_workers=args.download_workers)
    print()

    # Step 2: Train tokenizer
    train_tokenizer(data_dir=data_dir, tokenizer_dir=tokenizer_dir, val_filename=val_filename)
    print()
    print("Done! Ready to train.")

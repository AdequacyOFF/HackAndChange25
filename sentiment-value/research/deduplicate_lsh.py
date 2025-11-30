#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Set

import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm.auto import tqdm
import re


LOGGER = logging.getLogger(__name__)


def setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def normalize_text(text: str) -> str:
    """Простая нормализация текста (нижний регистр, пробелы, кавычки, тире)."""
    text = text.lower()
    text = text.replace("ё", "е")

    # унификация кавычек и тире
    text = re.sub(r"[“”«»]", '"', text)
    text = re.sub(r"[’‘]", "'", text)
    text = re.sub(r"[—–−]", "-", text)

    # схлопывание пробелов
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def char_shingles(text: str, n: int) -> Set[str]:
    """Множество символьных n-грамм."""
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def build_minhash(shingles: Iterable[str], num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in shingles:
        m.update(sh.encode("utf-8"))
    return m


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Deduplicate a text dataset using MinHash LSH over character n-grams.\n"
            "Input: .parquet file with column 'text'. Output: cleaned .parquet."
        )
    )
    p.add_argument("--input", type=Path, required=True, help="Input .parquet path with 'text' column")
    p.add_argument("--output", type=Path, required=True, help="Output .parquet path for deduplicated data")

    p.add_argument(
        "--ngram",
        type=int,
        default=5,
        help="Character n-gram size for shingles (default: 5)",
    )
    p.add_argument(
        "--num-perm",
        type=int,
        default=128,
        help="Number of permutations for MinHash (default: 128)",
    )
    p.add_argument(
        "--lsh-threshold",
        type=float,
        default=0.8,
        help="Threshold for MinHashLSH (approximate Jaccard) (default: 0.8)",
    )
    p.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.9,
        help="Exact Jaccard threshold to treat as duplicate (default: 0.9)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Verbosity level: -v (info), -vv (debug)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    LOGGER.info("Reading input parquet from %s", args.input)
    df = pd.read_parquet(args.input)

    if "text" not in df.columns:
        raise ValueError("Input parquet must contain a 'text' column")

    texts = df["text"].astype(str).tolist()
    n_docs = len(texts)
    LOGGER.info("Loaded %d texts", n_docs)

    # LSH по MinHash
    LOGGER.info(
        "Initializing MinHashLSH: num_perm=%d, lsh_threshold=%.3f, ngram=%d, jaccard_threshold=%.3f",
        args.num_perm,
        args.lsh_threshold,
        args.ngram,
        args.jaccard_threshold,
    )
    lsh = MinHashLSH(threshold=args.lsh_threshold, num_perm=args.num_perm)

    # Храним нормализованный текст для точного Jaccard
    stored_norm_texts: dict[int, str] = {}

    keep_mask = [False] * n_docs
    n_duplicates = 0

    for idx, raw_text in tqdm(
        enumerate(texts),
        total=n_docs,
        desc="Deduplicating",
    ):
        norm = normalize_text(raw_text)
        sh = char_shingles(norm, args.ngram)

        # Если текст пустой после нормализации – можно решить,
        # оставлять ли его. Здесь оставляем первый, остальные дропаем.
        m = build_minhash(sh, args.num_perm)
        candidates = lsh.query(m)

        is_dup = False
        if candidates:
            # Доп. фильтрация по точному Jaccard
            for cand_id in candidates:
                other_norm = stored_norm_texts[cand_id]
                other_sh = char_shingles(other_norm, args.ngram)
                jac = jaccard(sh, other_sh)
                if jac >= args.jaccard_threshold:
                    is_dup = True
                    break

        if is_dup:
            n_duplicates += 1
            continue

        # Новый уникальный текст – вставляем в LSH и помечаем как keep
        lsh.insert(idx, m)
        stored_norm_texts[idx] = norm
        keep_mask[idx] = True

    LOGGER.info("Found %d duplicates", n_duplicates)
    LOGGER.info("Remaining %d unique texts", sum(keep_mask))

    df_clean = df[keep_mask].reset_index(drop=True)

    LOGGER.info("Writing cleaned parquet to %s", args.output)
    df_clean.to_parquet(args.output, index=False)
    LOGGER.info("Done")


if __name__ == "__main__":
    main()

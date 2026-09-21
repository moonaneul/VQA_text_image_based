from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat


CHOICES = ["a", "b", "c", "d"]
KEYWORDS = [
    "적힌", "이름", "얼마", "문구", "가격", "번호", "전화번호", "메뉴",
    "간판", "글자", "써", "표지판", "영수증", "원", "몇", "색", "어디",
]


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    return re.sub(r"\s+", " ", text)


def question_template(value: object) -> str:
    text = normalize_text(value)
    text = re.sub(r"\d+(?:[.,:/-]\d+)*", "<num>", text)
    text = re.sub(r"[\"'‘’“”]", "", text)
    return text


def classify_question(question: str) -> str:
    q = normalize_text(question)
    if any(token in q for token in ["전화번호", "연락처"]):
        return "phone"
    if any(token in q for token in ["가격", "얼마", "금액", "원인가", "몇 원"]):
        return "price"
    if "메뉴" in q:
        return "menu"
    if any(token in q for token in ["적힌", "문구", "글자", "써 있", "상호명", "이름", "간판", "표지판"]):
        return "scene_text"
    if any(token in q for token in ["몇 개", "몇 명", "몇 마리", "개수", "수는"]):
        return "count"
    if any(token in q for token in ["무슨 색", "어떤 색", "색깔"]):
        return "color"
    if any(token in q for token in ["어디", "위치", "왼쪽", "오른쪽", "앞에", "뒤에"]):
        return "spatial"
    return "other"


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    ordered = sorted(values)
    def percentile(p: float) -> float:
        index = (len(ordered) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        if lo == hi:
            return float(ordered[lo])
        return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo))
    return {
        "count": len(values),
        "min": float(ordered[0]),
        "p25": percentile(0.25),
        "median": percentile(0.5),
        "p75": percentile(0.75),
        "max": float(ordered[-1]),
        "mean": float(statistics.fmean(values)),
    }


def resolve_image(data_dir: Path, raw_path: object) -> Path:
    relative = Path(str(raw_path).replace("\\", "/"))
    return relative if relative.is_absolute() else data_dir / relative


def dhash(image: Image.Image) -> str:
    gray = ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.get_flattened_data())
    bits = []
    for row in range(8):
        offset = row * 9
        bits.extend(pixels[offset + col] > pixels[offset + col + 1] for col in range(8))
    value = sum((1 << index) for index, bit in enumerate(bits) if bit)
    return f"{value:016x}"


def image_metadata(data_dir: Path, frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    records: list[dict] = []
    failures: list[dict] = []
    for split, frame in frames.items():
        for row in frame.itertuples(index=False):
            path = resolve_image(data_dir, row.path)
            record = {"split": split, "id": row.id, "path": str(row.path), "exists": path.is_file()}
            if not path.is_file():
                failures.append(record)
                records.append(record)
                continue
            try:
                file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    sample = image.convert("L")
                    sample.thumbnail((128, 128))
                    record.update(
                        width=width,
                        height=height,
                        aspect_ratio=width / height,
                        megapixels=(width * height) / 1_000_000,
                        mode=image.mode,
                        format=image.format,
                        file_bytes=path.stat().st_size,
                        mean_luma=ImageStat.Stat(sample).mean[0],
                        sha256=file_hash,
                        dhash=dhash(image),
                    )
            except Exception as exc:  # keep the audit running and report the exact file
                record["error"] = f"{type(exc).__name__}: {exc}"
                failures.append(record)
            records.append(record)

    metadata = pd.DataFrame.from_records(records)
    valid = metadata[metadata["exists"] & metadata.get("error", pd.Series(index=metadata.index, dtype=object)).isna()].copy()
    summary: dict[str, object] = {"failures": failures, "splits": {}}
    for split, group in valid.groupby("split"):
        summary["splits"][split] = {
            "count": int(len(group)),
            "width": distribution(group["width"].astype(float).tolist()),
            "height": distribution(group["height"].astype(float).tolist()),
            "aspect_ratio": distribution(group["aspect_ratio"].astype(float).tolist()),
            "megapixels": distribution(group["megapixels"].astype(float).tolist()),
            "file_kb": distribution((group["file_bytes"] / 1024).astype(float).tolist()),
            "portrait_pct": float((group["aspect_ratio"] < 0.95).mean() * 100),
            "squareish_pct": float(group["aspect_ratio"].between(0.95, 1.05).mean() * 100),
            "landscape_pct": float((group["aspect_ratio"] > 1.05).mean() * 100),
            "formats": {str(k): int(v) for k, v in group["format"].value_counts().items()},
            "modes": {str(k): int(v) for k, v in group["mode"].value_counts().items()},
        }

    exact_groups = valid.groupby("sha256").filter(lambda g: len(g) > 1)
    perceptual_groups = valid.groupby("dhash").filter(lambda g: len(g) > 1)
    summary["duplicate_images"] = {
        "exact_duplicate_rows": int(len(exact_groups)),
        "exact_duplicate_hashes": int(exact_groups["sha256"].nunique()) if len(exact_groups) else 0,
        "exact_cross_split_hashes": int(sum(group["split"].nunique() > 1 for _, group in exact_groups.groupby("sha256"))),
        "dhash_duplicate_rows": int(len(perceptual_groups)),
        "dhash_duplicate_hashes": int(perceptual_groups["dhash"].nunique()) if len(perceptual_groups) else 0,
        "dhash_cross_split_hashes": int(sum(group["split"].nunique() > 1 for _, group in perceptual_groups.groupby("dhash"))),
    }
    return metadata, summary


def csv_summary(data_dir: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    paths = {name: data_dir / f"{name}.csv" for name in ["train", "dev", "test"]}
    paths["sample_submission"] = data_dir / "sample_submission.csv"
    frames = {name: pd.read_csv(path, encoding="utf-8-sig") for name, path in paths.items()}
    summary: dict[str, object] = {"files": {}, "overlap": {}, "dev_responses": {}}

    for name, frame in frames.items():
        entry: dict[str, object] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "duplicate_ids": int(frame["id"].duplicated().sum()) if "id" in frame else None,
            "missing_by_column": {col: int(count) for col, count in frame.isna().sum().items()},
        }
        if "path" in frame:
            entry["duplicate_paths"] = int(frame["path"].duplicated().sum())
            entry["missing_images"] = int(sum(not resolve_image(data_dir, p).is_file() for p in frame["path"]))
        if "question" in frame:
            normalized = frame["question"].map(normalize_text)
            templates = frame["question"].map(question_template)
            categories = frame["question"].map(classify_question)
            entry["question_chars"] = distribution(normalized.map(len).astype(float).tolist())
            entry["unique_questions"] = int(normalized.nunique())
            entry["unique_templates"] = int(templates.nunique())
            entry["category_counts"] = {str(k): int(v) for k, v in categories.value_counts().items()}
            entry["keyword_counts"] = {keyword: int(normalized.str.contains(keyword, regex=False).sum()) for keyword in KEYWORDS}
        summary["files"][name] = entry

    train = frames["train"]
    test = frames["test"]
    dev = frames["dev"]
    summary["files"]["train"]["answer_counts"] = {str(k): int(v) for k, v in train["answer"].value_counts().sort_index().items()}

    train_q = set(train["question"].map(normalize_text))
    train_t = set(train["question"].map(question_template))
    for name, frame in {"dev": dev, "test": test}.items():
        q = frame["question"].map(normalize_text)
        t = frame["question"].map(question_template)
        summary["overlap"][f"train_{name}"] = {
            "rows_exact_question_seen_in_train": int(q.isin(train_q).sum()),
            "row_pct_exact": float(q.isin(train_q).mean() * 100),
            "unique_exact_question_intersection": int(len(train_q & set(q))),
            "rows_template_seen_in_train": int(t.isin(train_t).sum()),
            "row_pct_template": float(t.isin(train_t).mean() * 100),
            "unique_template_intersection": int(len(train_t & set(t))),
        }

    response_cols = [col for col in dev.columns if re.fullmatch(r"answer\d+", col, flags=re.IGNORECASE)]
    consensus = Counter()
    majority_ratios: list[float] = []
    tie_count = 0
    invalid = Counter()
    nonblank_counts = Counter()
    top_agreement_counts = Counter()
    unanimous_nonblank = 0
    for row in dev[response_cols].itertuples(index=False, name=None):
        responses = [normalize_text(value) for value in row if pd.notna(value) and normalize_text(value)]
        invalid.update(value for value in responses if value not in CHOICES)
        valid = [value for value in responses if value in CHOICES]
        nonblank_counts[len(valid)] += 1
        if not valid:
            consensus["no_valid_response"] += 1
            continue
        counts = Counter(valid)
        top = counts.most_common()
        max_count = top[0][1]
        top_agreement_counts[max_count] += 1
        majority_ratios.append(max_count / len(valid))
        winners = [choice for choice, count in top if count == max_count]
        if len(winners) > 1:
            tie_count += 1
        else:
            consensus[winners[0]] += 1
        if len(valid) >= 2 and len(counts) == 1:
            unanimous_nonblank += 1
    summary["dev_responses"] = {
        "columns": response_cols,
        "nonblank_response_count_distribution": {str(k): int(v) for k, v in sorted(nonblank_counts.items())},
        "top_agreement_count_distribution": {str(k): int(v) for k, v in sorted(top_agreement_counts.items())},
        "invalid_values": dict(invalid),
        "unique_majority_winner_counts": dict(consensus),
        "tie_rows": tie_count,
        "unanimous_among_available_rows": unanimous_nonblank,
        "majority_ratio": distribution(majority_ratios),
        "usable_strict_4_of_5_rows": int(sum(1 for row in dev[response_cols].itertuples(index=False, name=None)
                                             if Counter([normalize_text(v) for v in row if pd.notna(v)]).most_common(1)
                                             and Counter([normalize_text(v) for v in row if pd.notna(v)]).most_common(1)[0][1] >= 4)),
        "usable_strict_3_agreement_rows": int(sum(1 for row in dev[response_cols].itertuples(index=False, name=None)
                                                  if Counter([normalize_text(v) for v in row if pd.notna(v)]).most_common(1)
                                                  and Counter([normalize_text(v) for v in row if pd.notna(v)]).most_common(1)[0][1] >= 3)),
    }
    return frames, summary


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/arial.ttf")]:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def wrapped_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def make_contact_sheet(data_dir: Path, train: pd.DataFrame, output_path: Path, manifest_path: Path, seed: int) -> None:
    frame = train.copy()
    frame["category"] = frame["question"].map(classify_question)
    rng = random.Random(seed)
    categories = ["phone", "price", "menu", "scene_text", "count", "color", "spatial", "other"]
    selected: list[pd.Series] = []
    for category in categories:
        indices = frame.index[frame["category"] == category].tolist()
        if indices:
            selected.append(frame.loc[rng.choice(indices)])

    tile_w, tile_h = 800, 430
    columns = 2
    rows = math.ceil(len(selected) / columns)
    canvas = Image.new("RGB", (tile_w * columns, tile_h * rows), "white")
    title_font = load_font(24)
    body_font = load_font(18)
    small_font = load_font(16)
    manifest: list[dict] = []
    for idx, row in enumerate(selected):
        x0 = (idx % columns) * tile_w
        y0 = (idx // columns) * tile_h
        tile = Image.new("RGB", (tile_w, tile_h), "#f4f6f8")
        image_path = resolve_image(data_dir, row["path"])
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((440, 380), Image.Resampling.LANCZOS)
            image_x = 10 + (440 - image.width) // 2
            image_y = 40 + (380 - image.height) // 2
            tile.paste(image, (image_x, image_y))
        draw = ImageDraw.Draw(tile)
        draw.text((10, 8), f"{row['category']} | {row['id']} | answer={row['answer']}", fill="#0f172a", font=title_font)
        text_x = 470
        y = 48
        for line in wrapped_lines(draw, str(row["question"]), body_font, 310):
            draw.text((text_x, y), line, fill="#111827", font=body_font)
            y += 27
        y += 10
        for choice in CHOICES:
            prefix = "*" if row["answer"] == choice else " "
            choice_text = f"{prefix} {choice}. {row[choice]}"
            for line in wrapped_lines(draw, choice_text, small_font, 310):
                draw.text((text_x, y), line, fill="#334155", font=small_font)
                y += 22
            y += 4
        canvas.paste(tile, (x0, y0))
        manifest.append({key: row[key] for key in ["id", "path", "question", "a", "b", "c", "d", "answer", "category"]})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    pd.DataFrame(manifest).to_csv(manifest_path, index=False, encoding="utf-8-sig")


def render_markdown(csv_stats: dict, image_stats: dict) -> str:
    lines = ["# VQA dataset audit", "", "## Dataset integrity", ""]
    for split in ["train", "dev", "test", "sample_submission"]:
        entry = csv_stats["files"][split]
        lines.append(f"- {split}: {entry['rows']:,} rows, duplicate ids={entry['duplicate_ids']}, missing={sum(entry['missing_by_column'].values()):,}")
    lines.extend(["", "## Train labels", "", f"- {csv_stats['files']['train']['answer_counts']}"])
    lines.extend(["", "## Question overlap", ""])
    for name, entry in csv_stats["overlap"].items():
        lines.append(f"- {name}: exact={entry['row_pct_exact']:.1f}% ({entry['rows_exact_question_seen_in_train']:,} rows), template={entry['row_pct_template']:.1f}% ({entry['rows_template_seen_in_train']:,} rows)")
    lines.extend(["", "## Question categories", ""])
    for split in ["train", "dev", "test"]:
        lines.append(f"- {split}: {csv_stats['files'][split]['category_counts']}")
    lines.extend(["", "## Dev response quality", ""])
    dev = csv_stats["dev_responses"]
    lines.append(f"- Available response counts: {dev['nonblank_response_count_distribution']}")
    lines.append(f"- Top agreement counts: {dev['top_agreement_count_distribution']}")
    lines.append(f"- Ties: {dev['tie_rows']:,}; unanimous among available (at least 2): {dev['unanimous_among_available_rows']:,}; strict 4-of-5: {dev['usable_strict_4_of_5_rows']:,}")
    lines.extend(["", "## Image metadata", ""])
    for split in ["train", "dev", "test"]:
        entry = image_stats["splits"][split]
        lines.append(
            f"- {split}: {entry['count']:,} images; median {entry['width']['median']:.0f}x{entry['height']['median']:.0f}; "
            f"median {entry['megapixels']['median']:.2f} MP; portrait/square/landscape "
            f"{entry['portrait_pct']:.1f}%/{entry['squareish_pct']:.1f}%/{entry['landscape_pct']:.1f}%"
        )
    dup = image_stats["duplicate_images"]
    lines.append(f"- Exact cross-split duplicate hashes: {dup['exact_cross_split_hashes']}; perceptual-hash cross-split groups: {dup['dhash_cross_split_hashes']}")
    if image_stats["failures"]:
        lines.append(f"- Image read failures: {len(image_stats['failures'])}")
    else:
        lines.append("- Image read failures: 0")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the SSAFY multiple-choice VQA dataset without modifying source data.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, csv_stats = csv_summary(data_dir)
    metadata, image_stats = image_metadata(data_dir, {name: frames[name] for name in ["train", "dev", "test"]})
    metadata.to_csv(output_dir / "image_metadata.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    make_contact_sheet(data_dir, frames["train"], output_dir / "question_type_samples.jpg", output_dir / "question_type_samples.csv", args.seed)
    combined = {"csv": csv_stats, "images": image_stats}
    (output_dir / "eda_summary.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "eda_report.md").write_text(render_markdown(csv_stats, image_stats), encoding="utf-8")
    print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

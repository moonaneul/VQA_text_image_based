from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

FAILURE_TYPES = [
    "",
    "ocr_recognition",
    "target_localization",
    "number_text_binding",
    "exact_string_confusion",
    "question_understanding",
    "visual_spatial_reasoning",
    "ambiguity_or_label_issue",
    "other",
]

LIKELY_FIXES = [
    "",
    "higher_resolution",
    "target_crop_zoom",
    "tiling",
    "external_ocr",
    "bbox_localization",
    "choice_scoring",
    "qlora",
    "label_review",
    "none_or_unclear",
]

YES_NO_UNSURE = ["", "yes", "no", "unsure"]


def esc(value) -> str:
    if pd.isna(value):
        return ""
    return html.escape(str(value), quote=True)


def image_to_data_uri(path: Path, max_side: int) -> str:
    with Image.open(path) as src:
        image = ImageOps.exif_transpose(src).convert("RGB")
        image.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def option_tags(values: list[str]) -> str:
    return "".join(
        '<option value="' + html.escape(v, quote=True) + '">' + (html.escape(v) if v else "—") + "</option>"
        for v in values
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a visual HTML gallery for B0 error review.")
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=1000)
    args = parser.parse_args()

    frame = pd.read_csv(args.review_csv, encoding="utf-8-sig")

    cards = []
    for idx, row in frame.iterrows():
        image_path = args.data_dir / Path(str(row["path"]).replace("\\", "/"))
        if image_path.is_file():
            image_html = '<img src="' + image_to_data_uri(image_path, args.max_side) + '" alt="error image">'
        else:
            image_html = '<div class="missing">Image missing: ' + esc(image_path) + "</div>"

        options = []
        for key in ["a", "b", "c", "d"]:
            classes = ["option"]
            if str(row["answer"]).lower() == key:
                classes.append("answer")
            if str(row["prediction"]).lower() == key:
                classes.append("prediction")
            options.append(
                '<div class="' + " ".join(classes) + '"><strong>' + key + "</strong> " + esc(row[key]) + "</div>"
            )

        cards.append(
            '<section class="card" data-category="' + esc(row["category"]) + '">'
            '<div class="card-head"><strong>#' + str(idx + 1) + " · id " + esc(row["id"]) + '</strong>'
            '<span class="badge">' + esc(row["category"]) + "</span></div>"
            '<div class="grid"><div class="visual">' + image_html + '</div><div class="content">'
            '<div class="question">' + esc(row["question"]) + "</div>"
            '<div class="options">' + "".join(options) + "</div>"
            '<div class="answer-line"><span>GT <strong>' + esc(row["answer"]) + '</strong></span>'
            '<span>Pred <strong>' + esc(row["prediction"]) + '</strong></span>'
            '<span>Raw <code>' + esc(row.get("raw_output", "")) + "</code></span></div>"
            '<div class="review-grid">'
            '<label>Failure type<select class="failure_type">' + option_tags(FAILURE_TYPES) + "</select></label>"
            '<label>Text visible to human?<select class="text_visible_to_human">' + option_tags(YES_NO_UNSURE) + "</select></label>"
            '<label>Correct text present in image?<select class="correct_text_present_in_image">' + option_tags(YES_NO_UNSURE) + "</select></label>"
            '<label>Likely fix<select class="likely_fix">' + option_tags(LIKELY_FIXES) + "</select></label>"
            '</div><label>Review notes<textarea class="review_notes" rows="3" placeholder="왜 틀렸는지 짧게 기록"></textarea></label>'
            "</div></div></section>"
        )

    records_json = json.dumps(frame.to_dict(orient="records"), ensure_ascii=False)
    categories = sorted(frame["category"].dropna().astype(str).unique())
    category_options = "".join('<option value="' + esc(c) + '">' + esc(c) + "</option>" for c in categories)

    page = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>B0 Error Root-cause Review</title>
<style>
:root{--bg:#f7f7f8;--card:#fff;--text:#171717;--muted:#666;--line:#ddd;--good:#e8f7ee}
*{box-sizing:border-box}body{margin:0;font-family:Arial,"Noto Sans KR",sans-serif;background:var(--bg);color:var(--text)}
header{position:sticky;top:0;z-index:10;background:#fff;border-bottom:1px solid var(--line);padding:14px 20px}
.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.toolbar button,.toolbar select{min-height:38px;padding:7px 10px}
main{max-width:1400px;margin:0 auto;padding:18px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;margin-bottom:16px;overflow:hidden}
.card.done{border-color:#71b587}.card-head{display:flex;justify-content:space-between;padding:10px 14px;border-bottom:1px solid var(--line)}
.badge{background:#eee;padding:3px 8px;border-radius:999px;font-size:12px}.grid{display:grid;grid-template-columns:minmax(280px,46%) 1fr;gap:16px;padding:14px}
.visual img{width:100%;max-height:650px;object-fit:contain;background:#111;border-radius:8px}.question{font-size:18px;font-weight:700;line-height:1.45;margin-bottom:10px}
.option{padding:8px 10px;border:1px solid var(--line);border-radius:8px;margin:6px 0}.option.answer{background:var(--good)}.option.prediction{outline:2px solid #d66}
.option.answer.prediction{outline:2px solid #4a8}.answer-line{display:flex;gap:14px;flex-wrap:wrap;padding:10px 0;color:var(--muted)}
.review-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}label{display:flex;flex-direction:column;gap:5px;font-size:13px;color:var(--muted)}
select,textarea{width:100%;font:inherit;padding:8px;border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--text)}textarea{resize:vertical}
.legend{font-size:13px;color:var(--muted);margin-top:8px}.missing{padding:20px;color:#b00}
@media(max-width:800px){.grid{grid-template-columns:1fr}.review-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
<div class="toolbar">
<strong>B0 Error Review</strong>
<span id="progress">0 / COUNT labeled</span>
<select id="filter"><option value="">All categories</option>CATEGORY_OPTIONS</select>
<button type="button" id="next">Next unlabeled</button>
<button type="button" id="export">Export reviewed CSV</button>
</div>
<div class="legend">초록 배경=정답 선택지 · 붉은 테두리=모델 예측 · Failure type은 primary cause 하나만 선택</div>
</header>
<main>CARDS</main>
<script>
var original = RECORDS_JSON;
var cards = Array.from(document.querySelectorAll(".card"));
var progress = document.getElementById("progress");

function updateProgress(){
  var done = cards.filter(function(card){return !!card.querySelector(".failure_type").value;}).length;
  progress.textContent = done + " / " + cards.length + " labeled";
  cards.forEach(function(card){card.classList.toggle("done", !!card.querySelector(".failure_type").value);});
}
document.querySelectorAll("select,textarea").forEach(function(el){el.addEventListener("change", updateProgress);});
document.getElementById("filter").addEventListener("change", function(e){
  var value = e.target.value;
  cards.forEach(function(card){card.style.display = (!value || card.dataset.category === value) ? "" : "none";});
});
document.getElementById("next").addEventListener("click", function(){
  var target = cards.find(function(card){return card.style.display !== "none" && !card.querySelector(".failure_type").value;});
  if(target){target.scrollIntoView({behavior:"smooth",block:"start"});}
});
function csvEscape(v){
  var s = (v === null || v === undefined) ? "" : String(v);
  return '"' + s.replaceAll('"','""') + '"';
}
document.getElementById("export").addEventListener("click", function(){
  var extra = ["failure_type","text_visible_to_human","correct_text_present_in_image","likely_fix","review_notes"];
  var base = Object.keys(original[0] || {});
  var cols = Array.from(new Set(base.concat(extra)));
  var lines = [cols.map(csvEscape).join(",")];
  cards.forEach(function(card,i){
    var row = Object.assign({}, original[i]);
    extra.forEach(function(key){row[key] = card.querySelector("." + key).value;});
    lines.push(cols.map(function(c){return csvEscape(row[c]);}).join(","));
  });
  var blob = new Blob(["\\ufeff" + lines.join("\\r\\n")], {type:"text/csv;charset=utf-8"});
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  a.href = url;
  a.download = "B0_error_review_filled.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});
updateProgress();
</script>
</body>
</html>"""

    page = (
        page.replace("COUNT", str(len(frame)))
        .replace("CATEGORY_OPTIONS", category_options)
        .replace("CARDS", "".join(cards))
        .replace("RECORDS_JSON", records_json)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(f"wrote visual review gallery -> {args.output}")
    print(f"errors: {len(frame)}")


if __name__ == "__main__":
    main()

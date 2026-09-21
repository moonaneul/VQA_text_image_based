# VQA dataset audit

## Dataset integrity

- train: 6,714 rows, duplicate ids=0, missing=0
- dev: 2,683 rows, duplicate ids=0, missing=434
- test: 6,714 rows, duplicate ids=0, missing=0
- sample_submission: 6,714 rows, duplicate ids=0, missing=6,714

## Train labels

- {'a': 1700, 'b': 1644, 'c': 1716, 'd': 1654}

## Question overlap

- train_dev: exact=7.1% (190 rows), template=7.4% (199 rows)
- train_test: exact=13.2% (888 rows), template=13.4% (903 rows)

## Question categories

- train: {'scene_text': 3339, 'other': 1747, 'price': 898, 'spatial': 287, 'phone': 217, 'menu': 159, 'count': 52, 'color': 15}
- dev: {'scene_text': 1108, 'other': 799, 'price': 388, 'spatial': 146, 'menu': 126, 'phone': 90, 'count': 17, 'color': 9}
- test: {'scene_text': 3326, 'other': 1816, 'price': 865, 'spatial': 295, 'phone': 194, 'menu': 159, 'count': 37, 'color': 22}

## Dev response quality

- Available response counts: {'0': 2, '3': 30, '4': 364, '5': 2287}
- Top agreement counts: {'1': 6, '2': 649, '3': 2026}
- Ties: 465; unanimous among available (at least 2): 17; strict 4-of-5: 0

## Image metadata

- train: 6,714 images; median 720x960; median 0.69 MP; portrait/square/landscape 71.8%/3.1%/25.1%
- dev: 2,683 images; median 720x960; median 0.69 MP; portrait/square/landscape 71.3%/3.5%/25.2%
- test: 6,714 images; median 720x960; median 0.69 MP; portrait/square/landscape 71.8%/3.5%/24.7%
- Exact cross-split duplicate hashes: 31; perceptual-hash cross-split groups: 38
- Image read failures: 0

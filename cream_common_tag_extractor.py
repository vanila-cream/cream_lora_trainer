"""
Cream Common Tag Extractor Node for ComfyUI

Extracts common tags from caption files in a dataset folder.
Useful for building prompts for sample generation after LoRA training.
"""

import os
import hashlib
from collections import Counter


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')


class CreamCommonTagExtractor:
    """
    Extracts tags that appear commonly across caption files in a dataset folder.
    Outputs a comma-separated string of common tags for use in prompt building,
    and a tag frequency report grouped by percentage.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dataset_path": ("STRING", {
                    "default": "",
                    "tooltip": "이미지와 .txt 캡션 파일이 있는 폴더 경로.",
                }),
                "threshold": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "공통 태그 판정 기준. 전체 캡션 중 이 비율 이상에 등장하는 태그를 공통 태그로 출력합니다. 1.0이면 모든 캡션에 있는 태그만 포함.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("common_tags", "tag_frequency")
    OUTPUT_TOOLTIPS = (
        "캡션들에 공통으로 포함된 태그 (쉼표 구분). 샘플 생성 프롬프트에 활용할 수 있습니다.",
        "모든 태그의 출현 빈도를 퍼센트 그룹별로 정리한 분석 결과.",
    )
    FUNCTION = "extract_common_tags"
    CATEGORY = "training"
    DESCRIPTION = "데이터셋 폴더의 캡션 파일들에서 공통 태그를 추출하고, 태그 빈도를 분석합니다."

    @classmethod
    def IS_CHANGED(s, dataset_path, threshold):
        """Rerun when caption file contents change."""
        if not dataset_path or not os.path.isdir(dataset_path.strip()):
            return ""

        dataset_path = dataset_path.strip()
        hasher = hashlib.sha256()
        hasher.update(str(threshold).encode('utf-8'))

        for filename in sorted(os.listdir(dataset_path)):
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue
            base_name = os.path.splitext(filename)[0]
            caption_file = os.path.join(dataset_path, f"{base_name}.txt")
            if os.path.exists(caption_file):
                hasher.update(caption_file.encode('utf-8'))
                hasher.update(str(os.path.getmtime(caption_file)).encode('utf-8'))

        return hasher.hexdigest()

    def _read_captions(self, dataset_path):
        """Read all caption files from the dataset folder."""
        captions = []
        for filename in sorted(os.listdir(dataset_path)):
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue
            base_name = os.path.splitext(filename)[0]
            caption_file = os.path.join(dataset_path, f"{base_name}.txt")
            if os.path.exists(caption_file):
                with open(caption_file, 'r', encoding='utf-8') as f:
                    captions.append(f.read().strip())
        return captions

    def _build_tag_frequency(self, tag_counter, total):
        """Build a grouped frequency report string.

        Format:
            [100%] tag1, tag2
            [ 90%] tag3, tag4
            [ 50%] tag5
        """
        # Group tags by their percentage (rounded to nearest 10%)
        groups = {}
        for tag, count in tag_counter.items():
            pct = round(count / total * 10) * 10  # round to nearest 10%
            pct = min(pct, 100)
            groups.setdefault(pct, []).append(tag)

        # Sort groups descending, tags alphabetically within each group
        lines = []
        for pct in sorted(groups.keys(), reverse=True):
            tags_sorted = sorted(groups[pct])
            lines.append(f"[{pct:3d}%] {', '.join(tags_sorted)}")

        return "\n".join(lines)

    def extract_common_tags(self, dataset_path, threshold):
        if not dataset_path or not dataset_path.strip():
            raise ValueError("dataset_path를 입력해주세요.")

        dataset_path = os.path.expanduser(dataset_path.strip())

        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"데이터셋 경로를 찾을 수 없습니다: {dataset_path}")

        captions = self._read_captions(dataset_path)

        if not captions:
            print("[Cream TagExtractor] 캡션 파일을 찾을 수 없습니다.")
            return ("", "")

        # Count tags across captions
        tag_counter = Counter()
        first_tags = []
        for i, caption in enumerate(captions):
            tags = [t.strip() for t in caption.split(",") if t.strip()]
            if i == 0:
                first_tags = tags
            tag_counter.update(set(tags))  # count once per caption

        total = len(captions)
        min_count = max(1, int(total * threshold))

        # Build frequency report (all tags)
        frequency_report = self._build_tag_frequency(tag_counter, total)
        print(f"[Cream TagExtractor] 태그 빈도 분석 완료 (총 {len(tag_counter)}개 태그, {total}개 캡션)")

        # Extract common tags (above threshold)
        common_set = {tag for tag, count in tag_counter.items() if count >= min_count}

        if not common_set:
            print("[Cream TagExtractor] 공통 태그가 없습니다.")
            return ("", frequency_report)

        # Preserve order from first caption
        ordered = [t for t in first_tags if t in common_set]
        remaining = [t for t in sorted(common_set) if t not in set(first_tags)]
        ordered.extend(remaining)

        result = ", ".join(ordered)
        print(f"[Cream TagExtractor] 공통 태그 ({len(ordered)}개, 기준: {threshold:.0%}): {result}")

        return (result, frequency_report)

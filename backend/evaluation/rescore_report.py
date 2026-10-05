"""使用现有逐案例结果离线重算新版评测指标。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.metrics import evaluate_case, summarize


def rescore(report_dir: str | Path) -> Path:
    """读取 report_dir/cases.jsonl，写出 summary_v2.json。"""

    directory = Path(report_dir)
    cases_path = Path("evaluation/datasets/shopcare_evaluation_cases.json")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    raw_results = [
        json.loads(line)
        for line in (directory / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(cases) != len(raw_results):
        raise ValueError(f"案例数不一致: cases={len(cases)} results={len(raw_results)}")
    evaluations = [evaluate_case(case, result) for case, result in zip(cases, raw_results)]
    summary = summarize(evaluations, raw_results)
    output = directory / "summary_v2.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"offline summary written: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="离线重算 ShopCare 评测报告")
    parser.add_argument("report_dir", help="包含 cases.jsonl 的历史评测目录")
    args = parser.parse_args()
    rescore(args.report_dir)


if __name__ == "__main__":
    main()

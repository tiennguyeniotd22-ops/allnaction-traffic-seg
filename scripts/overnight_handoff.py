"""Verify the winning local files and write a Vietnamese handoff beside the run directories."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile
from notebook_identity import TEAM_ID, verified_team


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def points(value):
    return f"{100 * value:.2f}".replace(".", ",")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path, help="Local directory containing verified runs and versioned input ZIPs")
    p.add_argument("repo", type=Path)
    args = p.parse_args()
    root, repo = args.root.resolve(), args.repo.resolve()
    board = read(repo / "experiments/overnight/verified-scores.json")
    eligible = [k for k, v in board.items() if v["resources"].get("team_id") == TEAM_ID
                and (root / k / "replay.json").is_file()]
    if not eligible:
        raise ValueError(f"No independently replayed submission for team {TEAM_ID}")
    best = max(eligible, key=lambda key: board[key]["scores"]["all140"]["map50_95"])
    entry = board[best]
    run = root / best
    report = read(run / "run_report.json")
    comp = read(run / "comparison.json")
    held = read(run / "held-score.json")
    resource = read(run / "resources.json")
    assert resource == entry["resources"] and held == entry["holdout"]
    checks = {"submission.zip": entry["zip_sha256"], "best.pt": entry["best_sha256"]}
    assert checks["submission.zip"] == report["submission"]["zip_sha256"] == comp["zip_sha256"] == held["zip_sha256"]
    assert checks["best.pt"] == report["train"]["best_sha256"] == comp["best_sha256"]
    assert all(digest(run / name) == sha for name, sha in checks.items())
    assert (run / "submission.zip").stat().st_size == report["submission"]["zip_bytes"] <= 42 * 2**20
    with ZipFile(run / "submission.zip") as z:
        assert z.namelist() == ["model.onnx"] and hashlib.sha256(z.read("model.onnx")).hexdigest() == entry["onnx_sha256"] == report["model"]["onnx_sha256"]
    assert report["model"]["architecture"] == "yolo26n-seg" and report["model"]["nc"] == 11
    assert verified_team(report, resource["notebook_sha256"]) == TEAM_ID and report["export_args"]["imgsz"] == 640
    assert entry["opsets"][""] == 17 and comp["validation_sha_verified"] == 140
    assert len(comp["onnx_parity"]) == 6 and report["environment"]["packages"]["ultralytics"] == "8.4.214"
    scores = entry["scores"]
    overall = scores["all140"]
    assert comp["scores"]["pilot"] == scores and overall["map50_95"] == report["val_estimate"]["map50_95"]
    replayed = read(run / "replay.json")
    assert replayed["zip_sha256"] == checks["submission.zip"] and replayed["dataset_sha256"] == resource["dataset_sha256"]
    assert abs(replayed["score"]["map50_95"] - overall["map50_95"]) < 1e-8
    assert sum(overall["n_gt_per_class"].values()) == 1588 and overall["n_images"] == 140
    versions = []
    for path in root.glob("*/manifest.json"):
        if (path.parent / "dataset.zip").is_file():
            record = read(path)
            if isinstance(record, dict) and record.get("dataset_sha256") == resource["dataset_sha256"]:
                versions.append(path.parent)
    if len(versions) != 1:
        raise ValueError(f"Expected one local dataset with the input digest, got {versions}")
    version = versions[0]
    assert digest(version / "dataset.zip") == resource["dataset_sha256"]
    assert digest(repo / "notebooks/train_and_export.ipynb") == resource["notebook_sha256"]
    val = read(root.parent / "audit-phase12/validation-manifest.json")
    assert len(val) == 140
    with ZipFile(version / "dataset.zip") as z:
        for row in val:
            name = row["file"]
            assert hashlib.sha256(z.read("images/val/" + name)).hexdigest() == row["image_sha256"]
            assert hashlib.sha256(z.read("labels/val/" + Path(name).stem + ".txt")).hexdigest() == row["labels_sha256"]
    cfg = report["train"]["train_args"]
    with (run / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    epochs = int(rows[-1]["epoch"])
    count = report["dataset"]["counts"]["train"]
    previous = read(root.parent / "round4/run_report.json")
    assert previous["dataset"]["counts"]["val"] == report["dataset"]["counts"]["val"]
    before = previous["val_estimate"]["per_class_ap"]
    assert previous["classes"] == report["classes"]
    decision = report.get("checkpoint_selection", {})
    snap = decision.get("snapshot_epoch")
    revision = resource["revision"]
    tools_revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    lines = [
        "# Bàn giao mô hình phân đoạn giao thông",
        "",
        f"**Tệp nộp của đội {TEAM_ID}** · [{best}/submission.zip]({best}/submission.zip). "
        f"Mask mAP trên 140 ảnh validation đạt **{points(overall['map50_95'])}/100**. "
        "Điểm public và private của tệp này chờ hệ thống cuộc thi chấm.",
        "",
        f"[Checkpoint PyTorch]({best}/best.pt) · [Báo cáo notebook]({best}/run_report.json) · "
        f"[Cấu hình]({best}/args.yaml) · [Lịch sử epoch]({best}/results.csv) · "
        f"[Log huấn luyện]({best}/training.log) · "
        f"[Dữ liệu dùng cho notebook]({version.name}/dataset.zip) · [Manifest dữ liệu]({version.name}/manifest.json)",
        "",
        "## Kết quả",
        "",
        "| Tập | Ảnh | Mask mAP | AP50 | AP75 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ("all140", "team", "drive", "BDD100K", "Vietnam-v5"):
        s = scores[key]
        lines.append(f"| {key} | {s['n_images']} | {points(s['map50_95'])} | {points(s['map50'])} | {points(s['map75'])} |")
    lines += [
        "",
        "Nhóm 60 gồm 45 ảnh BDD100K và 15 ảnh Việt Nam. Tập Drive có 80 ảnh. "
        "Evaluator tính 1.588 instance hợp lệ trong 1.821 nhãn validation. Lớp 3 chỉ có số đo ở holdout riêng.",
        "",
        "| ID | Lớp | Instance train vòng 4 → bản chọn | GT hợp lệ val | Mask AP vòng 4 | Mask AP bản chọn |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for cid, name in enumerate(report["classes"]):
        ap = overall["per_class_ap"].get(name)
        old_count = previous["dataset"]["per_class"][name]["train_instances"]
        new_count = report["dataset"]["per_class"][name]["train_instances"]
        lines.append(f"| {cid} | {name} | {old_count} → {new_count} | {overall['n_gt_per_class'][name]} | "
                     f"{points(before[name]) if name in before else 'Thiếu GT'} | {points(ap) if ap is not None else 'Thiếu GT'} |")
    lines += [
        "",
        f"Holdout Other vehicle có 7 instance trong 6 ảnh thuộc 3 nhóm cảnh, đạt {points(held['map50_95'])} AP. "
        "Tập Việt Nam có 15 ảnh, trong đó hai khung 62b71b và 645dbe có cảnh rất giống nhau. "
        "Việc chọn giữa nhiều lượt trên cùng validation có thể làm điểm local lạc quan.",
        "",
        "## Tái lập",
        "",
        f"Run `{best}`, bộ dữ liệu `{version.name}`, {count['images']} ảnh và {count['instances']} instance train. "
        f"Chạy {epochs}/{cfg['epochs']} epoch, batch {cfg['batch']}, optimizer {cfg['optimizer']}, lr0 {cfg['lr0']}, "
        f"mask_ratio {cfg['mask_ratio']}, copy_paste {cfg['copy_paste']}, patience {cfg['patience']}. "
        f"TEAM_ID và seed đều bằng {TEAM_ID}. GPU {resource['gpu']}, fit và native validation {resource['fit_and_native_val_seconds']:.1f} giây, "
        f"đỉnh PyTorch cấp phát {resource['peak_allocated_mib']:.0f} MiB. Latency CPU p50 trong lượt đo "
        f"{report['val_estimate']['latency_ms_p50']:.1f} ms/ảnh.",
        "",
        f"Git huấn luyện `{revision}`. Notebook giữ nguyên. Trên VM của nhóm, chạy từ `~/baseline-cvat300` "
        "với bộ scripts lấy từ revision trên và đường dẫn ZIP đã kiểm SHA. Dùng thư mục mới cho mỗi lượt chạy.",
        "",
        "```bash",
        f'git clone --no-checkout https://github.com/tiennguyeniotd22-ops/allnaction-traffic-seg.git "replay-code-{best}"',
        f'git -C "replay-code-{best}" checkout --detach {tools_revision}',
        f'SCRIPTS="$PWD/replay-code-{best}/scripts"',
        f'NOTEBOOK="$PWD/replay-code-{best}/notebooks/train_and_export.ipynb"',
        f'.venv/bin/python "$SCRIPTS/run_notebook_pilot.py" --notebook "$NOTEBOOK" --dataset overnight/{version.name}/dataset.zip '
        f'--run overnight/replay-{best} --revision {revision} --epochs {cfg["epochs"]} --patience {cfg["patience"]} --experiment {resource.get("experiment","control")}'
        + (f" --snapshot-every {snap}" if snap else ""),
        "```",
        "",
        "Để chấm lại trực tiếp tệp nộp trên CPU của VM, chạy từ cùng thư mục. "
        "Lệnh này dựng lại tập dữ liệu từ ZIP rồi đối chiếu fingerprint của bộ đã chuẩn hóa, 140 cặp SHA và AP từng lớp.",
        "",
        "```bash",
        f'.venv/bin/python "$SCRIPTS/replay_submission.py" --notebook "$NOTEBOOK" --dataset overnight/{version.name}/dataset.zip '
        f'--submission overnight/{best}/submission/submission.zip --report overnight/{best}/run_report.json '
        f'--resources overnight/{best}/resources.json --validation audit_phase12/validation-manifest.json --out overnight/replay-cpu-new-{best}',
        "```",
        "",
        f"ZIP {report['submission']['zip_bytes']:,} byte, chứa đúng một `model.onnx`. Export FP32, opset 17, batch 1, 640×640.",
        "",
        "| Tệp | SHA-256 |",
        "|---|---|",
    ]
    if snap:
        lines.insert(lines.index("## Tái lập") + 2, f"Checkpoint được chọn ở cuối epoch {snap}.\n")
    if decision.get("mean_sources"):
        sources = decision["mean_sources"]
        start = lines.index("## Tái lập") + 2
        end = lines.index("```", lines.index("```bash", start))
        versions_by_digest = {}
        for path in root.glob("*/manifest.json"):
            if (path.parent / "dataset.zip").is_file():
                item = read(path)
                if isinstance(item, dict):
                    versions_by_digest[item.get("dataset_sha256")] = path.parent
        reproduction = [
            f"Mô hình là trung bình trọng số của {len(sources)} checkpoint đội {TEAM_ID}. "
            "Cùng backbone và thứ tự lớp. Mỗi bản đóng góp trọng số bằng nhau. Các số liệu ảnh train ở bảng trên "
            "mô tả bộ dữ liệu của lượt tham chiếu. Các bộ dữ liệu bên dưới có ảnh chung.",
            "", "| Lượt nguồn | Bộ dữ liệu | Ảnh train | Epoch | SHA-256 best.pt |",
            "|---|---|---:|---:|---|",
        ]
        commands = []
        for item in sources:
            name = item["run"]
            source_folder = versions_by_digest[item["dataset_sha256"]]
            assert digest(source_folder / "dataset.zip") == item["dataset_sha256"]
            assert digest(root / name / "best.pt") == item["weights_sha256"]
            rows = read(root / name / "run_report.json")["dataset"]["counts"]["train"]["images"]
            reproduction.append(f"| {name} | [{source_folder.name}]({source_folder.name}/dataset.zip) | "
                                f"{rows} | {item['epochs']} | `{item['weights_sha256']}` |")
            commands.append(f'.venv/bin/python "$SCRIPTS/run_notebook_pilot.py" --notebook "$NOTEBOOK" '
                            f'--dataset overnight/{source_folder.name}/dataset.zip --run overnight/replay-{name} '
                            f'--revision {item["revision"]} --epochs {item["epochs"]} --patience {item["patience"]} '
                            f'--experiment {item["experiment"]}')
        checkout = f"replay-code-{best}"
        others = " ".join("overnight/replay-" + item["run"] for item in sources[1:])
        refresh = " --refresh-bn" if decision.get("bn_refresh") else ""
        if refresh:
            applied = decision["bn_refresh"]
            refresh += f" --bn-momentum {applied['momentum']}"
            reproduction.extend(["", f"Hiệu chỉnh BatchNorm trên {len(applied['images'])} ảnh train, "
                                 f"gồm {applied['images_per_domain']} ảnh mỗi nguồn Việt Nam, BDD và COCO."])
        reproduction.extend(["", "Từ thư mục làm việc của VM, dùng các lệnh sau để chạy lại từng fit rồi lấy trung bình.",
            "", "```bash", f'git clone --no-checkout https://github.com/tiennguyeniotd22-ops/allnaction-traffic-seg.git "{checkout}"',
            f'git -C "{checkout}" checkout --detach {tools_revision}',
            f'SCRIPTS="$PWD/{checkout}/scripts"', f'NOTEBOOK="$PWD/{checkout}/notebooks/train_and_export.ipynb"',
            *commands, f'CUDA_VISIBLE_DEVICES="" .venv/bin/python "$SCRIPTS/overnight_last_checkpoint.py" '
            f'overnight/replay-{sources[0]["run"]} overnight/replay-{best} "$NOTEBOOK" '
            f'audit_phase12/validation-manifest.json --mean-with {others}{refresh}', "```"])
        lines[start:end+1] = reproduction
    elif "reuses_fit_from" in resource:
        extraction = (
            f'.venv/bin/python "$SCRIPTS/overnight_last_checkpoint.py" overnight/replay-{best} '
            f'overnight/replay-{best}-selected "$NOTEBOOK" audit_phase12/validation-manifest.json'
        )
        end = lines.index("```", lines.index("## Tái lập"))
        lines[end:end] = [
            extraction + (f" --epoch {snap}" if snap else ""),
        ]
    for name, sha in {**checks, "model.onnx":entry["onnx_sha256"], "dataset.zip":resource["dataset_sha256"],
                      "notebook":resource["notebook_sha256"], "weights khởi tạo":resource["initial_weights_sha256"]}.items():
        lines.append(f"| {name} | `{sha}` |")
    lines += ["", f"[Đối chiếu PyTorch–ONNX và điểm từng nguồn]({best}/comparison.json) · "
              f"[Holdout lớp 3]({best}/held-score.json) · [Tài nguyên]({best}/resources.json) · "
              "[Nguồn dữ liệu và điều khoản](../../allnaction-traffic-seg/DATASET_PROVENANCE_REPORT.md)", ""]
    dest = root / "BAN_GIAO.md"
    assert b"PILOT_COMPLETE" in (run / "training.log").read_bytes()
    temp = dest.with_suffix(".partial")
    temp.write_text("\n".join(lines), encoding="utf-8")
    temp.replace(dest)
    print(best, overall["map50_95"], checks["submission.zip"], dest)


if __name__ == "__main__":
    main()

"""Summarize only Team 21 ZIPs that have independent CPU replay evidence on disk."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from notebook_identity import TEAM_ID, verified_team


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fmt(value):
    return f"{value:.2f}".replace(".", ",")


def image_hashes(archive, split):
    prefix = f"images/{split}/"
    return {hashlib.sha256(archive.read(k)).digest() for k in archive.namelist()
            if k.startswith(prefix) and k.lower().endswith((".jpg", ".png", ".jpeg"))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("repo", type=Path)
    a = p.parse_args()
    base = a.repo / "experiments/overnight"
    entries = read(base / "verified-scores.json")
    validation = sha(a.root.parent / "audit-phase12/validation-manifest.json")
    runs, evidence = {}, {}
    for key, entry in entries.items():
        folder = a.root / key
        if entry["resources"].get("team_id") != TEAM_ID:
            continue
        report = read(folder / "run_report.json")
        resource = entry["resources"]
        verified_team(report, resource["notebook_sha256"])
        replay = read(folder / "replay.json")
        probe = read(folder / ("probe.json" if (folder / "probe.json").exists() else "score.json"))
        assert sha(folder / "submission.zip") == replay["zip_sha256"] == entry["zip_sha256"] == probe["zip_sha256"]
        assert replay["dataset_sha256"] == resource["dataset_sha256"] and replay["validation_sha256"] == validation
        assert replay["prepared_sha256"] == report["dataset"]["sha256"] and replay["notebook_sha256"] == resource["notebook_sha256"]
        assert probe["n_images"] == 46 and entry["holdout"]["gt_instances"] == 7
        assert entry["scores"]["all140"]["map50_95"] == replay["score"]["map50_95"]
        assert replay["score"]["n_images"] == 140 and sum(replay["score"]["n_gt_per_class"].values()) == 1588
        assert entry["zip_bytes"] <= 42 * 2**20 and entry["opsets"][""] == 17
        with ZipFile(folder / "submission.zip") as archive:
            assert archive.namelist() == ["model.onnx"]
            assert hashlib.sha256(archive.read("model.onnx")).hexdigest() == entry["onnx_sha256"]
        runs[key] = (entry, report, probe)
        evidence[key] = dict(zip_sha256=replay["zip_sha256"], val_manifest_sha256=validation,
                             dataset_sha256=replay["dataset_sha256"], prepared_sha256=replay["prepared_sha256"],
                             notebook_sha256=replay["notebook_sha256"], predictions_sha256=replay["predictions_sha256"],
                             images=replay["score"]["n_images"], per_class_ap=replay["score"]["per_class_ap"],
                             map50_95=replay["score"]["map50_95"], probe46=probe["map50_95"],
                             source_weights=report.get("checkpoint_selection", {}).get("mean_sources", []))
    best = max(runs, key=lambda name: runs[name][0]["scores"]["all140"]["map50_95"])
    reference = runs["team21-qc20"][0]["scores"]
    entry, report, _ = runs[best]
    overall = entry["scores"]["all140"]
    ci = entry["paired_scene_bootstrap_95_points"]
    scores = {name: rec[0]["scores"]["all140"]["map50_95"] for name, rec in runs.items()}
    lines = ["# Kết quả đội 21 trên 140 ảnh validation", "",
        f"**Bản có mask mAP cao nhất là `{best}`, đạt {fmt(100*scores[best])}/100.** "
        f"So với pilot, chênh {fmt(100*(scores[best]-scores['team21-qc20']))} điểm. "
        "Mỗi dòng dưới đây đã chấm lại từ ZIP trên CPU. Điểm public và private do hệ thống cuộc thi xác định.",
        "", "| Bản | Mask mAP | AP50 | AP75 | Việt Nam · 15 | Mapillary · 46 | Other vehicle · holdout |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name in sorted(runs, key=scores.get, reverse=True):
        e, _, probe = runs[name]
        s = e["scores"]["all140"]
        values = [s["map50_95"], s["map50"], s["map75"], e["scores"]["Vietnam-v5"]["map50_95"],
                  probe["map50_95"], e["holdout"]["map50_95"]]
        lines.append("| `"+name+"` | "+" | ".join(fmt(100*v) for v in values)+" |")
    root = a.root / best
    compare = read(root / "comparison.json")
    paired_reference = compare["scores"]["round3"]["all140"]["map50_95"]
    labels = [name for name, val in scores.items() if val == paired_reference]
    for trial, control in (("team21-btc20", "team21-qc20"), ("team21-road20", "team21-btc20")):
        change = read(a.root / trial / "comparison.json")
        assert change["training_config_differences"] == {}
        assert change["scores"]["round3"]["all140"]["map50_95"] == scores[control]
        for field in ("initial_weights_sha256", "notebook_sha256"):
            assert runs[trial][0]["resources"][field] == runs[control][0]["resources"][field]
        interval = change["paired_scene_bootstrap_95_points"]
        lines.extend(["", f"`{trial}` so với `{control}` chênh {fmt(100*(scores[trial]-scores[control]))} điểm. "
                      f"Khoảng bootstrap ghép cặp 95% từ {fmt(interval[0])} đến {fmt(interval[1])}. "
                      "Cấu hình train, trọng số khởi tạo và notebook trùng nhau."])
    lines += ["", f"Khoảng bootstrap 95% theo nhóm cảnh của bản được chọn so với `{', '.join(labels)}` "
              f"là [{fmt(ci[0])}, {fmt(ci[1])}] điểm (300 lần lấy mẫu). "
              "Nhiều bản được so trên cùng validation. Khoảng này chưa điều chỉnh theo số lần chọn mô hình.",
              "", "| Lớp | GT hợp lệ | AP pilot | AP bản chọn |", "|---|---:|---:|---:|"]
    prior_name = "team21-mean-sgd-bn"
    for name in runs:
        paired = a.root / name / "vs-incumbent.json"
        if not paired.is_file():
            continue
        direct = read(paired)
        prior = runs[prior_name][0]
        assert direct["scores"]["round3"] == prior["scores"]
        assert direct["scores"]["pilot"] == runs[name][0]["scores"]
        assert direct["zip_sha256"] == evidence[name]["zip_sha256"] and direct["validation_sha_verified"] == 140
        low, high = direct["paired_scene_bootstrap_95_points"]
        diff = 100*(scores[name]-scores[prior_name])
        evidence[name]["paired_vs_prior_leader"] = dict(reference=prior_name, reference_zip_sha256=prior["zip_sha256"],
            difference_points=diff, interval_95_points=[low, high], replicates=direct["bootstrap_replicates"])
        if name == best:
            lines[-3:-3] = ["", f"So trực tiếp với ZIP đầu vòng `{prior_name}` ({fmt(100*scores[prior_name])}), "
                             f"chênh lệch là {fmt(diff)} điểm. Khoảng bootstrap 95% [{fmt(low)}, {fmt(high)}]."]
    for name in report["classes"]:
        count = overall["n_gt_per_class"][name]
        old = fmt(100*reference["all140"]["per_class_ap"][name]) if count else "Thiếu GT"
        new = fmt(100*overall["per_class_ap"][name]) if count else "Thiếu GT"
        lines.append(f"| {name} | {count} | {old} | {new} |")
    lines += ["", "## Dữ liệu và trọng số", "",
              "Bộ val gồm 45 ảnh BDD, 15 ảnh Việt Nam và 80 ảnh Drive. Holdout riêng của Other vehicle có 7 instance trên 6 ảnh. "
              "Mỗi lượt dùng seed 21, YOLO26n-seg, cạnh 640 và bước đóng gói từ notebook.",
              "", "Đối chứng A sửa 22 tệp nhãn trên bộ 1.156 ảnh. A+B thêm 31 ảnh COCO với 59 instance. "
              "Bản head20 thử chuyển trọng số COCO sang bốn lớp đích bổ sung. "
              "Các bản có tên mean dùng trung bình trọng số FP32 của các checkpoint trước khi xuất một model ONNX FP32.",
              "", "| Checkpoint đầu vào của bản chọn | Bộ dữ liệu | Epoch | Công thức |",
              "|---|---|---:|---|"]
    sources = report.get("checkpoint_selection", {}).get("mean_sources") or [
        dict(run=best, dataset_sha256=entry["resources"]["dataset_sha256"], epochs=report["params"]["EPOCHS"],
             experiment=entry["resources"]["experiment"])]
    versions = {}
    for m in a.root.glob("*/manifest.json"):
        if (m.parent / "dataset.zip").is_file() and isinstance((item := read(m)), dict):
            versions[item.get("dataset_sha256")] = m.parent
    for source in sources:
        folder = versions[source["dataset_sha256"]]
        assert sha(folder/"dataset.zip") == source["dataset_sha256"]
        lines.append(f"| {source['run']} | {folder.name} | {source['epochs']} | {source['experiment']} |")
    original = a.root.parent / "round3/team780.zip"
    assert sha(original) == read(a.root.parent / "round4/resources.json")["dataset_sha256"]
    with ZipFile(original) as z:
        old_train, old_val = image_hashes(z, "train"), image_hashes(z, "val")
    assert (len(old_train), len(old_val)) == (640, 140)
    for version in {s["dataset_sha256"] for s in sources}:
        with ZipFile(versions[version] / "dataset.zip") as z:
            train = image_hashes(z, "train")
            assert old_train <= train and image_hashes(z, "val") == old_val and old_val.isdisjoint(train)
    lines.extend(["", "Kiểm SHA ảnh trên từng bộ dữ liệu đầu vào của bản chọn xác nhận đủ 780 ảnh gốc vòng 4. "
                  "Toàn bộ 640 ảnh train vẫn ở train và 140 ảnh val vẫn ở val."])
    if bn := report.get("checkpoint_selection", {}).get("bn_refresh"):
        lines.extend(["", f"Bản chọn có hiệu chỉnh BatchNorm từ {len(bn['images'])} ảnh train thuộc ba nguồn, "
                      f"batch {bn['batch']} và momentum {bn['momentum']}."])
        if bn.get("preprocessing") == "notebook":
            lines.append("Các ảnh hiệu chỉnh được đọc và resize bằng chính hàm tiền xử lý trong notebook BTC.")
    if "team21-vnwide20" in runs:
        wide = runs["team21-vnwide20"][0]
        assert runs["team21-vnwide20"][1]["dataset"]["counts"]["train"]["images"] == 1411
        lines.extend(["", f"Lượt `team21-vnwide20` dùng 1.411 ảnh train, gồm 212 crop Mapillary và 43 crop BDD. "
                      f"Có 24 tệp nhãn Việt Nam đã sửa. So với pilot, điểm val140 chênh "
                      f"{fmt(100*(scores['team21-vnwide20']-scores['team21-qc20']))} và điểm Việt Nam chênh "
                      f"{fmt(100*(wide['scores']['Vietnam-v5']['map50_95']-reference['Vietnam-v5']['map50_95']))}. "
                      "Hai tập train khác nhau cả về nguồn ảnh và nhãn."])
    if "team21-o2m20" in runs:
        r = runs["team21-o2m20"][0]
        assert r["resources"]["dataset_sha256"] == runs["team21-qc20"][0]["resources"]["dataset_sha256"]
        assert r["resources"]["experiment"] == "o2m-constant"
        interval = r["paired_scene_bootstrap_95_points"]
        lines.extend(["", f"Trên cùng bộ train của pilot, `team21-o2m20` giữ trọng số loss của nhánh one-to-many "
                      f"ở 0,8 trong 20 epoch. Pilot giảm dần từ 0,8 xuống 0,1. Chênh lệch trên val140 là "
                      f"{fmt(100*(scores['team21-o2m20']-scores['team21-qc20']))} điểm, khoảng bootstrap 95% "
                      f"[{fmt(interval[0])}, {fmt(interval[1])}]. Hai lượt cùng dùng bước xuất ONNX của BTC."])
    if "team21-wideqc20" in runs:
        joined = runs["team21-wideqc20"][0]
        assert runs["team21-wideqc20"][1]["dataset"]["counts"]["train"]["images"] == 1442
        ci_joined = joined["paired_scene_bootstrap_95_points"]
        lines.extend(["", "Bộ 1.442 ảnh giữ các nhãn đã QC trên 48 tệp chung có khác biệt giữa hai bộ, "
                      "cùng 31 ảnh COCO đã duyệt và 255 crop BDD/Mapillary. "
                      f"So cùng công thức train với bộ 1.411 ảnh, chênh lệch là "
                      f"{fmt(100*(scores['team21-wideqc20']-scores['team21-vnwide20']))} điểm val140, "
                      f"khoảng bootstrap 95% [{fmt(ci_joined[0])}, {fmt(ci_joined[1])}]."])
    lines += ["", "Trên máy đã có quyền đọc thư mục dự án trên VM, lấy ZIP bằng Google Cloud CLI. "
              "Thực hiện lệnh dữ liệu cho từng phiên bản cần chạy notebook.",
              "", "```powershell",
              f'gcloud compute scp "cvat-ai-label-server:/home/pc/baseline-cvat300/overnight/{best}/submission/submission.zip" . --project=plantcare-ai-510902 --zone=us-central1-a',
              *[f'gcloud compute scp "cvat-ai-label-server:/home/pc/baseline-cvat300/overnight/{folder}/dataset.zip" "./{folder}.zip" --project=plantcare-ai-510902 --zone=us-central1-a'
                for folder in dict.fromkeys(versions[s["dataset_sha256"]].name for s in sources)],
              "```", "", f"Tệp nộp có {entry['zip_bytes']:,} byte, chỉ chứa `model.onnx`. SHA-256 là `{entry['zip_sha256']}`.",
              "", "[Kết quả chấm lại từ ZIP trên CPU](overnight/team21-cpu-replays.json) · "
              "[Bảng điểm và đối chiếu PyTorch–ONNX](overnight/verified-scores.json) · "
              "[Nguồn dữ liệu và quyền sử dụng](../DATASET_PROVENANCE_REPORT.md) · [Notebook](../notebooks/train_and_export.ipynb)", ""]
    (a.repo/"experiments/TEAM21_KET_QUA.md").write_text("\n".join(lines), encoding="utf8")
    (base/"team21-cpu-replays.json").write_bytes((json.dumps(evidence, indent=2) + "\n").encode())
    print(best, scores[best], len(runs), "verified ZIPs")


if __name__ == "__main__":
    main()

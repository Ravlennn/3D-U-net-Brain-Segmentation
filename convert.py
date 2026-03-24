import os
import shutil
import json
from pathlib import Path
from tqdm.auto import tqdm
from dotenv import load_dotenv

RAW_DATASET = Path(r"D:\VScode\3D_U-net\data\dataset")  # MSLesSeg
TASK_ID = 1

load_dotenv()

nnunet_raw = Path(os.environ["nnUNet_raw"])
task_dir = nnunet_raw / f"Dataset{TASK_ID}_MSLesSeg"
imagesTr = task_dir / "imagesTr"
labelsTr = task_dir / "labelsTr"
imagesTr.mkdir(parents=True, exist_ok=True)
labelsTr.mkdir(parents=True, exist_ok=True)

MODALITY_MAP = {
    "FLAIR": "0000",
    "T1": "0001",
    "T2": "0002",
}


def get_patient_number(p_id: str) -> int:
    num = "".join(filter(str.isdigit, p_id))
    return int(num) if num else 0


def parse_dataset(base_path: Path) -> dict:
    dataset_dict = {}

    for patient_dir in sorted(base_path.iterdir()):
        if not patient_dir.is_dir():
            continue

        p_id  = patient_dir.name
        p_num = get_patient_number(p_id)
        dataset_dict[p_id] = {}

        if p_num <= 53:
            for timeline_dir in sorted(patient_dir.iterdir()):
                if not timeline_dir.is_dir():
                    continue
                timeline_id = timeline_dir.name
                dataset_dict[p_id][timeline_id] = {}
                for f in timeline_dir.glob("*.nii.gz"):
                    modality = f.name.replace(".nii.gz", "").split("_")[-1]
                    dataset_dict[p_id][timeline_id][modality] = f
        else:
            for f in patient_dir.glob("*.nii.gz"):
                modality = f.name.replace(".nii.gz", "").split("_")[-1]
                dataset_dict[p_id][modality] = f

    return dataset_dict


def flatten_dataset(data_dict: dict) -> list:
    flat = []
    for p_id, p_data in data_dict.items():
        has_timelines = any(isinstance(v, dict) for v in p_data.values())
        if has_timelines:
            for timeline_id, modalities in p_data.items():
                if isinstance(modalities, dict):
                    sample = {"patient_id": p_id, "timeline": timeline_id}
                    sample.update(modalities)
                    flat.append(sample)
        else:
            sample = {"patient_id": p_id, "timeline": "baseline"}
            sample.update(p_data)
            flat.append(sample)
    return flat


def convert(samples: list) -> tuple:
    n_train   = 0
    n_test    = 0
    n_skipped = 0


    imagesTs = task_dir / "imagesTs"
    labelsTs = task_dir / "labelsTs"
    imagesTs.mkdir(parents=True, exist_ok=True)
    labelsTs.mkdir(parents=True, exist_ok=True)

    for sample in tqdm(samples, desc="convertation"):
        p_id     = sample["patient_id"]
        timeline = sample["timeline"]
        case_id  = f"{p_id}_{timeline}"
        p_num    = get_patient_number(p_id)

        missing = [m for m in MODALITY_MAP if m not in sample]
        if missing:
            print(f"  [SKIP] {case_id}: no modalities {missing}")
            n_skipped += 1
            continue

        if p_num >= 54:
            for mod, suffix in MODALITY_MAP.items():
                shutil.copy2(sample[mod], imagesTs / f"{case_id}_{suffix}.nii.gz")

            if "MASK" in sample:
                shutil.copy2(sample["MASK"], labelsTs / f"{case_id}.nii.gz")

            n_test += 1

        else:
            if "MASK" not in sample:
                print(f"  [SKIP] {case_id}: no mask")
                n_skipped += 1
                continue

            for mod, suffix in MODALITY_MAP.items():
                shutil.copy2(sample[mod], imagesTr / f"{case_id}_{suffix}.nii.gz")

            shutil.copy2(sample["MASK"], labelsTr / f"{case_id}.nii.gz")
            n_train += 1

    return n_train, n_test, n_skipped



def write_json(n_cases: int):
    dataset_json = {
        "channel_names": {
            "0": "FLAIR",
            "1": "T1",
            "2": "T2"
        },
        "labels": {
            "background": 0,
            "lesion": 1
        },
        "numTraining": n_cases,
        "file_ending": ".nii.gz"
    }
    dst = task_dir / "dataset.json"
    with open(dst, "w") as f:
        json.dump(dataset_json, f, indent=4)
    print(f"\ndataset.json → {dst}")
    print(json.dumps(dataset_json, indent=4))



if not RAW_DATASET.exists():
        raise FileNotFoundError(f"NO DATASET{RAW_DATASET}")

dataset_dict = parse_dataset(RAW_DATASET)
samples      = flatten_dataset(dataset_dict)
print(f"FIND: {len(samples)}\n")

n_train, n_test, n_skipped = convert(samples)

print(f"\n{'─'*40}")
print(f"Train (imagesTr) : {n_train}")
print(f"Test  (imagesTs) : {n_test}")
print(f"SKIPPED: {n_skipped}")

write_json(n_train) 

print(f"\n{'─'*40}")


# nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -c 3d_fullres"
# nnUNetv2_train 1 3d_fullres 0 --npz"

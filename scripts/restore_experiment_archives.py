#!/usr/bin/env python3
"""Restore raw archived experiments into ignored `external/` using only local ZIPs."""
import argparse
from pathlib import Path
from zipfile import ZipFile

NAMES = [
    'SWP_LM_v11_controlled_experiment.zip',
    'SWP_LM_v12_scaling_results.zip',
    'SWP_LM_v13_fresh_data_scaling.zip',
    'SWP_LM_v14_dictionary_regularization_experiment.zip',
    'SWP_LM_v15_frequency_gated_experiment.zip',
    'SWP_LM_v16_naturalness_evaluation.zip',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path, required=True, help='Directory with original experiment ZIPs')
    args = ap.parse_args()
    target = Path(__file__).resolve().parents[1]/'external'
    target.mkdir(exist_ok=True)
    for name in NAMES:
        archive = args.input_dir/name
        if not archive.exists():
            print(f'MISSING: {archive}')
            continue
        dest = target/archive.stem
        dest.mkdir(exist_ok=True)
        count=0
        with ZipFile(archive) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f'{archive}: corrupted member {bad}')
            for info in zf.infolist():
                if info.is_dir():
                    continue
                candidate=(dest/info.filename).resolve()
                if not candidate.is_relative_to(dest.resolve()):
                    raise RuntimeError(f'Unsafe archive member: {info.filename}')
                candidate.parent.mkdir(parents=True, exist_ok=True)
                candidate.write_bytes(zf.read(info))
                count += 1
        print(f'{name} -> {dest} ({count} files)')
    print('Done. `external/` is excluded from Git.')


if __name__ == '__main__':
    main()

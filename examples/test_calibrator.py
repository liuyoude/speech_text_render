# -*- coding: utf-8 -*-
"""
Test script for PopulationCalibrator.
Runs calibration on examples/audios/ and prints statistics.
"""
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from core.calibrator import PopulationCalibrator, ALL_FEATURES


def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "dataset_config.yaml")
    config_path = os.path.abspath(config_path)

    device = "cuda" if "--cpu" not in sys.argv else "cpu"
    logger.info("Using device: %s", device)

    calibrator = PopulationCalibrator(config_path, device=device)
    stats = calibrator.calibrate()

    print("\n" + "=" * 80)
    print("CALIBRATION RESULTS")
    print("=" * 80)
    for group_key, features in sorted(stats.items()):
        print(f"\n--- {group_key} ---")
        print(f"  {'feature':<16} {'n':>6}  {'mean':>12}  {'std':>12}")
        print(f"  {'-'*16} {'-'*6}  {'-'*12}  {'-'*12}")
        for feat in ALL_FEATURES:
            if feat in features:
                d = features[feat]
                print(f"  {feat:<16} {d['n']:>6}  {d['mean']:>12.4f}  {d['std']:>12.4f}")

    output_dir = os.path.join(os.path.dirname(__file__), "results", "calibration")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "population_stats.json")
    calibrator.save(output_path)

    default_path = os.path.join(os.path.dirname(__file__), "..", "core", "default_population_stats.json")
    if os.path.isfile(default_path):
        with open(default_path, "r", encoding="utf-8") as f:
            defaults = json.load(f)
        print("\n" + "=" * 80)
        print("COMPARISON WITH DEFAULT (placeholder) VALUES")
        print("=" * 80)
        for group_key in sorted(set(stats) & set(defaults)):
            print(f"\n--- {group_key} ---")
            print(f"  {'feature':<16} {'cal_mean':>10} {'def_mean':>10} {'cal_std':>10} {'def_std':>10}")
            print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
            for feat in ALL_FEATURES:
                if feat in stats[group_key] and feat in defaults[group_key]:
                    c = stats[group_key][feat]
                    d = defaults[group_key][feat]
                    print(f"  {feat:<16} {c['mean']:>10.4f} {d['mean']:>10.4f} {c['std']:>10.4f} {d['std']:>10.4f}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

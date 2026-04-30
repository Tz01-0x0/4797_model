"""
Batch cleaning: run data_cleaning_pipeline for every valid participant.

Reads data/participant_metadata.csv, finds which raw Gorilla task file each
participant appears in, filters to that participant (keeping only the first
180 trials when duplicates exist), and writes per-participant CSVs into
data/cleaned/.

Run from <repo_root>:  python data_cleaning/run_all_cleaning.py
Requires that comprehensive_cleaning_pipeline.py has been run first to
produce data/participant_metadata.csv.
"""

import os
import sys
import glob
import pandas as pd

# Allow `import data_cleaning_pipeline` when called from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_cleaning_pipeline as dcp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
DATA_ROOT  = os.path.join(REPO_ROOT, 'data')
METADATA   = os.path.join(DATA_ROOT, 'participant_metadata.csv')
RAW_DIR    = os.path.join(DATA_ROOT, 'raw_gorilla')
OUTPUT_DIR = os.path.join(DATA_ROOT, 'cleaned')

def main():
    # Load valid participant list
    meta = pd.read_csv(METADATA)
    valid_pids = meta['participant_id'].tolist()
    print(f"Participants to process: {len(valid_pids)}")

    # Find all task CSV files
    task_files = sorted(glob.glob(os.path.join(RAW_DIR, '*_task-*.csv')))
    print(f"Raw task files found: {len(task_files)}")

    # Build a map: participant_id -> task_file
    pid_to_file = {}
    for tf in task_files:
        try:
            df = pd.read_csv(tf, encoding='utf-8-sig', usecols=['Participant Public ID', 'Display', 'Screen'])
            # Only consider files with actual response data
            resp = df[(df['Display'] == 'Stimuli_Response') & (df['Screen'] == 'Screen 1')]
            pids_in_file = resp['Participant Public ID'].unique()
            for pid in pids_in_file:
                if pid in valid_pids and pid not in pid_to_file:
                    pid_to_file[pid] = tf
        except Exception as e:
            print(f"  Warning: could not read {tf}: {e}")

    found = [p for p in valid_pids if p in pid_to_file]
    missing = [p for p in valid_pids if p not in pid_to_file]
    print(f"Found raw data for: {len(found)} participants")
    if missing:
        print(f"MISSING raw data for: {missing}")

    # Clean output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each participant
    success = 0
    errors = []
    for i, pid in enumerate(found):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(found)}] Processing {pid}")
        print(f"  Source: {os.path.basename(pid_to_file[pid])}")
        print(f"{'='*60}")

        try:
            # Load and filter the raw file
            trials = dcp.load_and_filter(pid_to_file[pid])

            # Filter to THIS participant only
            trials = trials[trials['Participant Public ID'] == pid]
            if len(trials) == 0:
                print(f"  ERROR: No trials found for {pid}")
                errors.append((pid, "No trials"))
                continue

            print(f"  Trials for this participant: {len(trials)}")

            # Deduplicate: if participant did the task multiple times,
            # keep only the FIRST complete run (180 trials).
            # Multiple runs have trial_number resetting to 1.
            if len(trials) > 180:
                orig_len = len(trials)
                # Ensure Event Index is numeric for proper sorting
                trials['Event Index'] = pd.to_numeric(trials['Event Index'], errors='coerce')
                trials = trials.sort_values('Event Index').reset_index(drop=True)
                # First run: rows 0 to 179 (first 180 trials in temporal order)
                trials = trials.iloc[:180].copy()
                print(f"  DEDUP: Multiple runs detected, keeping first 180 trials (was {orig_len})")

            # Run pipeline steps
            clean = dcp.extract_columns(trials)
            clean = dcp.encode_behavior(clean)

            # Set global for clean_rt
            dcp.n_timeout = clean['is_timeout'].sum()
            clean = dcp.clean_rt(clean)
            clean = dcp.quality_checks(clean)
            clean = dcp.build_hgf_inputs(clean)

            # Verify calibration order is correct
            cal = clean[clean['phase'] == 'calibration']
            cal_orders = cal['trial_order'].tolist()
            if cal_orders != list(range(1, len(cal) + 1)):
                print(f"  WARNING: Calibration trials not at start! Orders: {cal_orders[:5]}...{cal_orders[-3:]}")
            else:
                print(f"  OK: All {len(cal)} calibration trials at positions 1-{len(cal)}")

            # Export
            clean = dcp.export_cleaned(clean, OUTPUT_DIR, pid)
            success += 1

        except Exception as e:
            print(f"  ERROR processing {pid}: {e}")
            import traceback
            traceback.print_exc()
            errors.append((pid, str(e)))

    # Summary
    print(f"\n{'#'*60}")
    print(f"BATCH COMPLETE: {success}/{len(found)} succeeded")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for pid, err in errors:
            print(f"  {pid}: {err}")
    print(f"{'#'*60}")


if __name__ == '__main__':
    main()

"""
Single-file Gorilla -> HGF cleaning pipeline.

Reads one Gorilla task CSV, applies basic filtering (timeouts -> NaN), and
writes per-participant *_cleaned.csv (trial-level) + *_hgf_input.csv
(HGF-ready) into ./cleaned/ relative to the current working directory.
Also exposed as a module (run_all_cleaning.py imports it).

Usage:
    python data_cleaning/data_cleaning_pipeline.py <gorilla_csv_file>
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path


# Configuration

# RT thresholds (ms)
RT_MIN = 200       # below this counts as an anticipatory response (too fast to be a deliberate decision)
RT_MAX = 3000      # above this counts as a timeout
RT_TIMEOUT = 2999  # Gorilla's own timeout cutoff (~3001 ms)

# Whether to drop practice trials
REMOVE_PRACTICE = True

# Timeout handling: 'nan' (mark as NaN) or 'remove' (drop the row)
TIMEOUT_STRATEGY = 'nan'  # recommended: 'nan' — HGF fitModel skips NaN responses automatically

# Fast-RT handling: 'nan', 'remove', or 'keep'
FAST_RT_STRATEGY = 'nan'  # recommended: 'nan' — same reason as above


# Step 1: load and basic filtering

def load_and_filter(filepath):
    """Extract the actual trial-level rows from a Gorilla CSV."""

    df = pd.read_csv(filepath)
    print(f"[Step 1] Raw data: {len(df)} rows, {len(df.columns)} columns")

    # Keep only Stimuli_Response Screen 1 rows
    # Gorilla trial structure: Fixation -> Stimuli_pre_response -> Stimuli_Response (Screen 1) -> Feedback -> ITI
    # Screen 1 is where the actual key press is logged; Correct/Wrong/TimeOut are feedback rows (RT=0, not needed).
    trials = df[
        (df['Display'] == 'Stimuli_Response') &
        (df['Screen'] == 'Screen 1')
    ].copy()

    print(f"[Step 1] Trial rows: {len(trials)}")

    # Drop practice trials
    if REMOVE_PRACTICE:
        n_before = len(trials)
        trials = trials[trials['Spreadsheet: Phase'] != 'practice'].copy()
        print(f"[Step 1] Dropped practice trials: {n_before} -> {len(trials)}")

    return trials


# Step 2: extract and rename key columns

def extract_columns(trials):
    """Pick the columns needed for HGF modelling and rename them concisely."""

    cols = {
        # Metadata
        'Event Index': 'event_index',
        'Participant Public ID': 'participant_id',
        'Spreadsheet: Condition': 'condition',        # verbal_first / facial_first
        'Spreadsheet: Seed': 'seed',

        # Trial structure
        'Spreadsheet: Trial_Number': 'trial_number',
        'Spreadsheet: Phase': 'phase',
        'Spreadsheet: Phase_Number': 'phase_number',
        'Spreadsheet: Block': 'block',
        'Spreadsheet: Equivalence_Group': 'equivalence_group',  # A, B, C
        'Spreadsheet: Face_Pair_Type': 'face_pair_type',        # happy_vs_sad, etc.

        # Cue info
        'Spreadsheet: Facial_Cue_Accuracy': 'facial_cue_accuracy',  # 0.1-0.9 (Markov state)
        'Spreadsheet: Verbal_Cue_Accuracy': 'verbal_cue_accuracy',  # 0.2 or 0.8
        'Spreadsheet: Facial_Cue_Valid': 'facial_cue_valid',    # 1/0: was the facial cue correct on this trial
        'Spreadsheet: Verbal_Cue_Valid': 'verbal_cue_valid',    # 1/0: was the verbal cue correct on this trial
        'Spreadsheet: Facial_Cue_Direction': 'facial_cue_direction',  # left/right
        'Spreadsheet: Verbal_Cue_Direction': 'verbal_cue_direction',  # left/right
        'Spreadsheet: Cue_Conflict': 'cue_conflict',            # 1/0

        # Block state
        'Spreadsheet: Facial_Block': 'facial_block',
        'Spreadsheet: Verbal_Block': 'verbal_block',
        'Spreadsheet: Facial_Block_State': 'facial_block_state',  # high/low
        'Spreadsheet: Verbal_Block_State': 'verbal_block_state',  # high/low

        # Stimuli
        'Spreadsheet: Face_Left': 'face_left',
        'Spreadsheet: Face_Right': 'face_right',
        'Spreadsheet: Face_Left_Expr': 'face_left_expr',
        'Spreadsheet: Face_Right_Expr': 'face_right_expr',
        'Spreadsheet: Card_Left_Points': 'card_left_points',
        'Spreadsheet: Card_Right_Points': 'card_right_points',
        'Spreadsheet: Point_Difference': 'point_difference',
        'Spreadsheet: Correct_Side': 'correct_side',
        'Spreadsheet: Winning_Card': 'winning_card',
        'Spreadsheet: Reward_Points': 'reward_points',
        'Spreadsheet: Verbal_Text': 'verbal_text',

        # Response
        'Spreadsheet: Key_Left': 'key_left',
        'Spreadsheet: Key_Right': 'key_right',
        'Spreadsheet: Correct_Key': 'correct_key',
        'Response': 'response_key',          # Z (left) or M (right), NaN = timeout
        'Correct': 'correct',                # 1/0 (Gorilla-supplied)
        'Reaction Time': 'rt_raw',           # raw RT (ms)
    }

    clean = trials[list(cols.keys())].rename(columns=cols).copy()
    clean = clean.reset_index(drop=True)

    # Type conversions
    clean['event_index'] = clean['event_index'].astype(int)  # CRITICAL: must be int for correct sort in step 6
    clean['rt_raw'] = pd.to_numeric(clean['rt_raw'], errors='coerce')
    clean['trial_number'] = clean['trial_number'].astype(int)
    clean['facial_cue_valid'] = clean['facial_cue_valid'].astype(float)
    clean['verbal_cue_valid'] = clean['verbal_cue_valid'].astype(float)
    clean['cue_conflict'] = clean['cue_conflict'].astype(float)
    clean['card_left_points'] = clean['card_left_points'].astype(float)
    clean['card_right_points'] = clean['card_right_points'].astype(float)

    print(f"[Step 2] Extracted {len(clean.columns)} columns, {len(clean)} rows")
    return clean


# Step 3: encode behavioural variables

def encode_behavior(clean):
    """Encode the participant's choice behaviour."""

    # 3a: flag timeouts
    clean['is_timeout'] = clean['response_key'].isna()
    n_timeout = clean['is_timeout'].sum()
    print(f"[Step 3] Timeout trials: {n_timeout} / {len(clean)} ({n_timeout/len(clean)*100:.1f}%)")

    # 3b: encode choice direction
    # chose_left: 1 = chose left (Z key), 0 = chose right (M key), NaN = timeout
    clean['chose_left'] = np.where(
        clean['response_key'] == 'Z', 1.0,
        np.where(clean['response_key'] == 'M', 0.0, np.nan)
    )

    # 3c: did the participant follow the facial cue?
    facial_points_left = (clean['facial_cue_direction'] == 'left').astype(float)
    clean['followed_facial'] = np.where(
        clean['is_timeout'], np.nan,
        (clean['chose_left'] == facial_points_left).astype(float)
    )

    # 3d: did the participant follow the verbal cue?
    verbal_points_left = (clean['verbal_cue_direction'] == 'left').astype(float)
    clean['followed_verbal'] = np.where(
        clean['is_timeout'], np.nan,
        (clean['chose_left'] == verbal_points_left).astype(float)
    )

    # 3e: did the participant choose the higher-reward card?
    # correct_side tells us which side has the higher reward.
    correct_is_left = (clean['correct_side'] == 'left').astype(float)
    clean['chose_high_reward'] = np.where(
        clean['is_timeout'], np.nan,
        (clean['chose_left'] == correct_is_left).astype(float)
    )

    # Behaviour summary
    resp = clean[~clean['is_timeout']]
    print(f"[Step 3] Behaviour summary (timeouts excluded):")
    print(f"  Followed facial cue: {resp['followed_facial'].mean():.3f}")
    print(f"  Followed verbal cue: {resp['followed_verbal'].mean():.3f}")
    print(f"  Chose high-reward card: {resp['chose_high_reward'].mean():.3f}")

    return clean


# Step 4: RT cleaning and outlier flagging

def clean_rt(clean):
    """Clean the RT column: flag anticipatory and timeout trials."""

    clean['rt_clean'] = clean['rt_raw'].copy()

    # 4a: flag anticipatory responses
    clean['is_fast'] = (~clean['is_timeout']) & (clean['rt_raw'] < RT_MIN)
    n_fast = clean['is_fast'].sum()
    print(f"[Step 4] Anticipatory responses (RT < {RT_MIN}ms): {n_fast} / {len(clean)} ({n_fast/len(clean)*100:.1f}%)")

    # 4b: apply the chosen strategy
    # Timeout handling
    if TIMEOUT_STRATEGY == 'nan':
        # Mark timeout responses as NaN (HGF fitModel skips NaN automatically).
        clean.loc[clean['is_timeout'], 'response_for_model'] = np.nan

    # Fast-RT handling
    if FAST_RT_STRATEGY == 'nan':
        clean.loc[clean['is_fast'], 'response_for_model'] = np.nan
        print(f"[Step 4] Marked NaN (model will skip): timeout={n_timeout}, fast={n_fast}")
    elif FAST_RT_STRATEGY == 'keep':
        print(f"[Step 4] Keeping fast-RT trials (only flagged)")

    # For unflagged trials, response_for_model = chose_left.
    if 'response_for_model' not in clean.columns:
        clean['response_for_model'] = np.nan

    valid_mask = ~clean['is_timeout'] & ~clean['is_fast']
    clean.loc[valid_mask, 'response_for_model'] = clean.loc[valid_mask, 'chose_left']

    # Mark every exclusion type
    clean['exclusion_reason'] = 'valid'
    clean.loc[clean['is_timeout'], 'exclusion_reason'] = 'timeout'
    clean.loc[clean['is_fast'], 'exclusion_reason'] = 'fast_rt'

    # RT summary
    valid_rt = clean.loc[valid_mask, 'rt_raw']
    print(f"[Step 4] Valid trials: {valid_mask.sum()} / {len(clean)}")
    print(f"[Step 4] Valid RT: mean={valid_rt.mean():.0f}ms, median={valid_rt.median():.0f}ms")

    return clean


# Step 5: quality checks

def quality_checks(clean):
    """Run data-quality checks and flag anything worth attention."""

    print("\n" + "="*60)
    print("Quality Checks")
    print("="*60)

    valid = clean[clean['exclusion_reason'] == 'valid']

    # 5a: overall exclusion rate
    excl_rate = 1 - len(valid) / len(clean)
    status = "OK" if excl_rate < 0.20 else "WARNING"
    print(f"[{status}] Total exclusion rate: {excl_rate*100:.1f}% (threshold: <20%)")

    # 5b: timeout rate
    timeout_rate = clean['is_timeout'].mean()
    status = "OK" if timeout_rate < 0.10 else "WARNING"
    print(f"[{status}] Timeout rate: {timeout_rate*100:.1f}% (threshold: <10%)")

    # 5c: calibration-phase accuracy
    cal = valid[valid['phase'] == 'calibration']
    if len(cal) > 0:
        cal_acc = cal['correct'].mean()
        status = "OK" if cal_acc > 0.55 else "WARNING"
        print(f"[{status}] Calibration accuracy: {cal_acc*100:.1f}% (threshold: >55%)")

    # 5d: side bias
    chose_left_rate = valid['chose_left'].mean()
    side_bias = abs(chose_left_rate - 0.5)
    status = "OK" if side_bias < 0.25 else "WARNING"
    print(f"[{status}] Side bias: {chose_left_rate*100:.1f}% chose left (deviation from 50%: {side_bias*100:.1f}%, threshold: <25%)")

    # 5e: trial counts per phase
    print(f"\n[INFO] Valid trials per phase:")
    for phase in ['calibration', 'verbal_volatile', 'overlap', 'facial_volatile']:
        p = valid[valid['phase'] == phase]
        print(f"  {phase}: {len(p)} trials")

    # 5f: trial counts per equivalence group
    print(f"\n[INFO] Valid trials per Equivalence Group:")
    for eg in ['A', 'B', 'C']:
        e = valid[valid['equivalence_group'] == eg]
        print(f"  Group {eg}: {len(e)} trials")

    # 5g: behavioural pattern check
    print(f"\n[INFO] Follow-rates per phase:")
    for phase in ['calibration', 'verbal_volatile', 'overlap', 'facial_volatile']:
        p = valid[valid['phase'] == phase]
        if len(p) > 0:
            print(f"  {phase}: facial={p['followed_facial'].mean():.3f}, verbal={p['followed_verbal'].mean():.3f}")

    return clean


# Step 6: sort by trial + build HGF input matrix

def build_hgf_inputs(clean):
    """
    Build the input matrix and response vector required by the HGF model.

    Different architectures take different output shapes:

    Architecture A (2-HGF, facial collapsed):
        u = [facial_cue_valid, verbal_cue_valid, card_left_points, card_right_points]
        y = response (chose_left)
        -> facial outcomes for the three EGs are merged into one row per trial_number.

    Architecture B/C (4-HGF, facial separated):
        u = [facial_A_valid, facial_B_valid, facial_C_valid, verbal_valid, group_indicator]
        y = response (chose_left)
        -> one row per trial including the group indicator.
    """

    # Sort by event_index to preserve temporal order
    clean = clean.sort_values('event_index').reset_index(drop=True)

    # Add a global trial number (in temporal order)
    clean['trial_order'] = range(1, len(clean) + 1)

    # Encode equivalence group as numeric
    eg_map = {'A': 1, 'B': 2, 'C': 3}
    clean['eg_numeric'] = clean['equivalence_group'].map(eg_map)

    # Encode phase as numeric
    phase_map = {'calibration': 1, 'verbal_volatile': 2, 'overlap': 3, 'facial_volatile': 4}
    clean['phase_numeric'] = clean['phase'].map(phase_map)

    # Encode cue direction as numeric (used by the response model)
    clean['facial_direction_left'] = (clean['facial_cue_direction'] == 'left').astype(int)
    clean['verbal_direction_left'] = (clean['verbal_cue_direction'] == 'left').astype(int)

    print(f"\n[Step 6] HGF input matrix built")
    print(f"  Total trials: {len(clean)}")
    print(f"  Valid trials (response_for_model not NaN): {clean['response_for_model'].notna().sum()}")

    return clean


# Step 7: export

def export_cleaned(clean, output_dir, participant_id):
    """Export the cleaned data."""

    os.makedirs(output_dir, exist_ok=True)

    # 7a: full cleaned CSV
    csv_path = os.path.join(output_dir, f'{participant_id}_cleaned.csv')
    clean.to_csv(csv_path, index=False)
    print(f"\n[Step 7] Saved CSV: {csv_path}")

    # 7b: HGF-ready format (Architecture B/C ready to use)
    hgf_cols = [
        'trial_order', 'trial_number', 'phase', 'phase_numeric',
        'equivalence_group', 'eg_numeric',
        'facial_cue_valid', 'verbal_cue_valid',
        'facial_cue_accuracy', 'verbal_cue_accuracy',
        'facial_cue_direction', 'verbal_cue_direction',
        'facial_direction_left', 'verbal_direction_left',
        'card_left_points', 'card_right_points', 'point_difference',
        'correct_side', 'correct_key',
        'response_key', 'chose_left', 'rt_raw',
        'followed_facial', 'followed_verbal', 'chose_high_reward',
        'is_timeout', 'is_fast', 'exclusion_reason',
        'response_for_model',
    ]

    hgf_df = clean[hgf_cols].copy()
    hgf_path = os.path.join(output_dir, f'{participant_id}_hgf_input.csv')
    hgf_df.to_csv(hgf_path, index=False)
    print(f"[Step 7] Saved HGF input: {hgf_path}")

    # 7c: MATLAB .mat format (optional)
    try:
        from scipy.io import savemat

        # Architecture B/C format: one trial per row.
        mat_data = {
            'y': clean['response_for_model'].values.reshape(-1, 1),  # response vector
            'u_facial_valid': clean['facial_cue_valid'].values.reshape(-1, 1),
            'u_verbal_valid': clean['verbal_cue_valid'].values.reshape(-1, 1),
            'u_card_left_pts': clean['card_left_points'].values.reshape(-1, 1),
            'u_card_right_pts': clean['card_right_points'].values.reshape(-1, 1),
            'u_group_id': clean['eg_numeric'].values.reshape(-1, 1),
            'phase': clean['phase_numeric'].values.reshape(-1, 1),
            'trial_number': clean['trial_number'].values.reshape(-1, 1),
            'rt': clean['rt_raw'].values.reshape(-1, 1),
            'condition': clean['condition'].iloc[0],
            'participant_id': participant_id,
        }

        mat_path = os.path.join(output_dir, f'{participant_id}_hgf_input.mat')
        savemat(mat_path, mat_data)
        print(f"[Step 7] Saved .mat: {mat_path}")

    except ImportError:
        print("[Step 7] scipy not installed, skipping .mat export (pip install scipy)")

    # 7d: print exclusion summary
    print(f"\n{'='*60}")
    print(f"Data Cleaning Summary for {participant_id}")
    print(f"{'='*60}")
    print(f"  Condition: {clean['condition'].iloc[0]}")
    print(f"  Total main trials: {len(clean)}")
    print(f"  Timeouts: {clean['is_timeout'].sum()} ({clean['is_timeout'].mean()*100:.1f}%)")
    print(f"  Fast RT (< {RT_MIN}ms): {clean['is_fast'].sum()} ({clean['is_fast'].mean()*100:.1f}%)")
    print(f"  Valid for modeling: {(clean['exclusion_reason'] == 'valid').sum()}")
    print(f"  Overall accuracy (valid): {clean.loc[clean['exclusion_reason']=='valid', 'correct'].mean()*100:.1f}%")

    return clean


# Main

def run_pipeline(filepath):
    """Run the full data-cleaning pipeline."""

    print(f"\n{'#'*60}")
    print(f"HGF Data Cleaning Pipeline")
    print(f"Input: {filepath}")
    print(f"{'#'*60}\n")

    # Step 1: load and filter
    trials = load_and_filter(filepath)

    # Step 2: extract columns
    clean = extract_columns(trials)

    # Step 3: encode behaviour
    clean = encode_behavior(clean)

    # Step 4: RT cleaning
    global n_timeout  # for reporting in clean_rt
    n_timeout = clean['is_timeout'].sum()
    clean = clean_rt(clean)

    # Step 5: quality checks
    clean = quality_checks(clean)

    # Step 6: build HGF inputs
    clean = build_hgf_inputs(clean)

    # Step 7: export
    participant_id = clean['participant_id'].iloc[0]
    # Default output: cleaned/ subdirectory next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'cleaned')
    clean = export_cleaned(clean, output_dir, participant_id)

    return clean


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python data_cleaning_pipeline.py <gorilla_csv_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    clean = run_pipeline(filepath)

"""
End-to-end cleaning for the Gorilla facial-verbal experiment.

For every participant across all task-file versions, this script:
  - extracts trial-level data and writes one *_cleaned.csv + one *_hgf_input.csv
  - scores AQ50 (total + 5 subscales), DASS21, Fluid Intelligence
  - assigns ASD/NT group from self-report diagnosis + AQ
  - writes participant_metadata.csv, quality_report.csv, cleaning_report.txt

Inputs : <repo_root>/data/raw_gorilla/  (all Gorilla CSVs from experiment 259923)
Outputs: <repo_root>/data/cleaned/, <repo_root>/data/participant_metadata.csv,
         <repo_root>/data/quality_report.csv, <repo_root>/data/cleaning_report.txt,
         <repo_root>/data/pid_lookup_PRIVATE.csv (gitignored)

Run from <repo_root>:  python data_cleaning/comprehensive_cleaning_pipeline.py
"""

import pandas as pd
import numpy as np
import os
import re
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# Paths — everything data-related lives under <repo_root>/data/.
#   data/raw_gorilla/            raw Gorilla CSVs (input, gitignored)
#   data/cleaned/                per-participant cleaned + HGF input (output, shipped)
#   data/participant_metadata.csv, data/quality_report.csv, data/cleaning_report.txt  (output, shipped)
#   data/pid_lookup_PRIVATE.csv  Prolific -> P### map (output, gitignored)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
DATA_ROOT  = os.path.join(REPO_ROOT, 'data')
DATA_DIR   = os.path.join(DATA_ROOT, 'raw_gorilla')
OUTPUT_DIR = os.path.join(DATA_ROOT, 'cleaned')

# Anonymisation: every Prolific Public ID is replaced with a deterministic
# label P001, P002, ... before anything is written to disk. The mapping is
# saved once to PID_LOOKUP_FILE (gitignored) so re-runs are reproducible.
ANONYMISE = True
ANON_PREFIX = 'P'
PID_LOOKUP_FILE = os.path.join(DATA_ROOT, 'pid_lookup_PRIVATE.csv')

# RT: NO exclusion based on RT. Only timeouts (no response) are marked as NaN.
# Fast RTs are retained because Gorilla measures RT from response screen onset,
# and participants have already processed cues during Stimuli_pre_response.
# Even ultra-fast (<100ms) responses show above-chance cue sensitivity.
TIMEOUT_STRATEGY = 'nan'  # timeout trials -> response_for_model = NaN

# Quality flag thresholds (informational only, no exclusion)
MAX_TIMEOUT_RATE = 0.15
MIN_ACCURACY = 0.55
MAX_SIDE_BIAS = 0.30

# AQ cutoff for "broad autism phenotype"
AQ_CLINICAL_CUTOFF = 32

# Expected trials per participant
EXPECTED_TRIALS = 180

# AQ50 scoring (Baron-Cohen et al., 2001)

# Items scored 1 for "agree" (Definitely Agree or Slightly Agree)
AQ_AGREE_ITEMS = {2,4,5,6,7,9,12,13,16,18,19,20,21,22,23,26,33,35,39,41,42,43,45,46}
# Items scored 1 for "disagree" (Slightly Disagree or Definitely Disagree)
AQ_DISAGREE_ITEMS = {1,3,8,10,11,14,15,17,24,25,27,28,29,30,31,32,34,36,37,38,40,44,47,48,49,50}

# AQ50 Subscales (5 areas, 10 items each)
AQ_SUBSCALES = {
    'aq_social_skill':        {1, 11, 13, 15, 22, 36, 44, 45, 47, 48},
    'aq_attention_switching':  {2, 4, 10, 16, 25, 32, 34, 37, 43, 46},
    'aq_attention_to_detail':  {5, 6, 9, 12, 19, 23, 28, 29, 30, 49},
    'aq_communication':        {7, 17, 18, 26, 27, 31, 33, 35, 38, 39},
    'aq_imagination':          {3, 8, 14, 20, 21, 24, 40, 41, 42, 50},
}


def score_aq_item(q_num, quantised_val):
    """Score a single AQ item. Gorilla quantised: 1=DefAgree, 2=SlAgree, 3=SlDisagree, 4=DefDisagree"""
    if q_num in AQ_AGREE_ITEMS:
        return 1 if quantised_val <= 2 else 0
    elif q_num in AQ_DISAGREE_ITEMS:
        return 1 if quantised_val >= 3 else 0
    return 0


# Step 1: parse questionnaires

def parse_demographics():
    """Parse demographics questionnaire (Gorilla WIDE format)."""
    filepath = os.path.join(DATA_DIR, 'data_exp_259923-vall_questionnaire-lsfe.csv')
    df = pd.read_csv(filepath)

    # Wide format: row 0 = question text, rows 1+ = data
    data = df.iloc[1:].copy()
    data = data.dropna(subset=['Participant Public ID'])

    records = {}
    for _, row in data.iterrows():
        pid = str(row['Participant Public ID'])
        rec = {
            'age': str(row.get('Number Entry object-2 Value', '')).strip(),
            'gender': str(row.get('Multiple Choice object-3 Response', '')).strip(),
            'asc_diagnosis': str(row.get('Multiple Choice object-4 Response', '')).strip(),
            'randomiser_ne38': str(row.get('randomiser-ne38', '')).strip(),
            'randomiser_mwio': str(row.get('randomiser-mwio', '')).strip(),
            'randomiser_1rdd': str(row.get('randomiser-1rdd', '')).strip(),
        }
        for k in rec:
            if rec[k] == 'nan':
                rec[k] = ''

        # Fix known typo: one participant entered age as 22222222222
        # (keyboard stutter), corrected to 22
        if rec['age'] == '22222222222':
            rec['age'] = '22'

        records[pid] = rec

    print(f"[Demographics] {len(records)} participants")
    return records


def parse_aq50():
    """Parse AQ50 questionnaire (Gorilla WIDE format) with total + 5 subscales."""
    filepath = os.path.join(DATA_DIR, 'data_exp_259923-vall_questionnaire-ksqo.csv')
    df = pd.read_csv(filepath)

    quantised_cols = [c for c in df.columns if 'Quantised' in c]

    # Build question number -> column mapping from row 0
    q_num_to_col = {}
    for c in quantised_cols:
        q_text = str(df[c].iloc[0])
        m = re.match(r'^(\d+)\.', q_text.strip())
        if m:
            q_num_to_col[int(m.group(1))] = c

    data = df.iloc[1:].copy()
    data = data.dropna(subset=['Participant Public ID'])

    records = {}
    for _, row in data.iterrows():
        pid = str(row['Participant Public ID'])

        # Score each item
        item_scores = {}
        for q_num, col in q_num_to_col.items():
            val = pd.to_numeric(row[col], errors='coerce')
            if not np.isnan(val):
                item_scores[q_num] = score_aq_item(q_num, val)

        rec = {
            'aq_total': sum(item_scores.values()),
            'aq_items': len(item_scores),
        }

        # Subscale scores
        for subscale_name, subscale_items in AQ_SUBSCALES.items():
            sub_scores = [item_scores[q] for q in subscale_items if q in item_scores]
            rec[subscale_name] = sum(sub_scores)

        records[pid] = rec

    print(f"[AQ50] {len(records)} participants scored (total + 5 subscales)")
    return records


def parse_dass21():
    """Parse DASS21 questionnaire (Gorilla WIDE format) with 3 subscales."""
    filepath = os.path.join(DATA_DIR, 'data_exp_259923-vall_questionnaire-mywr.csv')
    df = pd.read_csv(filepath)

    dep_items = {3,5,10,13,16,17,21}
    anx_items = {2,4,7,9,15,19,20}
    str_items = {1,6,8,11,12,14,18}

    quantised_cols = [c for c in df.columns if 'Quantised' in c]
    item_cols = [(i+1, c) for i, c in enumerate(quantised_cols) if i < 21]

    data = df.iloc[1:].copy()
    data = data.dropna(subset=['Participant Public ID'])

    records = {}
    for _, row in data.iterrows():
        pid = str(row['Participant Public ID'])
        dep_score = anx_score = str_score = 0
        for item_num, col in item_cols:
            val = pd.to_numeric(row[col], errors='coerce')
            if np.isnan(val):
                continue
            score = val - 1  # Gorilla quantised 1-4 -> DASS 0-3
            if item_num in dep_items: dep_score += score
            elif item_num in anx_items: anx_score += score
            elif item_num in str_items: str_score += score

        # DASS21 scores multiplied by 2 to match DASS42 norms
        records[pid] = {
            'dass_depression': dep_score * 2,
            'dass_anxiety': anx_score * 2,
            'dass_stress': str_score * 2
        }

    print(f"[DASS21] {len(records)} participants scored")
    return records


def parse_fluid_intelligence():
    """Parse Fluid Intelligence test using Gorilla's Store: FI_Score (adaptive test, ~12-14 items)."""
    filepath = os.path.join(DATA_DIR, 'data_exp_259923-vall_task-6hny.csv')
    df = pd.read_csv(filepath, low_memory=False)

    records = {}
    for pid in df['Participant Public ID'].dropna().unique():
        pid_str = str(pid)
        p_data = df[df['Participant Public ID'].astype(str) == pid_str]

        # FI_Score is a running cumulative score; last value = final score
        fi_vals = pd.to_numeric(p_data['Store: FI_Score'], errors='coerce').dropna()
        fi_score = fi_vals.iloc[-1] if len(fi_vals) > 0 else np.nan

        # Count total items attempted
        item_displays = p_data[p_data['Display'].str.contains('Trial', case=False, na=False)]
        fi_items = len(item_displays['Display'].unique())  # unique display types = item count proxy

        # More accurate: count unique Question_No
        q_nos = p_data['Spreadsheet: Question_No'].dropna().unique()
        fi_total = len(q_nos)

        records[pid_str] = {
            'fi_score': fi_score,
            'fi_total_items': fi_total
        }

    print(f"[Fluid Intelligence] {len(records)} participants (adaptive test, score = cumulative correct)")
    return records


# Step 2: build participant metadata with group assignment

def build_metadata(demo, aq, dass, fi):
    """Combine all questionnaire data and assign groups."""

    all_pids = set(demo.keys()) | set(aq.keys())

    rows = []
    for pid in sorted(all_pids):
        row = {'participant_id': pid}

        if pid in demo:
            row.update(demo[pid])
        if pid in aq:
            row.update(aq[pid])  # includes aq_total + 5 subscales
        if pid in dass:
            row.update(dass[pid])
        if pid in fi:
            row.update(fi[pid])

        rows.append(row)

    meta = pd.DataFrame(rows)

    # Group Assignment — auto-rule based on self-report diagnosis + AQ cutoff.
    # Stored as `selfReportGroup` so it is always present and reproducible from
    # raw data alone. The canonical analytic label (`group`) is seeded from the
    # auto-rule here and may be overridden later in run() from an existing
    # participant_metadata.csv — this is how the project's hand-curated group
    # labels are preserved.
    def assign_group(row):
        diag = str(row.get('asc_diagnosis', '')).lower().strip()
        aq_score = row.get('aq_total', 0)
        if diag in ['yes', 'y']:
            return 'ASD'
        elif aq_score >= AQ_CLINICAL_CUTOFF:
            return 'NT_highAQ'
        else:
            return 'NT'

    meta['selfReportGroup'] = meta.apply(assign_group, axis=1)
    # Canonical labels default to the self-report rule. run() will overwrite
    # any matching rows from an existing curated participant_metadata.csv so
    # the hand-curated ASD / NT assignment from the report is preserved.
    meta['group'] = meta['selfReportGroup']

    print(f"\n[Metadata] Self-report group assignment (auto-rule):")
    print(f"  ASD (formal diagnosis): {(meta['selfReportGroup'] == 'ASD').sum()}")
    print(f"  NT (no diagnosis, AQ < {AQ_CLINICAL_CUTOFF}): {(meta['selfReportGroup'] == 'NT').sum()}")
    print(f"  NT_highAQ (no diagnosis, AQ >= {AQ_CLINICAL_CUTOFF}): {(meta['selfReportGroup'] == 'NT_highAQ').sum()}")

    return meta


# Step 3: load and merge all task data

def load_all_task_data():
    """Load all 10 main task files, handle v9 multi-seed issue (keep first seed only)."""

    task_files = sorted([f for f in os.listdir(DATA_DIR)
                         if '_task-' in f and '6hny' not in f])

    print(f"\n[Task Loading] Found {len(task_files)} main task files")

    all_participant_data = {}

    for tf in task_files:
        key = tf.split('_task-')[1].replace('.csv', '')
        filepath = os.path.join(DATA_DIR, tf)
        df = pd.read_csv(filepath, low_memory=False)

        trials = df[
            (df['Display'] == 'Stimuli_Response') &
            (df['Screen'] == 'Screen 1')
        ].copy()

        for pid_raw in trials['Participant Public ID'].dropna().unique():
            pid = str(pid_raw)
            p_data = trials[trials['Participant Public ID'].astype(str) == pid].copy()
            p_data = p_data[p_data['Spreadsheet: Phase'] != 'practice'].copy()

            seeds = p_data['Spreadsheet: Seed'].dropna().unique()
            if len(seeds) > 1:
                seed_first_ts = {s: p_data[p_data['Spreadsheet: Seed'] == s]['UTC Timestamp'].min() for s in seeds}
                first_seed = min(seed_first_ts, key=seed_first_ts.get)
                n_before = len(p_data)
                p_data = p_data[p_data['Spreadsheet: Seed'] == first_seed].copy()
                print(f"  [v9 fix] {pid}: seeds {list(seeds)} -> keeping seed {first_seed} "
                      f"({n_before} -> {len(p_data)} trials)")

            if pid in all_participant_data:
                print(f"  WARNING: {pid} in multiple task files, appending")
                all_participant_data[pid] = pd.concat([all_participant_data[pid], p_data])
            else:
                all_participant_data[pid] = p_data

    print(f"[Task Loading] Loaded {len(all_participant_data)} participants")

    for pid in sorted(all_participant_data.keys()):
        n = len(all_participant_data[pid])
        if n != EXPECTED_TRIALS:
            print(f"  {pid}: {n} trials (expected {EXPECTED_TRIALS})")

    return all_participant_data


# Step 4: clean individual participant data

def clean_participant(p_data, pid):
    """Apply cleaning pipeline to one participant. No RT-based exclusion; only timeouts -> NaN.
    `pid` is whatever label the caller wants stamped on the output (anonymous P###
    when called from run(), Prolific Public ID when called standalone)."""

    col_map = {
        'Event Index': 'event_index',
        'Participant Public ID': 'participant_id',
        'Experiment Version': 'experiment_version',
        'Spreadsheet: Condition': 'condition',
        'Spreadsheet: Seed': 'seed',
        'Spreadsheet: Trial_Number': 'trial_number',
        'Spreadsheet: Phase': 'phase',
        'Spreadsheet: Phase_Number': 'phase_number',
        'Spreadsheet: Block': 'block',
        'Spreadsheet: Equivalence_Group': 'equivalence_group',
        'Spreadsheet: Face_Pair_Type': 'face_pair_type',
        'Spreadsheet: Facial_Cue_Accuracy': 'facial_cue_accuracy',
        'Spreadsheet: Verbal_Cue_Accuracy': 'verbal_cue_accuracy',
        'Spreadsheet: Facial_Cue_Valid': 'facial_cue_valid',
        'Spreadsheet: Verbal_Cue_Valid': 'verbal_cue_valid',
        'Spreadsheet: Facial_Cue_Direction': 'facial_cue_direction',
        'Spreadsheet: Verbal_Cue_Direction': 'verbal_cue_direction',
        'Spreadsheet: Cue_Conflict': 'cue_conflict',
        'Spreadsheet: Facial_Block': 'facial_block',
        'Spreadsheet: Verbal_Block': 'verbal_block',
        'Spreadsheet: Facial_Block_State': 'facial_block_state',
        'Spreadsheet: Verbal_Block_State': 'verbal_block_state',
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
        'Spreadsheet: Key_Left': 'key_left',
        'Spreadsheet: Key_Right': 'key_right',
        'Spreadsheet: Correct_Key': 'correct_key',
        'Response': 'response_key',
        'Correct': 'correct',
        'Reaction Time': 'rt_raw',
    }

    available_cols = {k: v for k, v in col_map.items() if k in p_data.columns}
    clean = p_data[list(available_cols.keys())].rename(columns=available_cols).copy()
    # CRITICAL: Gorilla's raw "Event Index" column is read as a string because
    # one row in the export contains the literal "END OF FILE". Without this
    # numeric conversion, sort_values below sorts lexicographically ("100" < "40")
    # and the trial sequence handed to the HGF is shuffled, breaking the fits.
    clean['event_index'] = pd.to_numeric(clean['event_index'], errors='coerce').astype('Int64')
    clean = clean.sort_values('event_index').reset_index(drop=True)

    # Stamp the (possibly anonymised) ID into the participant_id column so the
    # raw Prolific Public ID never leaves this function.
    clean['participant_id'] = pid

    # Type conversions
    clean['rt_raw'] = pd.to_numeric(clean['rt_raw'], errors='coerce')
    for col in ['facial_cue_valid', 'verbal_cue_valid', 'cue_conflict',
                'card_left_points', 'card_right_points', 'correct']:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors='coerce')
    if 'trial_number' in clean.columns:
        clean['trial_number'] = pd.to_numeric(clean['trial_number'], errors='coerce').astype('Int64')

    # Behavioral encoding
    clean['is_timeout'] = clean['response_key'].isna()

    clean['chose_left'] = np.where(
        clean['response_key'] == 'Z', 1.0,
        np.where(clean['response_key'] == 'M', 0.0, np.nan)
    )

    if 'facial_cue_direction' in clean.columns:
        facial_left = (clean['facial_cue_direction'] == 'left').astype(float)
        clean['followed_facial'] = np.where(
            clean['is_timeout'], np.nan,
            (clean['chose_left'] == facial_left).astype(float)
        )

    if 'verbal_cue_direction' in clean.columns:
        verbal_left = (clean['verbal_cue_direction'] == 'left').astype(float)
        clean['followed_verbal'] = np.where(
            clean['is_timeout'], np.nan,
            (clean['chose_left'] == verbal_left).astype(float)
        )

    if 'correct_side' in clean.columns:
        correct_left = (clean['correct_side'] == 'left').astype(float)
        clean['chose_high_reward'] = np.where(
            clean['is_timeout'], np.nan,
            (clean['chose_left'] == correct_left).astype(float)
        )

    # Response for model: all responses kept, only timeouts = NaN
    clean['response_for_model'] = clean['chose_left'].copy()
    # Timeouts already NaN from chose_left encoding

    # HGF input columns
    clean['trial_order'] = range(1, len(clean) + 1)

    eg_map = {'A': 1, 'B': 2, 'C': 3}
    if 'equivalence_group' in clean.columns:
        clean['eg_numeric'] = clean['equivalence_group'].map(eg_map)

    phase_map = {'calibration': 1, 'verbal_volatile': 2, 'overlap': 3, 'facial_volatile': 4}
    if 'phase' in clean.columns:
        clean['phase_numeric'] = clean['phase'].map(phase_map)

    if 'facial_cue_direction' in clean.columns:
        clean['facial_direction_left'] = (clean['facial_cue_direction'] == 'left').astype(int)
    if 'verbal_cue_direction' in clean.columns:
        clean['verbal_direction_left'] = (clean['verbal_cue_direction'] == 'left').astype(int)

    # Quality report
    n_total = len(clean)
    n_responded = (~clean['is_timeout']).sum()
    n_timeout = clean['is_timeout'].sum()

    report = {
        'participant_id': pid,
        'n_trials': n_total,
        'n_responded': n_responded,
        'n_timeout': n_timeout,
        'timeout_rate': n_timeout / n_total if n_total > 0 else 0,
    }

    responded = clean[~clean['is_timeout']]
    if len(responded) > 0:
        report['accuracy'] = responded['correct'].mean() if 'correct' in responded.columns else np.nan
        report['mean_rt'] = responded['rt_raw'].mean()
        report['median_rt'] = responded['rt_raw'].median()
        report['chose_left_rate'] = responded['chose_left'].mean()
        report['side_bias'] = abs(responded['chose_left'].mean() - 0.5)
        if 'followed_facial' in responded.columns:
            report['follow_facial'] = responded['followed_facial'].mean()
        if 'followed_verbal' in responded.columns:
            report['follow_verbal'] = responded['followed_verbal'].mean()

        cal = responded[responded['phase'] == 'calibration'] if 'phase' in responded.columns else pd.DataFrame()
        report['calibration_accuracy'] = cal['correct'].mean() if len(cal) > 0 and 'correct' in cal.columns else np.nan

    # Quality flags (informational only)
    report['flag_high_timeout'] = report['timeout_rate'] > MAX_TIMEOUT_RATE
    report['flag_low_cal_acc'] = report.get('calibration_accuracy', 1) < MIN_ACCURACY
    report['flag_side_bias'] = report.get('side_bias', 0) > MAX_SIDE_BIAS
    report['flag_wrong_trial_count'] = n_total != EXPECTED_TRIALS
    report['any_flag'] = any([
        report['flag_high_timeout'], report['flag_low_cal_acc'],
        report['flag_side_bias'], report['flag_wrong_trial_count']
    ])

    return clean, report


# Step 5: export

def export_participant(clean, pid, output_dir):
    """Export cleaned data for one participant."""

    csv_path = os.path.join(output_dir, f'{pid}_cleaned.csv')
    clean.to_csv(csv_path, index=False)

    hgf_cols = [
        'trial_order', 'trial_number', 'phase', 'phase_numeric',
        'equivalence_group', 'eg_numeric',
        'facial_cue_valid', 'verbal_cue_valid',
        'facial_cue_accuracy', 'verbal_cue_accuracy',
        'facial_cue_direction', 'verbal_cue_direction',
        'facial_direction_left', 'verbal_direction_left',
        'card_left_points', 'card_right_points', 'point_difference',
        'correct_side',
        'response_key', 'chose_left', 'rt_raw',
        'followed_facial', 'followed_verbal', 'chose_high_reward',
        'is_timeout', 'response_for_model',
    ]
    available_hgf_cols = [c for c in hgf_cols if c in clean.columns]
    hgf_df = clean[available_hgf_cols].copy()
    hgf_path = os.path.join(output_dir, f'{pid}_hgf_input.csv')
    hgf_df.to_csv(hgf_path, index=False)

    try:
        from scipy.io import savemat
        mat_data = {
            'y': clean['response_for_model'].values.astype(float).reshape(-1, 1),
            'u_facial_valid': clean['facial_cue_valid'].values.astype(float).reshape(-1, 1),
            'u_verbal_valid': clean['verbal_cue_valid'].values.astype(float).reshape(-1, 1),
            'u_facial_dir_left': clean['facial_direction_left'].values.astype(float).reshape(-1, 1),
            'u_verbal_dir_left': clean['verbal_direction_left'].values.astype(float).reshape(-1, 1),
            'u_card_left_pts': clean['card_left_points'].values.astype(float).reshape(-1, 1),
            'u_card_right_pts': clean['card_right_points'].values.astype(float).reshape(-1, 1),
            'u_eg': clean['eg_numeric'].values.astype(float).reshape(-1, 1),
            'phase': clean['phase_numeric'].values.astype(float).reshape(-1, 1),
            'trial_number': clean['trial_order'].values.astype(float).reshape(-1, 1),
            'rt': clean['rt_raw'].values.astype(float).reshape(-1, 1),
        }
        mat_path = os.path.join(output_dir, f'{pid}_hgf_input.mat')
        savemat(mat_path, mat_data)
    except ImportError:
        pass

    return csv_path


# Anonymisation

def build_pid_map(prolific_pids, lookup_path=PID_LOOKUP_FILE, prefix=ANON_PREFIX):
    """Return a dict mapping each Prolific Public ID to a stable anon label
    (P001, P002, ...). Existing entries in the lookup CSV are preserved; only
    new PIDs receive new labels. The updated mapping is written back."""

    prolific_pids = [str(p) for p in prolific_pids if pd.notna(p)]
    existing = {}
    if os.path.exists(lookup_path):
        df = pd.read_csv(lookup_path, dtype=str)
        if {'prolific_id', 'anon_id'}.issubset(df.columns):
            existing = dict(zip(df['prolific_id'], df['anon_id']))

    used = set(existing.values())
    next_idx = max([int(v[len(prefix):]) for v in used if v.startswith(prefix)
                    and v[len(prefix):].isdigit()], default=0) + 1

    pid_map = dict(existing)
    for pid in sorted(set(prolific_pids)):
        if pid not in pid_map:
            pid_map[pid] = f"{prefix}{next_idx:03d}"
            next_idx += 1

    # Persist (sorted by anon_id for readability)
    rows = sorted(pid_map.items(), key=lambda kv: kv[1])
    pd.DataFrame(rows, columns=['prolific_id', 'anon_id']).to_csv(lookup_path, index=False)
    print(f"[Anonymise] {len(pid_map)} PID(s) mapped; lookup saved to {lookup_path}")
    print(f"[Anonymise] DO NOT commit {os.path.basename(lookup_path)} to a public repo.")
    return pid_map


# Main pipeline

def run():
    print("=" * 70)
    print("COMPREHENSIVE DATA CLEANING PIPELINE")
    print("HGF Facial-Verbal Cue Learning Study (Gorilla Exp 259923)")
    print("=" * 70)

    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Parse questionnaires
    print("\n--- STEP 1: Parsing Questionnaires ---")
    demo = parse_demographics()
    aq = parse_aq50()
    dass = parse_dass21()
    fi = parse_fluid_intelligence()

    # 2. Build metadata
    print("\n--- STEP 2: Building Participant Metadata ---")
    meta = build_metadata(demo, aq, dass, fi)

    # 3. Load task data
    print("\n--- STEP 3: Loading All Task Data ---")
    task_data = load_all_task_data()

    # 3b. Build the Prolific -> anonymous ID mapping
    if ANONYMISE:
        all_prolific = set(meta['participant_id'].astype(str)) | set(map(str, task_data.keys()))
        pid_map = build_pid_map(all_prolific)
        meta['participant_id'] = meta['participant_id'].astype(str).map(pid_map).fillna(meta['participant_id'])
    else:
        pid_map = {pid: pid for pid in set(meta['participant_id']).union(task_data.keys())}

    # 4a. Apply canonical override from an existing curated metadata.
    #
    # Two pieces of curation live in data/participant_metadata.csv:
    #   (i)  the row set      — which participants are in the analytic sample
    #                           (e.g., participants with 100% timeout were
    #                           excluded by hand and are not in canonical)
    #   (ii) the `group` label — ASD / NT after Prolific-recruitment-based
    #                           hand-recoding (the auto-rule does not
    #                           reproduce this)
    #
    # If a curated participant_metadata.csv is present we treat it as the
    # source of truth for BOTH (i) and (ii):
    #   - rows in the pipeline output that aren't in canonical get DROPPED
    #     (with a console warning), so re-running cleaning never silently
    #     re-introduces a participant who was deliberately excluded
    #   - rows in canonical that aren't in the pipeline get listed (the user
    #     should investigate; they don't appear in this run's output)
    #   - the canonical `group` overrides the auto-rule for matching rows
    #
    # If no curated file exists (first ever run), we fall back to writing
    # all pipeline rows with auto-rule labels and flag this in the console.
    meta_path = os.path.join(DATA_ROOT, 'participant_metadata.csv')
    qr_path   = os.path.join(DATA_ROOT, 'quality_report.csv')
    canonical_pids = None  # populated only if curated file exists

    if os.path.exists(meta_path):
        existing = pd.read_csv(meta_path, dtype=str)
        if {'participant_id', 'group'}.issubset(existing.columns):
            canonical = dict(zip(existing['participant_id'], existing['group']))
            canonical_pids = set(canonical)

            # (ii) Override group labels for shared rows
            n_overridden = 0
            for i in meta.index:
                pid = meta.at[i, 'participant_id']
                if pid in canonical:
                    if meta.at[i, 'group'] != canonical[pid]:
                        n_overridden += 1
                    meta.at[i, 'group'] = canonical[pid]

            # (i) Subset to canonical row set
            extra_pids   = sorted(set(meta['participant_id']) - canonical_pids)
            missing_pids = sorted(canonical_pids - set(meta['participant_id']))

            print(f"\n[Curation] Read {len(canonical)} canonical row(s) from "
                  f"{os.path.basename(meta_path)}; overrode {n_overridden} auto-rule label(s).")
            if extra_pids:
                print(f"[Curation] Dropping {len(extra_pids)} pipeline row(s) absent from "
                      f"canonical (e.g. excluded by hand for quality reasons): {extra_pids}")
                meta = meta[meta['participant_id'].isin(canonical_pids)].reset_index(drop=True)
            if missing_pids:
                print(f"[Curation] {len(missing_pids)} canonical row(s) missing from this run "
                      f"(raw data may be incomplete): {missing_pids}")
        else:
            print(f"\n[Curation] {os.path.basename(meta_path)} exists but lacks "
                  f"participant_id/group columns; using auto-rule labels.")
    else:
        print(f"\n[Curation] No existing {os.path.basename(meta_path)} found; "
              f"writing auto-rule labels for all pipeline rows (review and curate by hand if needed).")

    # 4. Clean each participant — runs after the curation override so the
    # quality_report inherits canonical group labels from `meta`.
    print("\n--- STEP 4: Cleaning Individual Participants ---")
    quality_reports = []

    for prolific_pid in sorted(task_data.keys()):
        anon_pid = pid_map[str(prolific_pid)]

        # Skip participants that the curated metadata excludes — keeping
        # cleaned/* aligned with the canonical analytic sample.
        if canonical_pids is not None and anon_pid not in canonical_pids:
            continue

        clean, report = clean_participant(task_data[prolific_pid], anon_pid)

        pid_meta = meta[meta['participant_id'] == anon_pid]
        if len(pid_meta) > 0:
            row = pid_meta.iloc[0]
            report['group'] = row['group']
            report['aq_total'] = row.get('aq_total', np.nan)
            report['condition'] = clean['condition'].iloc[0] if 'condition' in clean.columns else ''
            report['seed'] = clean['seed'].iloc[0] if 'seed' in clean.columns else ''

        quality_reports.append(report)
        export_participant(clean, anon_pid, OUTPUT_DIR)

    # 5. Save metadata and reports — write canonical labels straight to the
    # live files; no .draft.csv branch.
    print("\n--- STEP 5: Saving Outputs ---")
    meta.to_csv(meta_path, index=False)
    print(f"  Metadata: {meta_path}")
    qr = pd.DataFrame(quality_reports)
    qr.to_csv(qr_path, index=False)
    print(f"  Quality report: {qr_path}")

    # 6. Print Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nParticipants processed: {len(quality_reports)}")

    if len(qr) > 0:
        print(f"\n--- Group Distribution ---")
        if 'group' in qr.columns:
            print(qr['group'].value_counts().to_string())

        print(f"\n--- Quality Flags ---")
        flagged = qr[qr['any_flag'] == True]
        print(f"Participants with flags: {len(flagged)} / {len(qr)}")
        for _, row in flagged.iterrows():
            flags = []
            if row['flag_high_timeout']: flags.append(f"timeout={row['timeout_rate']:.1%}")
            if row.get('flag_low_cal_acc', False): flags.append(f"cal_acc={row.get('calibration_accuracy', 0):.1%}")
            if row.get('flag_side_bias', False): flags.append(f"side_bias={row.get('side_bias', 0):.1%}")
            if row['flag_wrong_trial_count']: flags.append(f"trials={row['n_trials']}")
            print(f"  {row['participant_id']}: {', '.join(flags)}")

        print(f"\n--- Behavioral Summary ---")
        for col in ['n_trials', 'n_responded', 'timeout_rate', 'accuracy', 'mean_rt', 'follow_facial', 'follow_verbal']:
            if col in qr.columns:
                vals = qr[col].dropna()
                if len(vals) > 0:
                    print(f"  {col}: M={vals.mean():.3f}, SD={vals.std():.3f}, range=[{vals.min():.3f}, {vals.max():.3f}]")

        if 'group' in qr.columns and len(qr['group'].unique()) > 1:
            print(f"\n--- ASD vs NT Comparison ---")
            for col in ['accuracy', 'mean_rt', 'follow_facial', 'follow_verbal', 'timeout_rate', 'aq_total']:
                if col in qr.columns:
                    asd = qr[qr['group'] == 'ASD'][col].dropna()
                    nt = qr[qr['group'] == 'NT'][col].dropna()
                    if len(asd) > 0 and len(nt) > 0:
                        print(f"  {col}: ASD M={asd.mean():.3f}(SD={asd.std():.3f}), NT M={nt.mean():.3f}(SD={nt.std():.3f})")

    # AQ subscale summary
    print(f"\n--- AQ50 Subscale Means by Group ---")
    for subscale in AQ_SUBSCALES.keys():
        if subscale in meta.columns:
            for grp in ['ASD', 'NT', 'NT_highAQ']:
                vals = meta[meta['group'] == grp][subscale].dropna()
                if len(vals) > 0:
                    print(f"  {subscale} [{grp}]: M={vals.mean():.1f}, SD={vals.std():.1f}")

    # FI summary
    print(f"\n--- Fluid Intelligence by Group ---")
    if 'fi_score' in meta.columns:
        for grp in ['ASD', 'NT', 'NT_highAQ']:
            vals = meta[meta['group'] == grp]['fi_score'].dropna()
            if len(vals) > 0:
                print(f"  FI [{grp}]: M={vals.mean():.1f}, SD={vals.std():.1f}, range=[{vals.min():.0f}, {vals.max():.0f}]")

    # DASS summary
    print(f"\n--- DASS21 by Group ---")
    for subscale in ['dass_depression', 'dass_anxiety', 'dass_stress']:
        if subscale in meta.columns:
            for grp in ['ASD', 'NT', 'NT_highAQ']:
                vals = meta[meta['group'] == grp][subscale].dropna()
                if len(vals) > 0:
                    print(f"  {subscale} [{grp}]: M={vals.mean():.1f}, SD={vals.std():.1f}")

    # Text report
    report_path = os.path.join(DATA_ROOT, 'cleaning_report.txt')
    with open(report_path, 'w') as f:
        f.write("Data Cleaning Report\n")
        f.write(f"Generated by comprehensive_cleaning_pipeline.py\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Participants processed: {len(quality_reports)}\n")
        f.write(f"RT exclusion: NONE (all responses retained; only timeouts -> NaN)\n")
        f.write(f"Output directory: {OUTPUT_DIR}\n\n")
        if len(qr) > 0 and 'group' in qr.columns:
            f.write("Group distribution:\n")
            f.write(qr['group'].value_counts().to_string() + "\n\n")
            f.write("Flagged participants:\n")
            for _, row in flagged.iterrows():
                flags = []
                if row['flag_high_timeout']: flags.append(f"timeout={row['timeout_rate']:.1%}")
                if row.get('flag_low_cal_acc', False): flags.append(f"cal_acc={row.get('calibration_accuracy', 0):.1%}")
                if row.get('flag_side_bias', False): flags.append(f"side_bias={row.get('side_bias', 0):.1%}")
                if row['flag_wrong_trial_count']: flags.append(f"trials={row['n_trials']}")
                f.write(f"  {row['participant_id']}: {', '.join(flags)}\n")

    print(f"\n  Report: {report_path}")
    print(f"  Cleaned files: {OUTPUT_DIR}/")
    print("\nDone!")

    return meta, qr


if __name__ == '__main__':
    meta, qr = run()

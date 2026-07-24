#!/bin/bash
#SBATCH --job-name=p5_extract_job1
#SBATCH --partition=cpu_medium
#SBATCH --array=0-7
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/extract_job1_%A_%a.out
#SBATCH --error=logs/extract_job1_%A_%a.err

set -euo pipefail

PROJECT=/gpfs/home/dk5565/forecasting/phase5/data_extraction
cd "$PROJECT"

mkdir -p logs output

# ──────────────────────────────────────────────────────────────────────────────
# Split the 42 patients from job1_patients.txt across 8 sub-jobs
# ──────────────────────────────────────────────────────────────────────────────
PATIENT_FILE="$PROJECT/job1_patients.txt"
NUM_SUB_JOBS=8
SUB_IDX=$SLURM_ARRAY_TASK_ID

TOTAL_PATIENTS=$(wc -l < "$PATIENT_FILE")
# Compute start (1-based) and count for this sub-job
BASE_SIZE=$((TOTAL_PATIENTS / NUM_SUB_JOBS))
REMAINDER=$((TOTAL_PATIENTS % NUM_SUB_JOBS))

# First REMAINDER sub-jobs get (BASE_SIZE + 1) patients, the rest get BASE_SIZE
if [ "$SUB_IDX" -lt "$REMAINDER" ]; then
    COUNT=$((BASE_SIZE + 1))
    START=$((SUB_IDX * (BASE_SIZE + 1) + 1))
else
    COUNT=$BASE_SIZE
    START=$((REMAINDER * (BASE_SIZE + 1) + (SUB_IDX - REMAINDER) * BASE_SIZE + 1))
fi

# Extract this sub-job's patient IDs to a temp file
SUB_PATIENT_FILE=$(mktemp "$PROJECT/logs/sub_patients_${SLURM_ARRAY_JOB_ID}_${SUB_IDX}_XXXXXX.txt")
sed -n "${START},$((START + COUNT - 1))p" "$PATIENT_FILE" > "$SUB_PATIENT_FILE"

echo "========================================"
echo "Phase 5 — Feature Extraction (Job 1 Sub-tasks)"
echo "Start:    $(date)"
echo "Node:     $(hostname)"
echo "Job:      $SLURM_ARRAY_JOB_ID"
echo "Sub-task: $SUB_IDX / $NUM_SUB_JOBS"
echo "Patients: $COUNT (lines $START-$((START + COUNT - 1)) of $TOTAL_PATIENTS)"
echo "========================================"
cat "$SUB_PATIENT_FILE"
echo "========================================"

# Activate conda environment
source /gpfs/share/apps/anaconda3/cpu/5.3.1/etc/profile.d/conda.sh
conda activate /gpfs/home/dk5565/.conda/envs/CSDI

python3 extract_features.py \
    --config config/pipeline_config.yaml \
    --patient-file "$SUB_PATIENT_FILE" \
    --job-idx 1 \
    --output-suffix "_sub_$(printf '%03d' $SUB_IDX)"

echo "========================================"
echo "End:      $(date)"
echo "========================================"

# Clean up temp file
rm -f "$SUB_PATIENT_FILE"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-prediction}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
STABLEWM_REPO="${STABLEWM_REPO:-../stable-worldmodel}"
ARTIFACT_ROOT="${CONTEXTWORLD_ARTIFACT_ROOT:-$(dirname "$(dirname "$ROOT")")/data/world_model/context_world}"
OUTPUT_DIR="${OUTPUT_DIR:-$ARTIFACT_ROOT/evaluation/history3}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
RUNNER="$ROOT/scripts/run_tworoom_history3_eval.py"
SKIP_PREDICTION="${SKIP_PREDICTION:-0}"
E1_RESULT="$OUTPUT_DIR/e1_speed_paired.json"
E2_RESULT="$OUTPUT_DIR/e2_speed_natural.json"
E4_RESULT="$OUTPUT_DIR/e4_speed_ctx_quick_s42.json"

case "$MODE" in
  prediction|smoke|quick|full)
    ;;
  *)
    echo "Usage: bash scripts/run_speed_stage1_eval.sh [prediction|smoke|quick|full]" >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "[speed-stage1] mode=$MODE device=$DEVICE"
if [[ "$SKIP_PREDICTION" == "1" ]]; then
  if [[ ! -s "$E1_RESULT" || ! -s "$E2_RESULT" ]]; then
    echo "SKIP_PREDICTION=1 requires existing E1/E2 result files" >&2
    exit 2
  fi
  echo "[speed-stage1] reusing existing E1/E2 results"
else
  echo "[speed-stage1] E1 paired prediction + E2 natural prediction"
  "$PYTHON_BIN" "$RUNNER" \
    --suite e1-e2-pred \
    --stablewm-repo "$STABLEWM_REPO" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_DIR/e1_e2_prediction.log"
fi

if [[ "$MODE" == "smoke" || "$MODE" == "quick" || "$MODE" == "full" ]]; then
  if [[ "$MODE" == "full" ]]; then
    echo "[speed-stage1] E3 no-context planning: seeds=42..47 num_eval=50/seed"
  elif [[ "$MODE" == "quick" ]]; then
    echo "[speed-stage1] E3 qualitative planning: speeds=3.1,5.0,7.0 seed=42 num_eval=9"
  else
    echo "[speed-stage1] E3 no-context planning: protocol smoke only"
  fi
  "$PYTHON_BIN" "$RUNNER" \
    --suite e3-no-context-plan \
    --stablewm-repo "$STABLEWM_REPO" \
    --planning-profile "$MODE" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_DIR/e3_no_context_planning_${MODE}.log"
fi

if [[ "$MODE" == "quick" ]]; then
  echo "[speed-stage1] E4 paired-context planning: fixed K=2 correct vs wrong, 9 paired queries"
  "$PYTHON_BIN" "$ROOT/scripts/eval_tworoom_icl_planning.py" \
    --stablewm-repo "$STABLEWM_REPO" \
    --device "$DEVICE" \
    --output "$E4_RESULT" \
    2>&1 | tee "$LOG_DIR/e4_paired_context_planning_quick.log"
fi

E3_RESULT=""
E3_SOURCE="/dev/null"
if [[ "$MODE" == "smoke" || "$MODE" == "quick" || "$MODE" == "full" ]]; then
  E3_RESULT="$OUTPUT_DIR/e3_speed_noctx_${MODE}.json"
  E3_SOURCE="$E3_RESULT"
fi
E4_SOURCE="/dev/null"
if [[ "$MODE" == "quick" ]]; then
  E4_SOURCE="$E4_RESULT"
fi
SUMMARY="$OUTPUT_DIR/speed_${MODE}_summary.json"

if command -v jq >/dev/null 2>&1; then
  jq -n \
    --arg e1_path "$E1_RESULT" \
    --arg e2_path "$E2_RESULT" \
    --arg e3_path "$E3_RESULT" \
    --arg e4_path "$E4_RESULT" \
    --slurpfile e1 "$E1_RESULT" \
    --slurpfile e2 "$E2_RESULT" \
    --slurpfile e3 "$E3_SOURCE" \
    --slurpfile e4 "$E4_SOURCE" \
    '{
      schema_version: 1,
      scope: "speed_only",
      model_id: "M_orig",
      e1_paired_prediction: {
        path: $e1_path,
        status: $e1[0].status,
        frozen: $e1[0].frozen_weight_audit.passed,
        bundles: $e1[0].data.bundles,
        families: $e1[0].data.families,
        k2_signal: $e1[0].diagnostic_signals.speed,
        k2_contrast: (
          [$e1[0].contrasts[]
           | select(.family == "speed" and .context_budget == 2)][0]
        )
      },
      e2_natural_prediction: {
        path: $e2_path,
        status: $e2[0].status,
        frozen: $e2[0].frozen_weight_audit.passed,
        clips: $e2[0].data.clips,
        families: $e2[0].data.families,
        contrasts: [
          $e2[0].contrasts[] | select(.family == "speed")
        ]
      },
      e3_no_context_planning: (
        if ($e3 | length) == 0 then
          {status: "not_run"}
        else
          {
            path: $e3_path,
            status: $e3[0].status,
            profile: $e3[0].profile,
            evidence_level: $e3[0].evidence_level,
            eval_seeds: $e3[0].protocol.eval_seeds,
            num_eval_per_seed: $e3[0].protocol.num_eval_per_seed,
            evaluations: $e3[0].aggregate.evaluations,
            successes: $e3[0].aggregate.successes,
            success_rate: $e3[0].aggregate.success_rate,
            std_seed_success_rate: $e3[0].aggregate.std_seed_success_rate,
            pooled_success_rate: $e3[0].aggregate.pooled_success_rate,
            factor_readback_passed: $e3[0].aggregate.factor_readback_passed,
            original_tworoom_reference: $e3[0].original_tworoom_reference
          }
        end
      ),
      e4_paired_context_planning: (
        if ($e4 | length) == 0 then
          {status: "not_run"}
        else
          {
            path: $e4_path,
            status: $e4[0].status,
            evidence_level: $e4[0].evidence_level,
            context_budget: $e4[0].selection.context_budget,
            conditions: $e4[0].selection.conditions,
            queries: $e4[0].aggregate.queries,
            pairing_audit_passed: $e4[0].pairing_audit.passed,
            frozen: $e4[0].frozen_weight_audit.passed,
            aggregate: ($e4[0].aggregate | del(.pairs))
          }
        end
      ),
      interpretation_guard: {
        e1_primary: "fixed_length_k2_correct_vs_wrong",
        e2_role: "continuous_same_episode_supporting_evidence",
        e3_role: "no_context_planning_baseline_only",
        e4_role: "fixed_length_k2_correct_vs_wrong_paired_planning",
        e4_evidence: "quick_probe_is_qualitative_not_formal_confirmation"
      }
    }' | tee "$SUMMARY"
else
  echo "[speed-stage1] jq not found; skipping compact summary" >&2
fi

echo "[speed-stage1] completed"
echo "[speed-stage1] E1: $E1_RESULT"
echo "[speed-stage1] E2: $E2_RESULT"
if [[ -n "$E3_RESULT" ]]; then
  echo "[speed-stage1] E3: $E3_RESULT"
fi
if [[ "$MODE" == "quick" ]]; then
  echo "[speed-stage1] E4: $E4_RESULT"
fi
echo "[speed-stage1] summary: $SUMMARY"

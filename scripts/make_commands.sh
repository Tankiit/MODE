#!/usr/bin/env bash
# Generate the full rerun grid (see RERUN_PLAN.md §5), one train_sieve.py
# command per line:
#   commands_small.txt  — GPT-2 Small: paired sweep + matched-token control (75 runs)
#   commands_large.txt  — GPT-2 Large: Table 1 + Table 2 (108 runs)
#
# Requires the code tasks of RERUN_PLAN.md §1 (--log_interval, --out_dir,
# --save_final).  Usage:  bash scripts/make_commands.sh
set -euo pipefail

SEEDS="42 43 44"
COMMON="--eval_interval 1000 --log_interval 100"

declare -A FLAGS=(
  [seonly]="--baseline rho1"
  [su_only]="--baseline su_only"
  [sl_only]="--baseline sl_only"
  [sd_only]="--baseline sd_only"
  [mix4]="--baseline sieve"
  [mix3]="--baseline sieve_no_se"
  [vnmix]="--baseline sieve --variance_normalized_bandit"
  [mix4_bigram]="--baseline sieve --sd_scorer bigram_freq"
  [mix4_noise]="--baseline sieve --sd_scorer random_noise"
  [clm]="--baseline clm"
)
TABLE1_VARIANTS="seonly su_only sl_only sd_only mix4 mix3 vnmix mix4_bigram mix4_noise"

# emit <file> <group> <model> <dataset> <variant> <alpha> <steps> <seed> [extra flags]
emit() {
  local file=$1 group=$2 model=$3 ds=$4 v=$5 a=$6 steps=$7 seed=$8 extra=${9:-}
  local atag name
  atag=$(awk -v a="$a" 'BEGIN { printf "%03d", a * 100 + 0.5 }')
  name="${group}_${model}_${ds}_${v}_a${atag}_st${steps}_s${seed}"
  echo "python train_sieve.py --model $model --dataset $ds ${FLAGS[$v]} --alpha $a" \
       "--steps $steps --seed $seed $COMMON --run_name $name --out_dir runs/$name${extra:+ $extra}" \
       >> "$file"
}

: > commands_small.txt
: > commands_large.txt

# ── B. GPT-2 Small paired sweep (App. C): S_E-only vs Mix-4, 2 corpora, 4 budgets
for ds in wikitext103 owm; do
  for a in 0.3 0.5 0.7 0.9; do
    for seed in $SEEDS; do
      for v in seonly mix4; do
        emit commands_small.txt B gpt2 "$ds" "$v" "$a" 25000 "$seed"
      done
    done
  done
done

# ── C. GPT-2 Small matched-token control (rebuttal remz): ~71.7M retained tokens
#    steps = 71.68M / (8 seq × 1024 tok × α); full CLM (α = 1) for 8,750 steps
for seed in $SEEDS; do
  for pair in "0.3 29167" "0.5 17500" "0.7 12500" "0.9 9722"; do
    read -r a steps <<< "$pair"
    for v in seonly mix4; do
      emit commands_small.txt C gpt2 wikitext103 "$v" "$a" "$steps" "$seed"
    done
  done
  emit commands_small.txt C gpt2 wikitext103 clm 1.0 8750 "$seed"
done

# ── A1. GPT-2 Large, Table 1: 9 variants at α = 0.7 on both corpora
for ds in wikitext103 owm; do
  for seed in $SEEDS; do
    for v in $TABLE1_VARIANTS; do
      extra=""
      # Final checkpoints for the Jaccard / correlation appendices (§7)
      if [[ $seed == 42 ]]; then
        if [[ $ds == wikitext103 && " seonly su_only sl_only sd_only mix4 " == *" $v "* ]] \
           || [[ $ds == owm && $v == mix4 ]]; then
          extra="--save_final"
        fi
      fi
      emit commands_large.txt A1 gpt2l "$ds" "$v" 0.7 25000 "$seed" "$extra"
    done
  done
done

# ── A2. GPT-2 Large, Table 2: α sweep for S_E-only, Mix-4, VN-Mix (α = 0.7 is in A1)
for ds in wikitext103 owm; do
  for a in 0.3 0.5 0.9; do
    for seed in $SEEDS; do
      for v in seonly mix4 vnmix; do
        emit commands_large.txt A2 gpt2l "$ds" "$v" "$a" 25000 "$seed"
      done
    done
  done
done

echo "commands_small.txt: $(wc -l < commands_small.txt) runs"
echo "commands_large.txt: $(wc -l < commands_large.txt) runs"

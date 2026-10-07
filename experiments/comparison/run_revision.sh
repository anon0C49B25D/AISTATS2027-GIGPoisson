#!/usr/bin/env bash
# The additional runs of 2026-10-04.
#
#   factorial    every prior (GIG, gamma, gamma mixture) under every rule (EIG, D-optimality,
#                systematic), Bayes-LUCB, and the GIG with nested estimators of the EIG
#   rivals       the compound-gamma prior and the penalised skew-normal, by EIG
#   overhead     a per-observation overhead c0 on every action, two levels per setting
#   diagnostics  nested-estimator decision agreement, prior-only scores, tests and tables
#   timing       single-process cost per EIG evaluation; run with nothing else on the machine
#
# Every stage is resumable: run.py skips the (episode, agent) runs already on disk. Each
# setting is its own invocation, so a failed gate stops that setting only.
#
# Usage: PY=/path/to/python JOBS=12 experiments/comparison/run_revision.sh [stage]
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
JOBS=${JOBS:-12}
stage=${1:-all}
LOG=experiments/comparison/results/revision.log
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

factorial() {
  for s in bulk gamma hardxray; do
    log "factorial $s"
    $PY experiments/comparison/run.py "$s" --reps 50 --tag fact_ --jobs "$JOBS" --agents \
      eig eig@gamma-poisson eig@gammamix-poisson \
      d-optimality d-optimality@gamma-poisson d-optimality@gammamix-poisson \
      systematic systematic@gamma-poisson systematic@gammamix-poisson \
      lucb lucb@gamma-poisson eig@gig-nmc eig@gig-pce
  done
}

rivals() {
  for s in bulk gamma hardxray; do
    log "rivals $s"
    $PY experiments/comparison/run.py "$s" --reps 50 --tag fact_ --regate --jobs "$JOBS" \
      --agents eig@compoundgamma-poisson eig@logskewnormal-quad-pen
  done
}

overhead() {
  # c0 at half and twice the shortest action of each setting (1 m^3, 0.5 Ms, 0.5 ks).
  for spec in "bulk 0.5" "bulk 2" "gamma 0.25" "gamma 1" "hardxray 0.25" "hardxray 1"; do
    set -- $spec
    log "overhead $1 c0=$2"
    $PY experiments/comparison/run.py "$1" --reps 50 --tag fact_ --overhead "$2" \
      --jobs "$JOBS" --agents eig d-optimality systematic eig@gamma-poisson eig@gammamix-poisson
  done
}

diagnostics() {
  log "nmc agreement"; $PY experiments/comparison/nmc_agreement.py --reps 10 --jobs "$JOBS"
  log "prior only";    $PY experiments/comparison/prior_only.py
  log "significance";  $PY experiments/comparison/significance.py --prefix fact_
  log "tables";        $PY experiments/comparison/tables.py
}

timing() { log "timing"; $PY experiments/comparison/timing.py; }

case "$stage" in
  factorial) factorial ;;
  rivals) rivals ;;
  overhead) overhead ;;
  diagnostics) diagnostics ;;
  timing) timing ;;
  all) factorial; rivals; overhead; diagnostics; timing ;;
  *) echo "unknown stage $stage"; exit 1 ;;
esac
log "done: $stage"

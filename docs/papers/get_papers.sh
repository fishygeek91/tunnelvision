#!/usr/bin/env bash
# Download open-access PDFs of the core bibliography into docs/papers/pdf/ (gitignored).
# Run from anywhere: bash docs/papers/get_papers.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p pdf

get() { # get <arxiv-id> <name>
  [ -f "pdf/$2.pdf" ] && { echo "have $2"; return; }
  echo "fetching $2 ..."
  curl -sL "https://arxiv.org/pdf/$1" -o "pdf/$2.pdf"
}

get 2203.12497 layden2023_qemcmc                 # THE algorithm (Nature 619, 282)
get 2405.04247 qemcmc_followup2024               # qe-MCMC follow-up analysis
get 2602.06171 qemcmc_combinatorial2026          # MIS + warm starts + PT on IBM hw
get 2606.23350 iqemc2026_irreversible            # irreversible qe-MCMC
get 2403.01775 qdhmc2024                         # quantum dynamical HMC (differentiate)
get 2109.01690 dwave_thermal_gibbs2021           # annealer effective-temperature (N2 neighbor)
get 2303.05640 qpmcmc2023_holbrook               # quantum parallel MCMC (differentiate)

# Paywalled/journal-only (get via library/author copy):
#  - PRR 7, 013231 (2025) coarse-grained qe-MCMC — check arXiv for preprint id
#  - George & McCulloch (1993) JASA — spike-and-slab canon
echo "done. PDFs in docs/papers/pdf/"

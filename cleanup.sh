#!/usr/bin/env bash
# Run from the root of your neural-tbprof checkout, AFTER copying in the new README.md and .gitignore.
# `git rm --cached` removes files from the repository but keeps them on your disk.
set -euo pipefail

# Junk that should never have been committed
git rm -r --cached --ignore-unmatch \
  .DS_Store data/.DS_Store \
  "tbml_project(1).zip" \
  tbml.egg-info \
  tbml/__pycache__ scripts/__pycache__

# Scratch scripts: check you don't need them first, then delete for real
git rm --ignore-unmatch scripts/tmp.py scripts/fix.py

# Large outputs: remove from the repo, keep locally
git rm -r --cached --ignore-unmatch \
  results/models results/matrices \
  results/filtered_combined_snps.json results/filtered_gene_name.json
git rm -r --cached --ignore-unmatch results/essentials
git add -f results/essentials/vcf_list.txt
git rm --cached --ignore-unmatch \
  results/metrics/*_predictions.csv \
  results/metrics/*_loss_history.csv \
  results/metrics/*_per_drug_history.csv \
  results/figures/*.pdf results/figures/*.svg

git add .gitignore README.md
git status --short | head -50
echo
echo "Review the list above, then: git commit -m 'Clean repo: README, .gitignore, drop build artefacts and large outputs'"

#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <branch-name>"
    exit 1
fi

BRANCH="$1"

git worktree add -b "$BRANCH" "../$BRANCH"
cd "../$BRANCH"
ln -s ../master/venv .
source .envrc
claude --dangerously-skip-permissions
#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <branch-name> [prompt-file]"
    exit 1
fi

BRANCH="$1"
PROMPT_FILE="${2:-$BRANCH}"

git worktree add -b "$BRANCH" "../$BRANCH"
cd "../$BRANCH"
ln -s ../master/venv .
source .envrc

if [ -f "../master/$PROMPT_FILE" ]; then
    claude --dangerously-skip-permissions "$(cat "../master/$PROMPT_FILE")"
else
    claude --dangerously-skip-permissions
fi
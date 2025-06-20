#!/bin/bash
set -e

BRANCH=$(git branch --show-current)

if [ "$BRANCH" = "master" ]; then
    echo "Already on master branch"
    exit 1
fi

echo "Merging branch: $BRANCH"
cd ../master
git merge --no-edit "$BRANCH"
git worktree remove --force "../$BRANCH"
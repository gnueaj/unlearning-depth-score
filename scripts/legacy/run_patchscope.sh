#!/bin/bash
# Patchscope: Unlearning Audit
# Usage: ./run_patchscope.sh [MODE]

set -e

MODE=${1:-default}

case $MODE in
    "default"|"qa")
        echo "[Patchscope] QA probe (direct knowledge query)..."
        python -m patchscope.run --probe_type qa
        ;;
    "cloze")
        echo "[Patchscope] Cloze probe (fill-in-the-blank)..."
        python -m patchscope.run --probe_type cloze
        ;;
    "choice")
        echo "[Patchscope] Choice probe (most stable)..."
        python -m patchscope.run --probe_type choice
        ;;
    "debug")
        echo "[Patchscope] Debug mode (source=target sanity check)..."
        python -m patchscope.run --debug
        ;;
    "full")
        echo "[Patchscope] Full analysis (5 examples, all probes)..."
        python -m patchscope.run --num_examples 5 --probe_type choice
        ;;
    "quick")
        echo "[Patchscope] Quick test..."
        python -m patchscope.run --preset quick
        ;;
    *)
        echo "Patchscope: Unlearning Audit via Hidden State Patching"
        echo ""
        echo "Usage: $0 [MODE]"
        echo ""
        echo "Probe Types:"
        echo "  qa      - Direct Q&A: 'Question: X?\\nAnswer:'"
        echo "  cloze   - Fill-in-blank: 'The answer is'"
        echo "  choice  - Multiple-choice (most stable)"
        echo ""
        echo "Modes:"
        echo "  default - QA probe (same as 'qa')"
        echo "  qa      - Direct question answering"
        echo "  cloze   - Fill-in-the-blank"
        echo "  choice  - Multiple-choice probability comparison"
        echo "  debug   - Source=Target sanity check"
        echo "  full    - 5 examples with choice probe"
        echo "  quick   - Quick test (3 layers)"
        echo ""
        echo "Examples:"
        echo "  $0              # Default QA probe"
        echo "  $0 choice       # Multiple-choice (most stable)"
        echo "  $0 debug        # Sanity check"
        exit 1
        ;;
esac

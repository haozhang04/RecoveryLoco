#!/usr/bin/env bash

set -e

export LEGGEDSKILL_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)"
LEGGED_GYM_SCRIPTS_PATH="${LEGGEDSKILL_PATH}/legged_gym/legged_gym/scripts"

if [[ -n "${CONDA_PREFIX}" ]]; then
    python_exe="${CONDA_PREFIX}/bin/python"
else
    echo "[Error] No conda environment activated. Please activate the conda environment first."
    exit 1
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

_leggedskill_usage() {
    echo "Usage:"
    echo "  ./bash/leggedskill.sh -i"
    echo "  ./bash/leggedskill.sh -l"
    echo "  ./bash/leggedskill.sh -t --task go1"
    echo "  ./bash/leggedskill.sh -p --task go1"
    echo "  ./bash/leggedskill.sh -d --task go1"
    echo ""
    echo "Options:"
    echo "  -i, --install   Install rsl_rl and legged_gym in editable mode"
    echo "  -l, --list      List registered tasks"
    echo "  -t, --train     Train with legged_gym scripts/train.py, headless by default"
    echo "  -p, --play      Play with legged_gym scripts/play.py"
    echo "  -d, --draw      Draw training curves with legged_gym scripts/plot.py"
}

_leggedskill_list_tasks() {
    PYTHONPATH="${LEGGEDSKILL_PATH}/legged_gym:${PYTHONPATH}" \
        "${python_exe}" "${LEGGED_GYM_SCRIPTS_PATH}/list_tasks.py" "$@"
}

_leggedskill_install() {
    "${python_exe}" -m pip install -e "${LEGGEDSKILL_PATH}/rsl_rl" "$@"

    "${python_exe}" -m pip install -e "${LEGGEDSKILL_PATH}/legged_gym" "$@"
}

_leggedskill_python_argcomplete_wrapper() {
    local IFS=$'\013'
    local SUPPRESS_SPACE=0
    if compopt +o nospace 2> /dev/null; then
        SUPPRESS_SPACE=1
    fi

    COMPREPLY=( $(IFS="$IFS" \
                    COMP_LINE="$COMP_LINE" \
                    COMP_POINT="$COMP_POINT" \
                    COMP_TYPE="$COMP_TYPE" \
                    _ARGCOMPLETE=1 \
                    _ARGCOMPLETE_SUPPRESS_SPACE=$SUPPRESS_SPACE \
                    PYTHONPATH="${LEGGEDSKILL_PATH}/legged_gym:${PYTHONPATH}" \
                    "${python_exe}" "${LEGGED_GYM_SCRIPTS_PATH}/train.py" 8>&1 9>&2 1>/dev/null 2>/dev/null) )
}

complete -o nospace -F _leggedskill_python_argcomplete_wrapper "./bash/leggedskill.sh" 2> /dev/null || true
complete -o nospace -F _leggedskill_python_argcomplete_wrapper "./leggedskill.sh" 2> /dev/null || true

case "$1" in
    -i|--install)
        shift
        _leggedskill_install "$@"
        ;;
    -l|--list)
        shift
        _leggedskill_list_tasks "$@"
        ;;
    -t|--train)
        shift
        PYTHONPATH="${LEGGEDSKILL_PATH}/legged_gym:${PYTHONPATH}" \
            "${python_exe}" "${LEGGED_GYM_SCRIPTS_PATH}/train.py" --headless "$@"
        ;;
    -p|--play)
        shift
        PYTHONPATH="${LEGGEDSKILL_PATH}/legged_gym:${PYTHONPATH}" \
            "${python_exe}" "${LEGGED_GYM_SCRIPTS_PATH}/play.py" "$@"
        ;;
    -d|--draw)
        shift
        PYTHONPATH="${LEGGEDSKILL_PATH}/legged_gym:${PYTHONPATH}" \
            "${python_exe}" "${LEGGED_GYM_SCRIPTS_PATH}/plot.py" "$@"
        ;;
    "")
        _leggedskill_usage
        ;;
    *)
        echo "[Error] Unknown option: $1"
        _leggedskill_usage
        exit 1
        ;;
esac

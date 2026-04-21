#!/usr/bin/env bash

set -u
set -o pipefail

TIMESTAMP="$(date +%F_%H%M%S)"
BACKUP_ROOT="${HOME}/board_backup_${TIMESTAMP}"

META_DIR="${BACKUP_ROOT}/meta"
LOGS_DIR="${BACKUP_ROOT}/logs"
PKGS_DIR="${BACKUP_ROOT}/pkgs"
SERVICES_DIR="${BACKUP_ROOT}/services"
LIBS_DIR="${BACKUP_ROOT}/libs"
APP_DIR="${BACKUP_ROOT}/app"
RUNTIME_DIR="${BACKUP_ROOT}/runtime"
DEBS_DIR="${BACKUP_ROOT}/debs"
TESTS_DIR="${BACKUP_ROOT}/tests"

SCRIPT_LOG="${BACKUP_ROOT}/logs/backup_script.log"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SUDO_BIN="${SUDO_BIN:-sudo}"

mkdir -p \
  "${META_DIR}" \
  "${LOGS_DIR}" \
  "${PKGS_DIR}" \
  "${SERVICES_DIR}" \
  "${LIBS_DIR}" \
  "${APP_DIR}" \
  "${RUNTIME_DIR}" \
  "${DEBS_DIR}" \
  "${TESTS_DIR}"

touch "${SCRIPT_LOG}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${SCRIPT_LOG}" >/dev/null
}

run_and_capture() {
  local output_file="$1"
  shift

  {
    printf '### COMMAND:'
    printf ' %q' "$@"
    printf '\n'
    printf '### START: %s\n' "$(date '+%F %T')"
    "$@"
    local rc=$?
    printf '### END: %s (rc=%s)\n' "$(date '+%F %T')" "${rc}"
    return "${rc}"
  } >"${output_file}" 2>&1 || true
}

append_section() {
  local output_file="$1"
  local title="$2"
  shift 2

  {
    printf '### %s\n' "${title}"
    printf '### TIME: %s\n' "$(date '+%F %T')"
    "$@"
    local rc=$?
    printf '\n### RC=%s\n\n' "${rc}"
    return "${rc}"
  } >>"${output_file}" 2>&1 || true
}

safe_sha256sum() {
  if [ "$#" -eq 0 ]; then
    return 0
  fi
  sha256sum "$@" || true
}

collect_pkg_logs() {
  local out_file="$1"
  : >"${out_file}"

  local log_file
  for log_file in /var/log/dpkg.log*; do
    [ -e "${log_file}" ] || continue
    {
      printf '### SOURCE: %s\n' "${log_file}"
      if [[ "${log_file}" == *.gz ]]; then
        gzip -cd "${log_file}" 2>/dev/null | grep -Ei 'aidlite|aidrtcm|aidlms|py-aidlite|aistack' || true
      else
        grep -Ei 'aidlite|aidrtcm|aidlms|py-aidlite|aistack' "${log_file}" || true
      fi
      printf '\n'
    } >>"${out_file}" 2>&1 || true
  done
}

build_runtime_archive() {
  local archive_path="$1"
  local -a runtime_sources=()
  local pattern

  local -a fixed_paths=(
    /opt/aidlux/cpf
    /usr/local/lib/libaidlite.so
    /usr/local/lib/libaidlms.so
    /etc/systemd/system/aidrtcm.service
    /etc/systemd/system/aid-lms.service
    /usr/local/share/aidlite/examples
    /sdcard/Documents/AidLuxLics
  )

  for pattern in "${fixed_paths[@]}"; do
    if [ -e "${pattern}" ]; then
      runtime_sources+=("${pattern}")
    else
      log "runtime source missing: ${pattern}"
    fi
  done

  while IFS= read -r pattern; do
    runtime_sources+=("${pattern}")
  done < <(compgen -G '/usr/local/lib/libaidrtcm.so*' || true)

  while IFS= read -r pattern; do
    runtime_sources+=("${pattern}")
  done < <(compgen -G '/usr/local/lib/python3.8/dist-packages/aidlite*' || true)

  while IFS= read -r pattern; do
    runtime_sources+=("${pattern}")
  done < <(compgen -G '/usr/local/lib/python3.8/dist-packages/aidlite_gpu*' || true)

  if [ "${#runtime_sources[@]}" -eq 0 ]; then
    log "no runtime sources found, archive skipped"
    return 0
  fi

  {
    printf '### RUNTIME SOURCES\n'
    printf '%s\n' "${runtime_sources[@]}"
    printf '\n### CREATE ARCHIVE: %s\n' "${archive_path}"
  } >"${RUNTIME_DIR}/aidlux_runtime_core_sources.txt"

  "${SUDO_BIN}" tar czf "${archive_path}" -P --ignore-failed-read "${runtime_sources[@]}" >>"${SCRIPT_LOG}" 2>&1 || true
}

copy_cached_debs() {
  local -a deb_files=()
  while IFS= read -r deb; do
    deb_files+=("${deb}")
  done < <(
    find /var/cache/apt/archives -maxdepth 1 -type f -iname '*.deb' \
      | grep -Ei 'aidlite|aidrtcm|aidlms|aistack|qnn' || true
  )

  if [ "${#deb_files[@]}" -eq 0 ]; then
    log "no matching cached deb files found"
    return 0
  fi

  local deb
  for deb in "${deb_files[@]}"; do
    cp -a "${deb}" "${DEBS_DIR}/" || true
  done
}

log "backup root: ${BACKUP_ROOT}"

run_and_capture "${META_DIR}/uname.txt" uname -a
run_and_capture "${META_DIR}/os-release.txt" cat /etc/os-release
run_and_capture "${META_DIR}/hostname.txt" hostname
run_and_capture "${META_DIR}/date.txt" date

run_and_capture "${PKGS_DIR}/dpkg_list.txt" dpkg -l
run_and_capture "${PKGS_DIR}/dpkg_aid_related.txt" bash -lc "dpkg -l | grep -Ei 'aid|qnn|snpe' || true"

run_and_capture "${META_DIR}/python3_version.txt" "${PYTHON_BIN}" --version
run_and_capture "${META_DIR}/python_sysinfo.txt" "${PYTHON_BIN}" -c 'import sys; print(sys.executable); print("\n".join(sys.path))'
run_and_capture "${PKGS_DIR}/pip_list.txt" "${PYTHON_BIN}" -m pip list
run_and_capture "${PKGS_DIR}/pip_show_aidlite.txt" "${PYTHON_BIN}" -m pip show aidlite pyaidlite

run_and_capture "${LIBS_DIR}/ldconfig_aid.txt" bash -lc "ldconfig -p | grep -E 'aidlite|aidlms|aidrtcm' || true"

{
  printf '### ls -l /usr/local/lib/libaid*\n'
  ls -l /usr/local/lib/libaid* 2>&1 || true
  printf '\n### sha256sum /usr/local/lib/libaid*\n'
  shopt -s nullglob
  aid_libs=(/usr/local/lib/libaid*)
  safe_sha256sum "${aid_libs[@]}"
} >"${LIBS_DIR}/libaid_files.txt" 2>&1

run_and_capture "${SERVICES_DIR}/aidrtcm_systemctl_cat.txt" "${SUDO_BIN}" systemctl cat aidrtcm
run_and_capture "${SERVICES_DIR}/aidrtcm_systemctl_status.txt" "${SUDO_BIN}" systemctl status --no-pager aidrtcm
run_and_capture "${SERVICES_DIR}/aid-lms_systemctl_cat.txt" "${SUDO_BIN}" systemctl cat aid-lms
run_and_capture "${SERVICES_DIR}/aid-lms_systemctl_status.txt" "${SUDO_BIN}" systemctl status --no-pager aid-lms

collect_pkg_logs "${LOGS_DIR}/aid_pkg_install_history.txt"

build_runtime_archive "${RUNTIME_DIR}/aidlux_runtime_core.tar.gz"

copy_cached_debs
run_and_capture "${DEBS_DIR}/local_candidates.txt" bash -lc "find /tmp /home/aidlux -type f 2>/dev/null | grep -Ei 'aidlite|aidrtcm|aidlms|aistack|qnn' | head -n 300 || true"

run_and_capture "${TESTS_DIR}/qnn236_example.txt" bash -lc "cd /usr/local/share/aidlite/examples/aidlite_qnn236/python && ${SUDO_BIN} -E ${PYTHON_BIN} qnn_yolov5_multi.py 3"
run_and_capture "${TESTS_DIR}/aidlite_import_test.txt" "${PYTHON_BIN}" -c 'import aidlite; print(aidlite.get_library_version()); print(aidlite.get_py_library_version())'

append_section "${META_DIR}/backup_summary.txt" "backup_root" printf '%s\n' "${BACKUP_ROOT}"
append_section "${META_DIR}/backup_summary.txt" "generated_at" date

log "backup completed"
printf 'Backup completed: %s\n' "${BACKUP_ROOT}"

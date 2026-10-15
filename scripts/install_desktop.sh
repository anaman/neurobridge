#!/bin/sh
# Install (or remove) the NeuroBridge desktop entries so it appears in the
# system menu like any other application.
#
#   ./scripts/install_desktop.sh              # per-user (~/.local/share), no root
#   ./scripts/install_desktop.sh --system     # all users (/usr/share), needs root
#   ./scripts/install_desktop.sh --uninstall  # remove the per-user entries
#
# Idempotent: re-running simply rewrites the same files, so it is also the way to
# refresh the entries after moving the project directory.
set -e

NB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="user"
ACTION="install"

for arg in "$@"; do
  case "$arg" in
    --system) MODE="system" ;;
    --uninstall|--remove) ACTION="uninstall" ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ "$MODE" = "system" ]; then
  APPS="/usr/share/applications"
  ICONS="/usr/share/icons/hicolor"
  SUDO="sudo"
else
  APPS="$HOME/.local/share/applications"
  ICONS="$HOME/.local/share/icons/hicolor"
  SUDO=""
fi

echo "NeuroBridge desktop install"
echo "  project : $NB_ROOT"
echo "  entries : $APPS"
echo "  icons   : $ICONS"
echo

# ---------------------------------------------------------------- icons
install_icons() {
  if [ ! -f "$NB_ROOT/assets/neurobridge-256.png" ]; then
    echo "  generating icons ..."
    "$NB_ROOT/.venv/bin/python" "$NB_ROOT/scripts/make_icon.py" >/dev/null
  fi
  for size in 256 128 64 48; do
    dir="$ICONS/${size}x${size}/apps"
    $SUDO mkdir -p "$dir"
    $SUDO cp -f "$NB_ROOT/assets/neurobridge-$size.png" "$dir/neurobridge.png"
  done
  echo "  installed icons: 256/128/64/48 px"
}

# ---------------------------------------------------------------- entries
write_entry() {
  # $1 = filename, $2 = Name, $3 = Comment, $4 = Exec, $5 = Terminal, $6 = Categories
  # $7 = GenericName (optional)
  # exactly one *main* category, or the menu shows the app several times
  # (desktop-file-validate warns about that, and it looks like a duplicate app)
  target="$APPS/$1"
  generic="${7:-EEG training session}"
  $SUDO mkdir -p "$APPS"
  $SUDO tee "$target" >/dev/null <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=$2
GenericName=$generic
Comment=$3
Exec=$4
Icon=neurobridge
Terminal=$5
Categories=$6
Keywords=EEG;brain;neurofeedback;BCI;NingNao;TH21A;ChatGPT;agent;
StartupNotify=true
StartupWMClass=Neurobridge
EOF
  $SUDO chmod 0644 "$target"
  echo "  wrote $target"
}

install_entries() {
  install_icons
  write_entry "neurobridge.desktop" \
    "NeuroBridge" \
    "Run EEG training sessions, analyse them, and start the agent bridge" \
    "$NB_ROOT/bin/nb-gui" "false" "Science;"
  write_entry "neurobridge-bridge.desktop" \
    "NeuroBridge Agent Bridge" \
    "Serve decoded EEG state to an AI agent on 127.0.0.1:8731" \
    "$NB_ROOT/bin/nb-serve" "true" "Science;"
  write_entry "neurobridge-monitor.desktop" \
    "NeuroBridge Monitor" \
    "Live EEG trace with CSV recording, and replay of recorded sessions" \
    "$NB_ROOT/bin/nb-monitor" "false" "Science;" \
    "EEG monitor and recorder"
}

# ---------------------------------------------------------------- removal
uninstall() {
  for f in neurobridge.desktop neurobridge-bridge.desktop neurobridge-monitor.desktop; do
    if [ -f "$APPS/$f" ]; then
      $SUDO rm -f "$APPS/$f"
      echo "  removed $APPS/$f"
    fi
  done
  for size in 256 128 64 48; do
    $SUDO rm -f "$ICONS/${size}x${size}/apps/neurobridge.png"
  done
  echo "  removed icons"
}

# ---------------------------------------------------------------- caches
refresh_caches() {
  command -v update-desktop-database >/dev/null 2>&1 && \
    $SUDO update-desktop-database "$APPS" 2>/dev/null || true
  command -v gtk-update-icon-cache >/dev/null 2>&1 && \
    $SUDO gtk-update-icon-cache -f -t "$ICONS" 2>/dev/null || true
}

# ---------------------------------------------------------------- validate
validate() {
  if command -v desktop-file-validate >/dev/null 2>&1; then
    for f in neurobridge.desktop neurobridge-bridge.desktop neurobridge-monitor.desktop; do
      if [ -f "$APPS/$f" ]; then
        if desktop-file-validate "$APPS/$f"; then
          echo "  validated $f"
        else
          echo "  WARNING: $f has validation issues (see above)" >&2
        fi
      fi
    done
  else
    echo "  (desktop-file-validate not installed; skipping validation)"
  fi
}

case "$ACTION" in
  install)
    install_entries
    refresh_caches
    validate
    echo
    echo "Done. Look for 'NeuroBridge' in the menu"
    echo "  (MATE: Applications > Science, or just type 'NeuroBridge' in the menu search)."
    echo "If it does not appear immediately, the panel menu may need a refresh:"
    echo "  mate-panel --replace &   # or log out and back in"
    echo
    echo "Command line equivalents:"
    echo "  $NB_ROOT/bin/nb-gui        # this menu entry"
    echo "  $NB_ROOT/bin/nb-monitor    # live monitor / CSV recorder / replay"
    echo "  $NB_ROOT/bin/nb --help     # everything else"
    ;;
  uninstall)
    uninstall
    refresh_caches
    echo "Done."
    ;;
esac

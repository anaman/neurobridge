#!/bin/sh
# Guided first hardware session for NeuroBridge.
#
#   ./scripts/first_session.sh
#
# Runs the checks in the order that catches problems before they waste a
# sitting: environment -> headband -> electrode contact -> baselines ->
# the alpha-reactivity verdict, which is the gate on trusting everything else.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
NB="$HERE/../bin/nb"

echo "=============================================================="
echo " NeuroBridge - guided first session"
echo "=============================================================="
echo
echo "[1/5] environment and headband"
echo "      (this scans BLE for 10 s - the band must be powered on)"
echo
"$NB" doctor --scan || true

echo
echo "[2/5] electrode preparation"
cat <<'EOF'
      - wipe each electrode with an alcohol pad or saline, let it dry
      - wipe your forehead and the bridge of your nose
      - seat the band: switch on the nose bridge, band just above the
        eyebrows, no hair trapped under any electrode
      - sit comfortably, jaw relaxed, screen at eye level
EOF
printf "\n      press Enter when the band is on and comfortable > "
read -r _ || true

echo
echo "[3/5] recording T002 (4 x 60 s eyes open) and T003 (3 x 20 s open/closed)"
echo "      ~11 minutes. Follow the spoken cues."
echo
"$NB" run T002 T003 -y

echo
echo "[4/5] evaluation"
"$NB" eval T003 --session latest || true

echo
echo "[5/5] what to look for"
cat <<'EOF'
      PASS on T003 means alpha rose when you closed your eyes - that is the
      cheapest proof the channel is carrying real cortical signal rather than
      electrode artefact. If it failed, do NOT move on to other tests: clean the
      electrodes with saline, reseat the band so no hair is trapped, and re-run.
      See docs/HARDWARE.md for the full troubleshooting order.

      Next sitting:  ./bin/nb run T001 T007 T004 -y
      Then plan the rest with:  ./bin/nb plan --remaining
EOF

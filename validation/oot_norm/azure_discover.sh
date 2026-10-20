#!/usr/bin/env bash
# Read-only Azure discovery. Authenticate yourself before running; no secrets.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo "Usage: bash azure_discover.sh SUBSCRIPTION_ID REGION OUTPUT_DIR" >&2
  exit 2
fi
subscription=$1
region=$2
out=$3
command -v az >/dev/null
mkdir -p "$out"
chmod 700 "$out"
az account show --subscription "$subscription" \
  --query '{id:id,name:name,state:state}' -o json > "$out/account.json"
az vm list-skus --subscription "$subscription" --location "$region" \
  --resource-type virtualMachines --all \
  --query "[?capabilities[?name=='GPUs' && value!='0']].{name:name,family:family,capabilities:capabilities,restrictions:restrictions,locations:locations}" \
  -o json > "$out/gpu-skus.json"
az vm list-usage --subscription "$subscription" --location "$region" \
  -o json > "$out/quota.json"
echo "Read-only discovery saved to $out. No resources created."
echo "Still verify credits, current price, SKU compatibility and allocation capacity."
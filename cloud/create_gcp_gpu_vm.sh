#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:?Set PROJECT to your Google Cloud project id}"
ZONE="${ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-seven-cartpole-gpu}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-16}"
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
GPU_COUNT="${GPU_COUNT:-1}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-300GB}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
BUCKET="${BUCKET:?Set BUCKET to a globally unique Cloud Storage bucket name}"
RUN_NAME="${RUN_NAME:-ppo_7link_gpu}"
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
N_ENVS="${N_ENVS:-16}"

if ! command -v gcloud >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Error: gcloud was not found.

Run this script from Google Cloud Shell:
  https://console.cloud.google.com/

Click the Cloud Shell terminal icon, then run:
  git clone https://github.com/egao0125/7-Cartpole-PPO.git
  cd 7-Cartpole-PPO

Or install the Google Cloud CLI locally if you intentionally want to manage GCP from this machine.
EOF
  exit 127
fi

gcloud config set project "$PROJECT"

if ! gcloud compute images describe-from-family "$IMAGE_FAMILY" \
  --project="$IMAGE_PROJECT" >/dev/null 2>&1; then
  cat >&2 <<EOF
Error: image family '${IMAGE_FAMILY}' was not found in project '${IMAGE_PROJECT}'.

List available PyTorch GPU image families with:
  gcloud compute images list \\
    --project=${IMAGE_PROJECT} \\
    --no-standard-images \\
    --filter="family~'^pytorch.*cu'" \\
    --format="table(family,name)"

Then rerun with:
  IMAGE_FAMILY=<family-from-list> bash cloud/create_gcp_gpu_vm.sh
EOF
  exit 1
fi

if ! gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${BUCKET}" --location="${ZONE%-*}"
fi

STARTUP_SCRIPT=$(mktemp)
cat > "$STARTUP_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export RUN_NAME="${RUN_NAME}"
export TOTAL_STEPS="${TOTAL_STEPS}"
export N_ENVS="${N_ENVS}"
export GCS_PATH="gs://${BUCKET}/seven-cartpole/runs"
curl -fsSL https://raw.githubusercontent.com/egao0125/7-Cartpole-PPO/main/cloud/gcp_train_startup.sh -o /tmp/gcp_train_startup.sh
bash /tmp/gcp_train_startup.sh
EOF

gcloud compute instances create "$VM_NAME" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --maintenance-policy=TERMINATE \
  --boot-disk-size="$BOOT_DISK_SIZE" \
  --boot-disk-type=pd-balanced \
  --metadata-from-file=startup-script="$STARTUP_SCRIPT" \
  --metadata=install-nvidia-driver=True

rm -f "$STARTUP_SCRIPT"

echo "VM created: ${VM_NAME}"
echo "SSH: gcloud compute ssh ${VM_NAME} --zone=${ZONE}"
echo "Logs: gcloud compute ssh ${VM_NAME} --zone=${ZONE} --command='tail -f /opt/7-Cartpole-PPO/runs/${RUN_NAME}/train.log'"
echo "Videos/checkpoints will sync to: gs://${BUCKET}/seven-cartpole/runs/${RUN_NAME}/"

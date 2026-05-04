# Cosmos Curator - NVCF Guide

- [Cosmos Curator - NVCF Guide](#cosmos-curator---nvcf-guide)
    - [Store Configuration Settings](#store-configuration-settings)
    - [Build \& Upload Helm Chart](#build--upload-helm-chart)
    - [Upload Container Image](#upload-container-image)
    - [Upload Model Weights](#upload-model-weights)
    - [Create, Deploy, Invoke Function](#create-deploy-invoke-function)
      - [Create a Function](#create-a-function)
      - [Deploy the Function](#deploy-the-function)
      - [Invoke the Function](#invoke-the-function)

[NVIDIA Cloud Functions (NVCF)](https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/overview.html)
is a serverless API to deploy & manage AI workloads on GPUs.
Cosmos Curator can be deployed on NVCF for a semi/full-managed experience.

As mentioned in [End User Guide](./end-user-guide.md#launch-pipelines-on-nvidia-dgx-cloud),
please reach out to NVIDIA Cosmos Curator team to get help for onboarding.

**Note this guide does not cover the complete process for NVCF deployment,** but assumes onboarding and initial setup have been completed.

### Store Configuration Settings

```bash
# Set the NVCF Org ID and API key
export NGC_NVCF_ORG=<your_org_id>
export NGC_NVCF_API_KEY=<your_api_key>
# If you NVCF Org has a hierarchy of team, set the team name - this is rare
export NGC_NVCF_TEAM=<your_team_name>

# Set the NVCF cluster information
export NVCF_BACKEND=<your_backend_cluster_name>
export NVCF_GPU_TYPE=<your_cluster_gpu_type>
export NVCF_INSTANCE_TYPE=<your_instance_type>

# Save above configuration settings to `~/.config/cosmos_curator/client.json`
cosmos-curator nvcf config set
```

### Build & Upload Helm Chart

Follow the instructions in this [README](../../charts/cosmos-curator/README.md).

### Upload Container Image

Modify `~/.config/cosmos_curator/templates/image/image_upload.json` to fill in the image name and tag; for example,

```json
{
    "image": "cosmos-curator",
    "tag": "1.0.0",
    "definition": {
    }
}
```

Re-tag the built image with an `nvcr.io` prefix and upload it.

```bash
docker tag cosmos-curator:1.0.0 nvcr.io/$NGC_NVCF_ORG/cosmos-curator:1.0.0
cosmos-curator nvcf image upload-image --data-file ~/.config/cosmos_curator/templates/image/image_upload.json
```

If you are on a specifc team in your Org, you will probably want to use the private registry at team level.
So the `image` entry in the `json` should be

```json
    "image": "<team-name>/cosmos-curator",
```

And the image should be re-tagged as

```bash
docker tag cosmos-curator:1.0.0 nvcr.io/$NGC_NVCF_ORG/$NGC_NVCF_TEAM/cosmos-curator:1.0.0
```

### Upload Model Weights

```bash
# download models from hugging face to local
cosmos-curator local launch --image-name cosmos-curator --image-tag 1.0.0 --curator-path . -- pixi run --as-is python3 -m cosmos_curator.core.managers.model_cli download

# sync to NVCF
cosmos-curator nvcf model sync-models \
    --data-file cosmos_curator/configs/all_models.json \
    --download-dir "${COSMOS_CURATOR_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curator_local_workspace/models/"
```

Again if you are on a specifc team in your Org, it's likely models should be uploaded to your team's private registry;
But for models, it is handled automatically by the CLI as long as you have `NGC_NVCF_TEAM` set as mentioned [above](#store-configuration-settings).

### Create, Deploy, Invoke Function 

#### Create a Function

Modify `~/.config/cosmos_curator/templates/function/create_curator_helm.json` to fill in the `crt` and `key` for your Thanos-like instance.
This is for the Prometheus agent to remote-write the metrics.
- If you don't have a Thanos-like instance yet,
  - you can remove `byo-metrics-receiver-client-crt` and `byo-metrics-receiver-client-key` from the `secrets` list;
  - then disable `metrics` when deploying the function, see details in next step.

The telemetries section can be uncommented and filled out if the appropriate endpoints are available in your account.
See the [NVCF External Observability guide](https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/observability.html#external-observability)
for additional information.

```bash
cosmos-curator nvcf function create-function \
    --name "${USER}-cosmos-curator" \
    --health-ep /api/local_raylet_healthz --health-port 52365 \
    --helm-chart https://helm.ngc.nvidia.com/${NGC_NVCF_ORG}/charts/cosmos-curator-2.1.1.tgz \
    --data-file ~/.config/cosmos_curator/templates/function/create_curator_helm.json
```

#### Deploy the Function

Modify `~/.config/cosmos_curator/templates/function/deploy_curator_helm.json` to fill in
- image tag in `configuration.image.tag`
- Org ID in the `nvcf.io/<ORG-ID>/...` image string in `configuration.image.repository`
- GPU count (per node) in `configuration.resources.requests` and `configuration.resources.limits`
- Thanos remote-write receiver URL in `configuration.metrics.remoteWrite.endpoint`
  - As mentioned above, if you don't have an endpoint, set `configuration.metrics.enabled` to `false`

```bash
# --instance-count controls number of nodes
cosmos-curator nvcf function deploy-function \
    --max-concurrency 2 \
    --instance-count 2 \
    --data-file ~/.config/cosmos_curator/templates/function/deploy_curator_helm.json
```

The deployment can take up to 15 minutes, you can run the following command to check the status:

```bash
cosmos-curator nvcf function get-deployment-detail
```

If you are deploying a brand new container image, the deployment may fail due to timeout when pulling the new image.
In that case, a re-deployment should just work:

```bash
# Un-deploy the function
cosmos-curator nvcf function undeploy-function

# Deploy it again
cosmos-curator nvcf function deploy-function \
    --max-concurrency 2 \
    --instance-count 2 \
    --data-file ~/.config/cosmos_curator/templates/function/deploy_curator_helm.json
```

#### Invoke the Function

Modify `~/.config/cosmos_curator/templates/function/invoke_video_split.json` to fill in input & output paths.

```bash
cosmos-curator nvcf function invoke-function \
    --data-file ~/.config/cosmos_curator/templates/function/invoke_video_split.json \
    --s3-config-file ~/.aws/credentials
```

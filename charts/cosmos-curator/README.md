# Cosmos Curator Helm Chart

This Helm chart deploys Cosmos Curator. The chart supports two deployment modes:

- **NVCF Deployment**: Managed deployment through NVIDIA Cloud Functions
- **Native Kubernetes Deployment**: Direct deployment to your own Kubernetes cluster

## Prerequisites

### Common Requirements
* [Helm binary](https://helm.sh/docs/intro/install/) (v3.0+)
* [OpenTelemetry Helm repository](https://opentelemetry.io/) - Can be disabled at deploy time if not using metrics

### NVCF-Specific Requirements
* [NGC CLI and NGC profile](https://org.ngc.nvidia.com/setup/installers/cli) - Used to push the packaged chart
* Access to [NGC registry](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html) - Used for hosting the container image
* Environment Variable `NGC_NVCF_ORG` - Your NVCF organization ID - used to determine URLs

### Native Kubernetes Requirements
* Kubernetes cluster (v1.28+) with GPU-enabled nodes
* NVIDIA GPUs accessible from containers (drivers, container runtime, device plugin configured)
* `kubectl` access with appropriate permissions
* Cosmos Curator container built and published to an accessible repo
* S3-compatible storage credentials (for video data)

## Common Setup

### Add OpenTelemetry Repository
```bash
helm repo add open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts
helm repo update
```

### Set Chart Version
The latest version is `2.2.2`. Set this as an environment variable:
```bash
export CHART_VERSION=2.2.2
```

## Deployment

### NVCF Deployment

#### Initial Setup

1. **Configure NGC CLI** (one-time):
```bash
ngc config set
# Provide:
# - Your NGC API key
# - "ascii" for output format
# - Your NVCF ORG ID
# - "no-team" (unless explicitly assigned to a team)
# - "no-ace"
```

2. **Create Chart Metadata** (for new orgs):
```bash
ngc registry chart create --short-desc "Chart for NVCF function for cosmos curator" ${NGC_NVCF_ORG}/cosmos-curator
```

#### Package and Publish

1. **Build dependencies and package**:
```bash
helm dep build charts/cosmos-curator/
helm package charts/cosmos-curator --version ${CHART_VERSION}
```

2. **Push to NGC registry**:
```bash
ngc registry chart push ${NGC_NVCF_ORG}/cosmos-curator:${CHART_VERSION}
```

3. **Deploy using NVCF CLI** (see [NVCF documentation](docs/client/END_USER_GUIDE.md)):
```bash
cosmos-curator nvcf --help
```

#### Remove Chart (If cleanup is required)
```bash
ngc registry chart remove ${NGC_NVCF_ORG}/cosmos-curator:${CHART_VERSION}
```

### Native Kubernetes Deployment
All paths are relative to the charts working directory
#### Prerequisites Setup

**Provide needed customization information:**
1. **Modify `values-standalone.yaml`** with your configuration (Review the file for details of expected values and examples)
2. **For sensitive values** (API keys, credentials), you have two options:
   - Fill them directly in `values-standalone.yaml`, OR
   - pass via `--set` flags at install time

> **Note:** `values.yaml` contains the full set of parameters but should not typically need modification. The minimal set of changes is captured in the values-standalone.yaml

#### Install Chart

Using the standalone values file with secrets provided via `--set`:
```bash
helm upgrade cosmos-curator --namespace cosmos-curator --create-namespace --install . -f values.yaml -f values-standalone.yaml --set imagePullSecret.dockerConfigJson.password=${API_KEY} --set ngcCatalog.secret.key=${MODEL_KEY} --set replicas=1
```

#### Access the Service
When the deployment is ready (view with kubectl get pods), jobs can be submitted. The following example leverages the REST API, but it is also possible to use Ray (or its API) directly.
Port-forward for local access and invoke:
```bash
kubectl -n cosmos-curator port-forward svc/cosmos-curator 8000:8000 > /dev/null &
# Copy an example invoke from  ../../examples/nvcf/function, and add "s3_config":"<base64 encoded credentials>" as standalone-invoke.json
REQUEST_ID=$(uuidgen)
curl -sX POST localhost:8000/v1/run_pipeline \
  -H "NVCF-REQID: ${REQUEST_ID}" \
  -d @standalone-invoke.json

# To check progress (percentage)
curl -s "localhost:8000/v1/progress?request_id=${REQUEST_ID}"

# Or get full logs
curl -s "localhost:8000/v1/logs?request_id=${REQUEST_ID}"
```

#### Monitor Deployment
```bash
# Check pod status
kubectl get pods -l app=cosmos-curator

# View logs
kubectl logs -l app=cosmos-curator --tail=100

# Check Ray dashboard
kubectl port-forward svc/cosmos-curator 8265:8265
```

#### Uninstalling
```bash
helm uninstall --namespace cosmos-curator cosmos-curator
```



## Configuration

### Configuration Options

Refer to the  `values.yaml` for a complete list and default values.

### Persistent Storage

The `/config` directory (used for model caching via `modelCacheDir: "/config/models"`) as well as Ray spill can be configured for various persistence options.
#### Custom storage class
Preferred option - should be high bandwidth storage class, at least 500GB of capacity.
```yaml
persistence:
  enabled: true
  size: 500Gi
  storageClass: "fast-ssd"
```
#### Host path
Likely only makes sense on a local setup with dedicated NVME/SSD disks not available via a storage class
```yaml
persistence:
  enabled: false
scratchDir: "/mnt/models"  # Must exist on the node
```
#### emptyDir
This is the simplest option, fine for nodes with adequate free space available to the container filesystem
```yaml
persistence:
  enabled: false
scratchDir: ""
```

#### PVC lifecycle

By default, PVCs are automatically deleted when the StatefulSet is removed (`helm uninstall`), preventing orphaned volumes and storage charges. This is controlled by:

```yaml
persistence:
  retentionPolicy:
    whenDeleted: Delete
    whenScaled: Retain
```

To keep PVCs for manual cleanup or re-use between deployments, set `whenDeleted: Retain`.



### Metrics and Monitoring

The chart supports two approaches for exposing Prometheus metrics:

#### OpenTelemetry Collector

The default OTEL collector scrapes metrics and remote writes to an external Prometheus endpoint:

```yaml
metrics:
  enabled: true  # Enables both metrics endpoint and OTEL collector
  remoteWrite:
    endpoint: "https://your-prometheus/api/v1/receive"
    certPath: "/etc/certs/tls.crt"
    keyPath: "/etc/certs/tls.key"
```

Or an OTLP endpoint - by default, it will use the [NVCF BYOO endpoint](https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/observability.html#appendix-c-adding-custom-application-metrics-logs-traces).
```yaml
metrics:
  enabled: true
  remoteWrite:
    enabled: false
  otlp:
    enabled: true
    endpoint: "${env:OTEL_EXPORTER_OTLP_METRICS_ENDPOINT}"
```
#### ServiceMonitor

For in-cluster Prometheus configured to monitor ServiceMonitor CRs

```yaml
metrics:
  enabled: false  # Disable OTEL collector
  serviceMonitor:
    enabled: true
    # Labels to match your Prometheus selector
    labels:
      prometheus: kube-prometheus
    interval: 30s
```


**Common Metric Filtering:**

Both approaches use the same metric filtering rules defined in `metrics.prometheus.scrapeConfigs[0].metric_relabel_configs`. By default, high-cardinality Ray metrics (tasks, actors, object store details) are dropped. To customize:

```yaml
metrics:
  prometheus:
    scrapeConfigs:
      - job_name: "ray-service-metrics"
        metric_relabel_configs:
          - source_labels: [__name__]
            regex: custom_metric_pattern
            action: drop
```

The ServiceMonitor automatically converts these OTEL-style configs (snake_case) to Prometheus Operator format (camelCase).

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
* Kubernetes cluster (v1.32+) with GPU-enabled nodes
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
The latest version is `2.5.0`. Set this as an environment variable:
```bash
export CHART_VERSION=2.5.0
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

3. **Deploy using NVCF CLI** (see [NVCF documentation](docs/client/end-user-guide.md)):
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

# Check Ray dashboard (rayAuth is on by default — retrieve the token first)
kubectl -n cosmos-curator get secret ray-auth-token -o jsonpath='{.data.token}' | base64 -d
kubectl port-forward svc/cosmos-curator 8265:8265
# Then open http://localhost:8265 and enter the token when prompted
```

#### Uninstalling
```bash
helm uninstall --namespace cosmos-curator cosmos-curator
```



## Configuration

### Configuration Options

Refer to the  `values.yaml` for a complete list and default values.

`extraVolumes` and `extraVolumeMounts` add pod volumes and mounts to the main
curator container. When `logging.otlp.enabled` is true, the same mounts are also
added to the OTLP log collector sidecar so explicit `otlp.tls.*Path` settings can
be backed by operator-managed volumes such as the cert-manager CSI driver.

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



### Observability

The chart supports metrics, traces, and logs through several independent export
paths:

| Config | Runs where | Signal | What it does |
| --- | --- | --- | --- |
| `metrics.remoteWrite.*` | Chart-managed collector deployment | Metrics | Scrapes curator/Ray Prometheus metrics and exports them to Prometheus remote write. |
| `metrics.otlp.*` | Chart-managed collector deployment | Metrics | Scrapes the same curator/Ray Prometheus metrics and exports them to OTLP HTTP metrics. This can be enabled with or without remote write. |
| `metrics.serviceMonitor.*` | In-cluster Prometheus Operator | Metrics | Creates a ServiceMonitor so an existing in-cluster Prometheus can scrape curator/Ray metrics directly. This does not use the chart-managed collector. |
| `metrics.otlpPush.*` | Curator container | Metrics | Pushes in-pipeline metrics directly to OTLP from the curator process. This does not use the chart-managed collector. |
| `tracing.otlp.*` | Curator container | Traces | Enables direct OTLP trace export from the curator process. |
| `logging.otlp.*` | Per-pod collector sidecar | Logs | Tails curator/Ray log files and exports them to OTLP logs. |

`otlp.*` is the shared OTLP transport for logs, traces, and in-process metrics
push. It is also used by the chart-managed collector's OTLP metrics exporter
unless `metrics.otlp.endpoint` is set. Use `metrics.otlp.*` only when that
collector metrics exporter needs a different OTLP endpoint or client TLS
settings from the other OTLP signals.

Do not include `/v1/logs`, `/v1/metrics`, or `/v1/traces` in the shared
`otlp.endpoint`; signal-specific exporters add the path. Only
`metrics.otlp.endpoint` expects the full metrics endpoint when it is used.
When `otlp.endpoint` is set, `${env:...}` values in `metrics.otlp.endpoint` are
treated as collector-local defaults and the shared endpoint takes precedence.

`otlp.tls.secret.*` mounts a direct Kubernetes Secret for pod-local OTLP
consumers: the log sidecar, in-process traces, and in-process metrics push. You
can also set explicit `otlp.tls.certPath`, `keyPath`, and `caPath` values and
mount those files with `extraVolumes`/`extraVolumeMounts`; those mounts are
shared with the log sidecar when log export is enabled. The chart-managed metrics
collector runs as the `opentelemetry-collector` subchart; to use a direct
Kubernetes Secret there, configure `opentelemetry-collector.extraVolumes` and
`opentelemetry-collector.extraVolumeMounts` to mount the files at the paths used
by `otlp.tls.*`, or use `metrics.extractNVCFSecrets` for NVCF-style collector
secrets.

Do not use collector-only `${env:...}` substitution in top-level `otlp.endpoint`
when `metrics.otlpPush.enabled` or `tracing.otlp.enabled` is true. That endpoint
is also exported to the Python process as `OTEL_EXPORTER_OTLP_ENDPOINT`, where
`${env:...}` is not expanded.

Remote write and collector OTLP can be enabled together. The collector scrapes
the curator/Ray Prometheus endpoint once and fans out to both exporters. These
exporters intentionally keep separate certificate paths so a remote-write
backend and an OTLP backend can require different client certificates.

Upgrading from chart 2.3: existing values that set `metrics.otlp.enabled: true`
can keep using the legacy `metrics.otlp.endpoint` default,
`${env:OTEL_EXPORTER_OTLP_METRICS_ENDPOINT}`, for the chart-managed collector's
OTLP metrics exporter. New shared logs/traces/metrics configurations should use
top-level `otlp.endpoint`.

For NVCF BYOO OTLP metrics only, the previous recipe still works:

```yaml
metrics:
  enabled: true
  remoteWrite:
    enabled: false
  otlp:
    enabled: true
    endpoint: "${env:OTEL_EXPORTER_OTLP_METRICS_ENDPOINT}"
```

See the [NVIDIA Cloud Functions observability documentation](https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/observability.html#appendix-c-adding-custom-application-metrics-logs-traces) for the BYOO metrics endpoint setup.

When tracing and JSON logging are both enabled, log lines emitted inside a span
include `trace_id` and `span_id`; see
**[Linking log lines to traces](../../docs/curator/guides/observability.md#linking-log-lines-to-traces)**.

#### Logging

The chart exposes two structured logging knobs, written to the curator ConfigMap
and propagated to head and worker pods through the statefulset's `envFrom`:

```yaml
logging:
  format: json          # "text" (default) | "json"
  rayBackendJson: false # opt-in JSON for Ray's C++ backend/system logs
```

- `format: json` emits structured JSON application logs (Ray driver/workers, the
  launcher scripts, and the pre-`ray.init` fallback) using a single flat,
  Ray-aligned schema. `format: text` (default) leaves output human-readable and
  unchanged.
- `rayBackendJson: true` additionally emits Ray's C++ backend logs (raylet, GCS,
  ...) as JSON by setting `RAY_BACKEND_LOG_JSON=1`. It is independent of `format`
  and never auto-set by curator/xenna code.
- `logging.otlp.enabled: true` defaults both `PYTHON_LOG_FORMAT=json` and
  `RAY_BACKEND_LOG_JSON=1` through `logging.otlp.forceJsonFormat: true`; set it
  to `false` when you want OTLP shipping but human-readable container streams.

For the full field schema, Elasticsearch / log-shipper guidance, the
`log_to_driver` trade-off, and non-Helm (local Docker / Slurm) usage, see the
**[Observability Guide](../../docs/curator/guides/observability.md#structured-logging)**.

The chart can also export Ray log files to a generic OTLP logs endpoint using
an OpenTelemetry Collector sidecar on each StatefulSet pod. This does not change
stdout/stderr logging, so platform log indexing continues to work.

The sidecar tails `/tmp/ray/session_*/logs/*.log`, `*.out`, `*.err`, and
`/tmp/curator/stdout-stderr.log` from the same emptyDir mounted by the curator
container. Ray `job-driver-*` files are collected by default because they are the
durable source for submitted-driver output. Set
`logging.otlp.collectJobDriverLogs: false` only when another source, such as the
stdout tee, is known to export the same lines. `monitor.log` is excluded by
default because Ray's autoscaler monitor
emits a verbose polling loop; include it temporarily only while debugging Ray
autoscaler or cluster status issues. JSON-looking lines are parsed as JSON, Ray
Python text logs have their timestamp and level mapped to OTLP log fields, and
retain fields such as `code.filepath` and `lineno`. Other non-JSON lines are
preserved as text bodies. By default the sidecar adds Kubernetes pod metadata
from the downward API: Kubernetes namespace, pod name, pod UID, pod IP, node
name, service account, StatefulSet name, and StatefulSet pod index.

When OTLP logging is enabled, the pod termination grace period defaults to 60
seconds and the sidecar uses a 15 second native Kubernetes `preStop.sleep` hook
before receiving SIGTERM. This gives the collector time to continue reading final
container output and flush it during shutdown. No readiness probe is added for
the sidecar, so logging health does not gate pipeline traffic.

The filelog receiver stores checkpoints under
`/var/lib/otelcol/file_storage` on a dedicated `emptyDir`, so restarted
collectors resume from the last checkpoint within the same pod. The exporter
sending queue is intentionally memory-only so a telemetry backend outage cannot
grow the sidecar `emptyDir` until kubelet evicts the curator pod.

Node labels such as GPU product are not available from the downward API unless the
platform also copies them onto pod labels or annotations. To include those fields,
either add static `logging.otlp.extraResourceAttributes` or use `logging.otlp.extraEnv`
with a pod label/annotation fieldRef and set an attribute value such as
`${env:GPU_PRODUCT}`.

For direct OTLP ingestion, the receiving backend decides which resource
attributes become index labels and which remain structured metadata. Keep any
additional indexed attributes low-cardinality.

By default, the OTLP log sidecar also copies the chart's effective metrics external
labels into log resource attributes. In NVCF deployments, the CLI currently adds
`function_id`, `version_id`, `gpu`, `org`, and when present `backend`, `regions`,
and `availability_zones` under `metrics.extraExternalLabels`, so those same
values are available on OTLP log records. Set
`logging.otlp.includeMetricsExternalLabels: false` to decouple logs from metric
labels.

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


#### Common metric filtering

The chart collector and ServiceMonitor paths use the same metric filtering rules
defined in `metrics.prometheus.scrapeConfigs[0].metric_relabel_configs`. By
default, high-cardinality Ray metrics (tasks, actors, object store details) are
dropped. To customize:

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

## Network Exposure

Without NetworkPolicy, any pod in the cluster can reach the ports the curator StatefulSet defines. Namespace boundaries do not restrict pod-to-pod traffic in vanilla Kubernetes — only a NetworkPolicy (or a CNI-level default-deny) enforces that.

| Port | Value | Name | Risk if unprotected |
|------|-------|------|---------------------|
| `curator.port` | 8000 | Invocation (HTTP) | Unauthenticated inference triggering from any in-cluster caller |
| `health.port` | 52365 | Health / readiness | Low — read-only probe endpoint |
| `rayPorts.gcsServerPort` | 6379 | Ray GCS | Arbitrary task submission or cluster state manipulation |
| `rayPorts.dashboardPort` | 8265 | Ray dashboard | Job submission via Jobs API (arbitrary code execution), full log and data access |
| `rayPorts.objectManagerPort` | 8076 | Ray object manager | Object-store data exfiltration or injection |
| `rayPorts.nodeManagerPort` | 8077 | Ray node manager | Ray node eviction or resource starvation |
| `metrics.port` | 9002 | Prometheus metrics | Low — metric scraping |

Egress is deliberately unrestricted — the NetworkPolicy applies `policyTypes: [Ingress]` only.

### What NetworkPolicy does not protect against

NetworkPolicy governs pod-to-pod traffic only. Node-originated traffic — kubelet liveness/readiness probes, `kubectl port-forward` connections, and node-mode load balancer health checks — is exempt from NetworkPolicy on mainstream CNIs (Calico, Cilium, AWS VPC CNI, the managed GKE/EKS/AKS integrations), so the workflows documented in this README keep working. This is implementation-defined by the k8s spec, not guaranteed — if your CNI applies policy to node traffic, or you use container-native load balancing where health checks reach pod IPs directly (GCP NEGs, AWS NLB `target-type: ip`), add an `ipBlock` extraIngress rule for the node/LB CIDRs:

```yaml
networkPolicy:
  enabled: true
  extraIngress:
    - ports:
        - port: 52365  # health.port — kubelet probes
          protocol: TCP
      from:
        - ipBlock:
            cidr: 10.0.0.0/8  # replace with your node/LB CIDR
```

### NVCF deployments

NVCF enforces pod isolation at the platform level — function pods are already network-isolated by the NVCF runtime regardless of what the chart renders. This is not a security gap: NVCF provides equivalent protection through its own infrastructure. The practical constraint is that NVCF's Helm-functions allowlist does not include `NetworkPolicy`, so the chart deploy will be rejected if one is present. Set `networkPolicy.enabled: false` for all NVCF deployments.

### Standalone / native Kubernetes deployments

Enable the built-in policy:

```yaml
networkPolicy:
  enabled: true
```

This allows all TCP traffic between the chart's own pods on every port **except** `curator.port` (blocked even to same-namespace pods), and blocks all ingress from other namespaces. The OTel metrics collector (`metrics.enabled=true`) is automatically allowed to scrape `metrics.port`.

To allow a client in the **same namespace** to reach the invocation endpoint:

```yaml
networkPolicy:
  enabled: true
  allowInvocationFrom:
    - podSelector:
        matchLabels:
          app.kubernetes.io/name: my-client
```

To allow an **API gateway or authn/authz proxy in another namespace**:

```yaml
networkPolicy:
  enabled: true
  allowInvocationFrom:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: api-gateway
```

To allow a **cluster-level Prometheus** in another namespace to scrape metrics:

```yaml
networkPolicy:
  enabled: true
  extraIngress:
    - ports:
        - port: 9002
          protocol: TCP
      from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: monitoring
```

> **GitOps / ArgoCD / Flux note:** `rayAuth.enabled: true` (the default) auto-generates a 64-character token on every render via `randAlphaNum`. In a GitOps workflow this causes perpetual drift — pin the token explicitly to keep renders stable:
> ```yaml
> rayAuth:
>   enabled: true
>   token: "<your-stable-token>"
> ```

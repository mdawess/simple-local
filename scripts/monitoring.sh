#!/usr/bin/env bash
set -euo pipefail

# Azure Monitor for a deployed container app: alert rules on the platform
# metrics, a log alert on the server's own request log, and saved queries.
# Idempotent — re-running updates the existing rules in place.

RG="${RG:-simple-local-ca}"
APP="${APP:-simple-local-vl}"
WORKSPACE="${WORKSPACE:-workspace-simplelocalcaF3Gc}"
ALERT_EMAIL="${ALERT_EMAIL:-$(az account show --query user.name -o tsv)}"
ACTION_GROUP="${ACTION_GROUP:-simple-local-oncall}"

APP_ID=$(az containerapp show -n "$APP" -g "$RG" --query id -o tsv)
WORKSPACE_ID=$(az monitor log-analytics workspace show -n "$WORKSPACE" -g "$RG" --query id -o tsv)

echo "app       $APP_ID"
echo "workspace $WORKSPACE_ID"
echo "email     $ALERT_EMAIL"

ACTION_GROUP_ID=$(az monitor action-group create \
  --name "$ACTION_GROUP" --resource-group "$RG" \
  --short-name simplelocal \
  --action email oncall "$ALERT_EMAIL" \
  --query id -o tsv)

metric_alert() {
  local name=$1 severity=$2 window=$3 description=$4 condition=$5
  az monitor metrics alert create \
    --name "$name" --resource-group "$RG" --scopes "$APP_ID" \
    --condition "$condition" \
    --description "$description" \
    --severity "$severity" \
    --window-size "$window" --evaluation-frequency 5m \
    --action "$ACTION_GROUP_ID" \
    --output none
  echo "metric alert   $name"
}

# Sev 1 — the app is failing requests or crash-looping. Both mean the endpoint
# is unusable right now.
metric_alert "$APP-5xx" 1 5m \
  "$APP is returning server errors" \
  "total Requests > 0 where statusCodeCategory includes 5xx"

metric_alert "$APP-restarts" 1 5m \
  "$APP container restarted — model reload costs a cold start" \
  "max RestartCount > 0"

# Sev 2 — still serving, but degraded. Sized for the search path: a text query
# measures ~111ms p50 and ~250ms p95 on a warm T4, so a 5-minute average past 1s
# is several times worse than normal and means requests are queueing.
#
# Bulk image indexing through the same endpoint runs 2s+ per batched request and
# will trip this. That is the tradeoff of one endpoint serving both; if indexing
# starts paging you, give it its own app rather than loosening this back up.
metric_alert "$APP-latency" 2 5m \
  "$APP average response time above 1s" \
  "avg ResponseTime > 1000"

metric_alert "$APP-memory" 2 15m \
  "$APP working set above 90% of its limit" \
  "avg MemoryPercentage > 90"

# Deliberately not a GpuUtilizationPercentage alert: a warm replica with no
# traffic reads 0% and would page constantly. CPU fallback is instead caught by
# the duration of real requests, below — a T4 encode is ~2.3s, so 20s means the
# GPU is not being used.

# The server logs one line per request: model=... status=... duration_ms=...
# This catches application-level failures the ingress metric never sees,
# including the 401s from unauthenticated callers.
#
# Threshold is 15 in a 15-minute window, not 1: a single 401 from a probe or a
# stray 404 is noise, and paging on it trains you to ignore the alert. 15 is a
# rate that means something is actually wrong.
az monitor scheduled-query create \
  --name "$APP-request-errors" --resource-group "$RG" \
  --scopes "$WORKSPACE_ID" \
  --description "$APP logged 15 or more failed requests in 15 minutes" \
  --severity 2 \
  --evaluation-frequency 5m --window-size 15m \
  --condition "count 'failures' >= 15" \
  --condition-query failures="
    ContainerAppConsoleLogs_CL
    | where ContainerAppName_s == '$APP'
    | where Log_s has 'simple_local.requests'
    | extend status = toint(extract('status=([0-9]+)', 1, Log_s))
    | where status >= 400" \
  --action-groups "$ACTION_GROUP_ID" \
  --output none
echo "log alert      $APP-request-errors"

az monitor scheduled-query create \
  --name "$APP-cpu-fallback" --resource-group "$RG" \
  --scopes "$WORKSPACE_ID" \
  --description "$APP requests are slow enough to suggest the GPU is not in use" \
  --severity 3 \
  --evaluation-frequency 15m --window-size 30m \
  --condition "count 'slow' > 0" \
  --condition-query slow="
    ContainerAppConsoleLogs_CL
    | where ContainerAppName_s == '$APP'
    | where Log_s has 'simple_local.requests'
    | extend duration_ms = toint(extract('duration_ms=([0-9]+)', 1, Log_s))
    | where duration_ms > 20000" \
  --action-groups "$ACTION_GROUP_ID" \
  --output none
echo "log alert      $APP-cpu-fallback"

saved_search() {
  local id=$1 name=$2 query=$3
  az monitor log-analytics workspace saved-search create \
    --resource-group "$RG" --workspace-name "$WORKSPACE" \
    --name "$id" --display-name "$name" --category "simple-local" \
    --saved-query "$query" \
    --output none
  echo "saved search   $name"
}

saved_search sl-request-log "Request log (all apps)" "
ContainerAppConsoleLogs_CL
| where Log_s has 'simple_local.requests'
| extend model = extract('model=([^ ]+)', 1, Log_s),
         status = toint(extract('status=([0-9]+)', 1, Log_s)),
         duration_ms = toint(extract('duration_ms=([0-9]+)', 1, Log_s))
| project TimeGenerated, ContainerAppName_s, model, status, duration_ms
| order by TimeGenerated desc"

saved_search sl-latency-percentiles "Latency percentiles by model" "
ContainerAppConsoleLogs_CL
| where Log_s has 'simple_local.requests'
| extend model = extract('model=([^ ]+)', 1, Log_s),
         duration_ms = toint(extract('duration_ms=([0-9]+)', 1, Log_s))
| where isnotnull(duration_ms)
| summarize requests = count(),
            p50 = percentile(duration_ms, 50),
            p95 = percentile(duration_ms, 95),
            p99 = percentile(duration_ms, 99)
        by ContainerAppName_s, model, bin(TimeGenerated, 1h)
| order by TimeGenerated desc"

saved_search sl-errors "Errors and rejected requests" "
ContainerAppConsoleLogs_CL
| where Log_s has 'simple_local.requests'
| extend status = toint(extract('status=([0-9]+)', 1, Log_s)),
         error = extract('error=(.*)', 1, Log_s)
| where status >= 400
| project TimeGenerated, ContainerAppName_s, status, error, Log_s
| order by TimeGenerated desc"

# Image pull dominates cold start on the GPU image (9.4GB), so the pull and the
# start are tracked separately rather than as one number.
saved_search sl-cold-starts "Cold start timeline" "
ContainerAppSystemLogs_CL
| where Reason_s in ('AssigningReplica', 'PullingImage', 'PulledImage',
                     'ContainerCreated', 'ContainerStarted', 'ContainerAppReady', 'GpuInfo')
| project TimeGenerated, ContainerAppName_s, RevisionName_s, ReplicaName_s, Reason_s, Log_s
| order by TimeGenerated asc"

saved_search sl-replica-failures "Replica failures and probe errors" "
ContainerAppSystemLogs_CL
| where Reason_s has_any ('ProbeFailed', 'ContainerTerminated', 'ProcessExited', 'Deadline')
| project TimeGenerated, ContainerAppName_s, RevisionName_s, ReplicaName_s, Reason_s, Log_s
| order by TimeGenerated desc"

echo
echo "portal: https://portal.azure.com/#@/resource${APP_ID}/alerts"

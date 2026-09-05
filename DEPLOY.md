# Deploying to Azure Container Apps

Two images, deployed the same way. Container Apps runs the container as-is, so
every endpoint keeps the shape it has locally — `/v1/embeddings` for the text
embedder, `/v1/predict` for the multimodal one.

| route | model directory | hardware | scale |
| --- | --- | --- | --- |
| `text-embed` (Qwen3-Embedding-0.6B, llama.cpp) | `implementations/embeddings` | **CPU** | scale to zero |
| `image-embed` (Qwen3-VL-Embedding-2B, torch) | `implementations/vl-embeddings` | **T4 GPU** | min 1 replica |

Weights are baked into both images, so a cold start is model-load only. Measured
for the CPU image: **~2.6s** from container start to serving. The GPU image is
much larger (torch + 4.2GB of weights); measure it rather than assume.

## Routes, not model names

An app is named after the **route it serves**, not the model serving it —
`text-embed`, not `simple-local-qwen3-embedding`. With a custom DNS suffix on
the environment the app name *is* the public hostname, so a name is an API:

```
https://text-embed.model.aeriumhq.com     <- the contract
  └─ config.yml decides which model is behind it
```

Swapping the model behind a route is then a config change and a rebuild. No
caller changes a URL, because the URL never moved.

Apps run in **multiple-revision mode**, so a new image arrives taking no traffic
at all:

```bash
make apply CONFIG=...                       # new revision, labelled candidate, 0% traffic
simple-local revisions -c CONFIG            # what exists, what is live, what it weighs
curl https://text-embed---candidate.model.aeriumhq.com/health   # test it in place
simple-local promote -c CONFIG              # atomic label swap; the route never drops
simple-local promote -c CONFIG --weight 10  # or send it 10% first
simple-local rollback -c CONFIG             # swap straight back
```

`promote` is a label swap, so it is seconds and restarts nothing — the cheapest
tier there is. The previous revision keeps the `candidate` label, which is what
makes `rollback` the same operation in reverse.

Note that `---` in a candidate URL keeps it a **single DNS label**, so one
`*.model.aeriumhq.com` wildcard certificate covers every revision URL too.

## Custom domain

Option A: the environment carries a DNS suffix and every app in it serves at
`<app>.<zone>` automatically — adding a model needs no DNS change at all.

```yaml
deploy:
  name: text-embed
  domain:
    zone: model.aeriumhq.com
    zone_resource_group: aerium-dns
```

`simple-local domain -c CONFIG` reports the state and prints exactly what the
zone needs:

```
text-embed  ->  https://text-embed.model.aeriumhq.com

  zone model.aeriumhq.com: NOT created in aerium-dns
  environment suffix: not set

  records in the zone:
    A     *.model.aeriumhq.com     4.174.246.194
    TXT   asuid.model.aeriumhq.com D2C65222...
```

Setup, once:

1. `az network dns zone create -g aerium-dns -n model.aeriumhq.com`
2. At the registrar holding `aeriumhq.com`, add **NS** records for the `model`
   label pointing at the zone's four nameservers. MX records live on the apex,
   so mail is untouched — delegation only moves the `model.*` subtree.
3. Publish the A and TXT records above into the zone.
4. Set the suffix, with a wildcard certificate: ACA managed certificates cannot
   be wildcards, so `*.model.aeriumhq.com` has to be supplied. With the zone in
   Azure DNS, Let's Encrypt DNS-01 issues one for free and unattended.

```bash
az containerapp env update -n simple-local-env -g simple-local-ca \
  --dns-suffix model.aeriumhq.com \
  --certificate-file wildcard.pfx --certificate-password '...'
```

## The unit of deployment is a directory

A model directory holds everything the image needs, and nothing outside it is
copied in:

```
implementations/vl-embeddings/
  config.yml            # models, and the deploy: block
  config.container.yml  # the config that ships, when it differs from local
  runtime.py            # for kind: custom
```

The build context *is* that directory. The Dockerfile is generated from the
config rather than checked in, so there is no second place to keep in sync and
no way to forget a platform quirk — the venv is always kept out of WORKDIR, for
instance, because a hand-written Dockerfile that forgets to do that fails in a
way that takes an afternoon to diagnose (see the chown note below).

Paths inside a config resolve relative to that config, so `runtime: runtime.py:VLEmbedder`
means the file next to it — the same directory works from the repo, from `/srv`
in the image, or anywhere you move it.

`simple-local render -c <config>` prints exactly what would be built. What comes
out of the config is:

```yaml
deploy:
  build:
    base_image: python:3.12-slim   # default: llama.cpp's when a model is kind: llm
    system_packages: [build-essential]
    requirements: [torch]          # pip installs beyond simple-local
    extras: [vl]                   # simple-local extras
    env: {HF_HOME: /models}
    prefetch: true                 # bake weights in; a cold start stays model-load only
    package: simple-local==0.2.0   # unset stages a wheel built from this checkout
```

`package` is worth setting once there is a published version: it pins the
*runtime*, which is a separate rollback axis from the config. Left unset, the
build stages a wheel from the working tree, which is what you want while
developing and not what you want in production.

## make deploy

The whole flow below is wrapped up in one command, driven by the same config you
serve locally:

```bash
make deploy CONFIG=implementations/embeddings/config.yml
```

It creates whatever is missing — resource group, registry, environment, GPU
workload profile — builds the image in ACR, creates or updates the app, and
prints the URL. On first create it also prints the generated API key.

```bash
make deploy CONFIG=... ARGS=--rebuild    # force a rebuild even if the digest matches
simple-local promote -c CONFIG           # make the candidate revision live
simple-local rollback -c CONFIG          # swap the live label back
make stop CONFIG=...                     # stop it running, keep the deployment
make start CONFIG=...                    # undo stop, no rebuild
make teardown CONFIG=...                 # delete it, asks to confirm
make list                                # every deployment and its URL
```

`make deploy` is `plan` and `apply` in one step. It only rebuilds when something
in the model directory actually changed; a memory or replica change updates in
place. See "Plan before you apply" below.

`stop` deactivates the revision rather than scaling down, because
`--max-replicas` has a floor of 1 — there is no "zero capacity" scale setting to
ask for. A stopped app keeps its URL and secrets, and returns 404 until
`start` brings it back. Note that `start` restores the replica counts from
the config but does *not* rebuild, so it is seconds rather than minutes.

`make list` shows every deployment in the resource group:

```
NAME                     URL                                              STATE    REPLICAS
simple-local-embeddings  https://simple-local-embeddings.<region>.io      running  1-3
simple-local-vl          https://simple-local-vl.<region>.io              stopped  0-1
```

It reports stopped apps as such, which the platform will not do for you: an app
whose revisions are all deactivated still reports `runningStatus: Running`, so
the state comes from the revisions instead.

What to deploy comes from an optional `deploy:` block in the config, which the
server ignores:

```yaml
deploy:
  name: simple-local-vl
  gpu: NC8as-T4
  cpu: 8.0
  memory: 56.0Gi
  min_replicas: 0
  max_replicas: 1
  env:
    VL_DEVICE: cuda
    VL_DTYPE: float16
```

Every key has a default — the app name falls back to `simple-local-<directory>`
and the image config to `config.container.yml` next to the config — so a config
with no `deploy:` block at all still deploys.

Each build gets a unique tag — a timestamp plus a digest of the whole model
directory. The timestamp is because pushing over a tag the app already runs
produces no new revision, so the deploy would appear to succeed and change
nothing. The digest is so `plan` can tell a stale image from a current one
without building to find out.

The digest covers the generated Dockerfile, the served config, and every other
file in the directory. It deliberately does *not* cover the contents of the
simple-local wheel — only the requirement string — so editing the package source
does not force a rebuild on every plan. Bump `build.package`, or `--rebuild`.

## Plan before you apply

Most changes don't need a rebuild: cpu, memory, GPU and replica counts are
Container Apps settings, and only the image contents — the weights, the served
config, anything else in the model directory — require ACR. `plan` works out
which it is, prices it, and hands back a hash to apply against.

```bash
simple-local profiles -l <region> [--gpu]   # what that region actually offers
simple-local cost -c CONFIG                 # estimated monthly cost at list price
make validate CONFIG=...        # limits and sizing, without touching Azure
make plan CONFIG=...            # diff the config against the live deployment
make apply CONFIG=... HASH=...  # run the plan that HASH was approved for
```

```
simple-local-vl in simple-local-ca — running

update: new revision from the same image; replicas restart
estimated 90s, restarts replicas: True

  memory: '16Gi' -> '24Gi'  [update]
    resizes the replica; the image is reused

apply with: simple-local apply -c <config> --plan-hash 75c4e239575e2428
```

| tier | what it does | cost |
| --- | --- | --- |
| `scale` | replica bounds only; running replicas untouched | ~20s |
| `update` | cpu, memory, GPU, env — new revision from the same image | ~90s |
| `rebuild` | ACR build and push, then a new revision | ~10 min |
| `create` | provisions the app and everything it needs | ~12 min |

The hash covers both the desired config *and* the observed deployment, so
`apply` refuses if either moved after the plan was read:

```
plan is stale: approved 75c4e239, current b6f48147 — re-plan and review again
```

That is the whole point of the two-step: a console can render a plan, wait for
someone to press Deploy, and know that what runs is what they agreed to.

### Cost

`plan` carries a monthly estimate, and prices the change rather than just the
end state:

```
update: new revision from the same image; replicas restart
estimated 90s, restarts replicas: True
cost: USD 442-3,027/month (+USD 1,009 at max)
```

`simple-local cost -c <config>` shows the breakdown:

```
simple-local-vl on Consumption-GPU-NC8as-T4 (4 vCPU, 16Gi, 1-4 replicas)

  per replica, per month:
    vCPU + memory, active              USD       525.60
    vCPU + memory, idle                USD       210.24
    1x GPU                             USD       231.26

  floor (min_replicas, at rest)      USD       441.50
  ceiling (max_replicas, active)     USD     3,027.46
```

Two numbers rather than one, because the honest answer is a range. **Floor** is
what `min_replicas` costs sitting there doing nothing; **ceiling** is
`max_replicas` busy for the whole month. A scale-to-zero app with no traffic
genuinely costs nothing, and the estimate says so.

Rates come from the [Azure Retail Prices API](https://prices.azure.com), cached
next to the profile catalog, and are **list price** — no reservation, savings
plan, AHB or enterprise discount. Requests are billed separately and not
included. Treat it as an upper bound and as a way to compare two options, not as
a forecast of the invoice.

Consumption profiles bill the replica's cpu/memory request per second, split
into active and idle rates. Dedicated profiles bill the **whole node** by the
hour whether or not the replica asks for it — an `NC96-A100` is about USD 25,000
a month at list regardless of what you request on it, which is worth seeing
before rather than after.

Cost is deliberately not part of the plan hash. Prices move on Azure's schedule
and a price change must not invalidate a plan someone already approved.

### Where the hardware limits come from

Nothing in this repo hardcodes what Azure offers. `simple-local profiles` reads
`az containerapp env workload-profile list-supported` for a region, caches it
under `~/.cache/simple-local/profiles/`, and every check reports which it used:

```
live from Azure (westus3)
cached 12d ago (canadacentral)
```

`plan` fetches live; `validate` uses the cache so it stays offline and fast.
With neither, sizing is reported as **not checked** rather than passing quietly.
Two things Azure will not tell you, so a config declares them:

```yaml
deploy:
  quotas:                            # no quota API exists for Container Apps
    Consumption-GPU-NC8as-T4: 2
  profiles:                          # hardware the catalog does not have yet
    NC40-H100: {category: GPUAccelerated, cores: 40, memory_gi: 320, gpus: 1}
```

`validate` needs no Azure credentials and exits nonzero on an error, so it works
as a CI gate on config changes. It enforces what the sizing notes below only
describe — the 8GiB Consumption ceiling, GPU node sizes, `ubatch_size` against
`max_input_tokens` — including the failure where a too-large `context_length`
loads cleanly and then dies on the first big input:

```
ERROR  models.embed.inference.context_length: estimated peak memory does not fit:
       KV 7Gi + compute 1.64Gi = 8.64Gi against a 8Gi limit, excluding model weights.
       The server loads fine and then dies on the first large input.
```

Every command takes `--json` for programmatic use. `apply --json` streams NDJSON
progress events and ends with a `result` object.

The rest of this document is what that script does, for when you need to do it
by hand or change how it works.

## Build

Build in ACR — it's native amd64, so no emulation from a Mac, and it pushes in
one step:

```bash
ACR=<your-registry>
CONTEXT=$(simple-local render -c implementations/embeddings/config.yml --json | jq -r .context)

az acr build --registry $ACR --platform linux/amd64 \
  -t simple-local-embeddings:latest "$CONTEXT"
```

`simple-local apply` does this for you, staging a build context in a temp
directory first; the manual form is here for when you need to look at it.

Use a **Premium** ACR if you want [artifact streaming](https://learn.microsoft.com/azure/container-registry/container-registry-artifact-streaming),
which is the main lever on GPU cold start.

## Text embeddings (CPU, scales to zero)

```bash
RG=<resource-group>
ENV=<environment-name>
KEY=$(openssl rand -hex 16)

az containerapp env create --name $ENV --resource-group $RG --location swedencentral

az containerapp create \
  --name simple-local-embeddings \
  --resource-group $RG \
  --environment $ENV \
  --image $ACR.azurecr.io/simple-local-embeddings:latest \
  --registry-server $ACR.azurecr.io \
  --target-port 8080 --ingress external \
  --cpu 2.0 --memory 4.0Gi \
  --min-replicas 0 --max-replicas 3 \
  --secrets "api-key=$KEY" \
  --env-vars "SIMPLE_LOCAL_API_KEY=secretref:api-key"
```

## Multimodal embeddings (T4 GPU, stays warm)

GPU needs a workload profile on the environment. A100 and T4 quota is enabled by
default for pay-as-you-go and enterprise agreements; otherwise request it
through a support case.

```bash
az containerapp env workload-profile add \
  --name $ENV --resource-group $RG \
  --workload-profile-name NC8as-T4 \
  --workload-profile-type Consumption-GPU-NC8as-T4

az containerapp create \
  --name simple-local-vl \
  --resource-group $RG \
  --environment $ENV \
  --image $ACR.azurecr.io/simple-local-vl:latest \
  --registry-server $ACR.azurecr.io \
  --target-port 8080 --ingress external \
  --workload-profile-name NC8as-T4 \
  --cpu 8.0 --memory 56.0Gi \
  --min-replicas 1 --max-replicas 2 \
  --secrets "api-key=$KEY" \
  --env-vars "SIMPLE_LOCAL_API_KEY=secretref:api-key" "VL_DEVICE=cuda" "VL_DTYPE=float16"
```

`--min-replicas 1` keeps it warm: image search shouldn't pay a cold start. Drop
to `0` for a bulk-indexing deployment where a slow first request is fine.

## Verify

```bash
FQDN=$(az containerapp show -n simple-local-vl -g $RG --query properties.configuration.ingress.fqdn -o tsv)

curl -s https://$FQDN/health
curl -s https://$FQDN/v1/predict \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"input": [{"image": "https://example.com/photo.jpg"}]}'
```

`nvidia-smi` from the container console (Monitoring → Console in the portal)
confirms the GPU is actually attached.

## Sizing notes

- **T4 has no bfloat16.** `VL_DTYPE=float16` there; use `bfloat16` only on A100.
- **Latency**: measure before promising numbers. If a T4 is too slow for
  interactive image search, the options are A100 (change the workload profile),
  larger `batch_size` for indexing throughput, or MRL truncation
  (`dimensions: 512`) to shrink vectors — the last affects storage and recall,
  not encode time.
- **Multi-GPU replicas exist, but only on dedicated profiles in some regions.**
  The Consumption GPU profiles give one GPU per replica; `westus3` also offers
  dedicated `NC48-A100` (2 GPUs) and `NC96-A100` (4). Fractional GPUs aren't
  supported, and only the first container in an app gets the GPU. Run
  `simple-local profiles -l <region> --gpu` rather than trusting this list.
- **No H100, H200 or B200 on Container Apps.** Those exist as Azure *VM* sizes
  (`NC40ads H100 v5` and friends) but ACA exposes no workload profile for them
  in any of the 63 physical regions. T4 and A100 are the whole GPU menu here;
  anything beyond that means AKS, Azure ML, or Batch. Training in particular
  belongs on one of those — ACA is a serving platform.
- **Serverless T4 quota is 2 per subscription by default**, which caps replicas
  regardless of `max_replicas`. Past it the platform refuses the pod outright —
  `exceeded quota: consumption-gpu-t4 ... limited: 2` in the system logs — while
  the app keeps reporting healthy on the replicas it already has. Raising it is
  a support case. Measured capacity for search queries: **~11-12 q/s per
  replica**, ~16 q/s across two under saturation.
- **Scaling out is not fast.** A new replica appears within ~30s but needs ~90s
  more to load the model and reach full throughput, so bursts get about a minute
  and a half of degradation. Capacity for a known peak belongs in
  `min_replicas`, not `max_replicas`.
- **CUDA**: the image ships its own runtime via the torch wheels, so platform
  CUDA upgrades don't break it.

## Request logging (Azure Database for MySQL)

Both container configs log every call, response, and error to MySQL over TLS.
The server creates the `request_log` table itself on first connect. Writes go
through a background queue, so a slow or unreachable database never blocks or
fails an inference request.

Create a flexible server you own — it lives in your subscription, so you control
access, retention, and backups:

```bash
az mysql flexible-server create \
  --resource-group $RG --name simple-local-log \
  --location swedencentral \
  --admin-user sladmin --admin-password "<strong-password>" \
  --tier Burstable --sku-name Standard_B1ms \
  --database-name simple_local \
  --public-access <container-apps-outbound-ip>
```

Then pass the connection details to both apps:

```bash
az containerapp update -n simple-local-embeddings -g $RG \
  --set-env-vars \
    "MYSQL_HOST=simple-local-log.mysql.database.azure.com" \
    "MYSQL_USER=sladmin" \
    "MYSQL_DATABASE=simple_local" \
    "MYSQL_PASSWORD=secretref:mysql-password"
```

`ssl: true` is already set in the configs — Azure MySQL rejects unencrypted
connections. To pin the CA rather than just encrypt, download Azure's root
certificate into the image and set `ssl_ca` to its path.

**Leaving `MYSQL_HOST` unset turns logging off entirely**, so the same image runs
with or without a database.

One thing to decide deliberately: the log stores full request *and response*
bodies (truncated at 60KB). For image search that means base64 image payloads
land in the database. If you only want metadata, null out the `request` and
`response` columns in `dblog.py` — the token counts, latencies, models, and
errors are separate columns and stay intact.

## Sizing llama.cpp slots against the memory cap

Concurrency on the CPU embedder is set by `inference.parallel` — llama.cpp
slots, not vCPU. Eight slots serve eight concurrent requests; a ninth queues no
matter how many cores the replica has.

Two things make this harder than turning the number up.

**`context_length` is the total across slots**, so each request gets
`context_length / parallel`. Raising `parallel` without raising
`context_length` silently halves the per-request limit.

**A Consumption replica caps at 8GiB**, and it is a hard cap — 4 vCPU forces
8GiB. `Flex` is a larger Consumption-category profile (32 vCPU / 128GiB) that is
worth investigating before reaching for a dedicated profile, but the numbers
below are for the standard `Consumption` replica. Everything has to fit:

| | cost |
| --- | --- |
| KV cache (preallocated at startup) | ~112KiB per token of *total* context |
| compute buffer (per in-flight request) | ~3.2GiB for a ~4000-token input |
| model (Qwen3-Embedding-0.6B, Q8_0) | ~0.65GiB |

So `context_length: 32768` costs ~3.5GiB of KV and idles at ~47%, leaving room
for roughly one large request in flight. `context_length: 65536` also *loads
fine* — and then dies on any input past ~2000 tokens, because the KV cache fits
but the compute buffer no longer does.

That failure is worth recognising: the container starts, `/health` returns 200,
the model reports ready, and it stays that way until a large input arrives. The
server dies mid-request, the client sees `Server disconnected without sending a
response`, and the supervisor restarts it about 5s later. Nothing in the config
looks wrong.

The current settings — `parallel: 8`, `context_length: 32768`, `ubatch_size`
and `max_input_tokens` at 4096 — are sized for **short search queries**, which
is what this endpoint serves. Measured there: 8 concurrent, ~31 q/s, p95 150ms.
Inputs over 4096 tokens get a 413 rather than taking the server down.

Keep `ubatch_size` >= `max_input_tokens`. An embedding input longer than one
ubatch hangs rather than erroring — see the note in `runtimes/llm.py`.

For a bulk-indexing deployment, give it its own app and config rather than
raising the limits here: long inputs need the memory headroom that concurrency
is using. `make deploy CONFIG=<its config>` with a `deploy:` block naming a
different app is the whole setup.

## Monitoring (Azure Monitor)

MySQL logging above is the durable record of *what was asked*. Azure Monitor is
the operational view — is it up, is it slow, is it on the GPU — and it needs no
code in the image: the Container Apps environment already ships stdout and
platform events to the Log Analytics workspace it was created with.

```bash
./scripts/monitoring.sh
```

Idempotent, so re-running updates the rules in place. It reads `RG`, `APP`,
`WORKSPACE`, `ALERT_EMAIL` and `ACTION_GROUP` from the environment; the defaults
target `simple-local-vl`. Point `APP` at the CPU app to get the same rules
there, minus the GPU one.

### Alert rules

| rule | sev | fires when |
| --- | --- | --- |
| `-5xx` | 1 | any 5xx through ingress over 5 min |
| `-restarts` | 1 | container restarted — every restart is a model reload |
| `-latency` | 2 | average response time above 1s over 5 min |
| `-memory` | 2 | working set above 90% of the limit for 15 min |
| `-request-errors` | 2 | the server logged a request with status >= 400 |
| `-cpu-fallback` | 3 | a request took over 20s — the GPU is likely not in use |

The last two are log alerts rather than metric alerts, and both exist because
the metric they'd replace doesn't say what you need.

`-request-errors` parses the server's own `model=… status=… duration_ms=…`
line, so it catches what never reaches the ingress metric — including the 401s
from unauthenticated callers.

`-cpu-fallback` is how you find out the runtime silently dropped to CPU, which
is worth catching because a T4 app that answers correctly but slowly looks
healthy on every other signal. The obvious version of this rule — alert on
`GpuUtilizationPercentage` near zero — is wrong here: with `--min-replicas 0` a
warm but idle replica legitimately reads 0%, so it would page constantly. A T4
encode is ~2.3s, so request duration is the signal that only moves when
something is actually broken. For a spot check rather than an alert:

```bash
az monitor metrics list --resource $APP_ID \
  --metric GpuUtilizationPercentage --interval PT5M --aggregation Average -o table
```

All six notify the `simple-local-oncall` action group, which is email-only.
Add SMS or a webhook with `az monitor action-group update` — the rules
themselves don't change.

### Saved queries

Under **Log Analytics → Queries → simple-local**, or from the CLI:

```bash
az monitor log-analytics query -w <workspace-guid> --analytics-query "
ContainerAppConsoleLogs_CL
| where Log_s has 'simple_local.requests'
| extend model = extract('model=([^ ]+)', 1, Log_s),
         duration_ms = toint(extract('duration_ms=([0-9]+)', 1, Log_s))
| summarize p50 = percentile(duration_ms, 50), p95 = percentile(duration_ms, 95)
        by ContainerAppName_s, model, bin(TimeGenerated, 1h)" -o table
```

"Cold start timeline" is the one to reach for on the GPU app: it separates image
pull from container start, which is what tells you whether a slow rollout is the
9.4GB pull or the model load.

### Keep the venv out of WORKDIR

Container Apps runs a recursive `chown` over WORKDIR before starting a
container, and gives it roughly three minutes. The torch and CUDA wheels are
enough files to blow that budget, which kills the replica before the server ever
runs:

```
Container 'simple-local-vl' was terminated with exit code '1' and reason
'ContainerCreateFailure'. Command: ["chown","-R","10000:10000",...,"/srv"]
canceled after 191756 ms.
```

Generated Dockerfiles always put the venv at `/opt/venv`, outside WORKDIR, so
WORKDIR holds only the model directory and the chown is trivial. Note that it is
the *file count* under WORKDIR that matters, not image size — the 4.2GB of
weights under `HF_HOME=/models` are outside WORKDIR and were never part of the
problem, and the image pull itself completes fine in ~105s.

This is the main reason the Dockerfile is generated rather than written by hand.
The CPU image would survive a venv inside WORKDIR today, and a hand-written one
would probably put it there — right up until someone adds a heavy dependency.

This failure is worth knowing by sight because it is quiet: the revision stays
at 0 replicas and takes no traffic, while the previous revision keeps serving.
`/health` returns 200 throughout and the app shows `Running`. The "Replica
failures and probe errors" saved query is how you see it, and checking that a
deploy actually moved traffic to the new revision is how you avoid being fooled:

```bash
az containerapp revision list -n simple-local-vl -g $RG \
  --query "[].{name:name,replicas:properties.replicas,health:properties.healthState}" -o table
```

## Access

Both apps use `--ingress external`, so they get a public HTTPS URL with an
Azure-managed certificate, protected by the bearer key in
`SIMPLE_LOCAL_API_KEY`. Requests without it get a 401, and those 401s are logged
too.

If the customer later wants network-level isolation, `--ingress internal` plus a
VNet keeps the same code and config — it only changes who can reach the URL.

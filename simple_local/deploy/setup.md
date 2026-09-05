# Custom domain setup: `*.model.aeriumhq.com`

A one-time runbook to put the deployments on `model.aeriumhq.com`, with stable
route names and swappable revisions behind them.

**The existing deployments keep serving throughout.** Nothing here modifies
`simple-local-embeddings` or `simple-local-vl` until the final phase, which you
only run once the new routes are verified and callers have moved. Every earlier
phase is additive, so you can stop after any of them.

Hostinger does not support NS records, so the subdomain is not delegated to
Azure DNS. Each route binds its own hostname and Azure issues a free,
auto-renewing certificate — four DNS records in total, no zone, no wildcard
certificate, and nothing that expires.

Budget about **half an hour of work** plus image build time.

## Current state, for reference

Captured before any changes:

| | |
| --- | --- |
| subscription | `a7d822da-6f36-48b7-8597-967ce3f0baff` |
| environment | `simple-local-env` in `simple-local-ca` (canadacentral) |
| environment static IP | `4.174.246.194` |
| default domain | `redstone-84592858.canadacentral.azurecontainerapps.io` |
| domain verification ID | `D2C652226AEBE5D144B3BCF9B22418E4024A69CB55E6F17675D6C0684A7F930B` |
| workload profiles | `Consumption`, `NC8as-T4` |
| running apps | `simple-local-embeddings` (Single, 1-4), `simple-local-vl` (Single, 1-4) |

Where it lands:

| new route | replaces | model directory |
| --- | --- | --- |
| `text-embed.model.aeriumhq.com` | `simple-local-embeddings` | `implementations/embeddings` |
| `image-embed.model.aeriumhq.com` | `simple-local-vl` | `implementations/vl-embeddings` |

## Before you start

```bash
az account show --query "{name:name, id:id}" -o tsv     # right subscription?
az account set --subscription a7d822da-6f36-48b7-8597-967ce3f0baff

cd ~/Documents/personal/simple-local
uv run pytest -q                                        # should be all green
```

Record what is serving now, so you can compare later:

```bash
uv run simple-local list
curl -s https://simple-local-embeddings.redstone-84592858.canadacentral.azurecontainerapps.io/health
curl -s https://simple-local-vl.redstone-84592858.canadacentral.azurecontainerapps.io/health
```

You will also need the API keys for the existing apps if you want to test
inference rather than just `/health`:

```bash
az containerapp secret show -n simple-local-vl -g simple-local-ca \
  --secret-name api-key --query value -o tsv
```

---

## Phase 1 — DNS records at Hostinger

*Touches nothing that is running. Safe to stop after this.*

Hostinger does not support NS records, so the subdomain cannot be delegated to
Azure DNS. That rules out the wildcard-certificate approach and leaves the
simpler one: bind each hostname to its app and let **Azure issue and renew a
free certificate**. No zone to create, no delegation, no `.pfx` to own and
rotate every 90 days. The cost is two records per route instead of two in total.

**1.1 Get the records.** The app does not need to exist yet:

```bash
uv run simple-local domain -c implementations/embeddings/config.yml
uv run simple-local domain -c implementations/vl-embeddings/config.yml
```

Which prints, for `text-embed`:

```
  DNS records to add in the aeriumhq.com zone:
    CNAME  text-embed.model           text-embed.redstone-84592858.canadacentral.azurecontainerapps.io
    TXT    asuid.text-embed.model     D2C652226AEBE5D144B3BCF9B22418E4024A69CB55E6F17675D6C0684A7F930B
```

**1.2 Add them.** Hostinger → `aeriumhq.com` → **Manage DNS records**. Four
records in total, two per route:

| Type | Name | Value |
| --- | --- | --- |
| CNAME | `text-embed.model` | `text-embed.redstone-84592858.canadacentral.azurecontainerapps.io` |
| TXT | `asuid.text-embed.model` | `D2C652226AEB…` |
| CNAME | `image-embed.model` | `image-embed.redstone-84592858.canadacentral.azurecontainerapps.io` |
| TXT | `asuid.image-embed.model` | `D2C652226AEB…` |

> The name includes `.model` because you are editing the **`aeriumhq.com`** zone,
> not a delegated `model.aeriumhq.com` one. A record named `text-embed` would
> resolve at `text-embed.aeriumhq.com` and never be found. The `domain` command
> prints the full FQDN under each record — check it reads
> `text-embed.model.aeriumhq.com`.

Both TXT records hold the same value: the verification ID is scoped to the
subscription, not to an app.

**Do not touch the `Nameservers` panel.** It should keep saying
`ns1.dns-parking.com` / `ns2.dns-parking.com`. Adding Azure nameservers there
would make Azure authoritative for all of `aeriumhq.com`, where no such zone
exists — resolvers would get NXDOMAIN for your apex roughly half the time, so
**mail would fail intermittently and at random**. Nothing in this runbook needs
that panel.

**1.3 Verify.** Usually minutes.

```bash
dig CNAME text-embed.model.aeriumhq.com +short   # the azurecontainerapps.io name
dig TXT asuid.text-embed.model.aeriumhq.com +short
dig MX aeriumhq.com +short                       # must be unchanged
```

The CNAME will not resolve to an address until the app exists — that is expected
at this point. What matters is that the record itself is being served.

**Rollback:** delete the four records. Nothing else has changed.

---

## Phase 2 — Nothing to do

The wildcard certificate step is gone. Azure issues a managed certificate during
Phase 3 and renews it automatically for as long as the DNS records stay in
place. There is no expiry to diary.

---

## Phase 3 — Nothing to do

The environment DNS suffix step is gone too, which removes the one change that
touched the environment your live apps run in. `simple-local-embeddings` and
`simple-local-vl` are now untouched right up to Phase 7.

> If you ever move DNS somewhere that supports NS records, the wildcard route
> becomes available again: set `certificate: wildcard` and `zone_resource_group`
> in the config, and `simple-local domain` will print the delegation and
> environment-suffix steps instead. Worth doing only once something is creating
> models without a human, since it is what removes per-route DNS work.

---

## Phase 4 — Deploy the new routes

*Additive. The old apps are untouched and still serving.*

Start with the CPU one; it is cheaper and faster to iterate on.

```bash
make plan CONFIG=implementations/embeddings/config.yml
```

Expect `create`, roughly 12 minutes, and a cost estimate. Then:

```bash
make apply CONFIG=implementations/embeddings/config.yml
```

It prints the generated API key on first create — **save it**, it is only shown
once (`az containerapp secret show -n text-embed -g simple-local-ca
--secret-name api-key --query value -o tsv` reads it back).

Now bind the hostname — this is what makes Azure issue the certificate, and it
needs the Phase 1 records to be resolving:

```bash
uv run simple-local domain -c implementations/embeddings/config.yml --bind
```

Certificate issuance takes a minute or two. Then verify:

```bash
curl -s https://text-embed.model.aeriumhq.com/health
uv run simple-local revisions -c implementations/embeddings/config.yml
```

The first revision takes the `live` label and 100% of traffic.

Then the GPU one — allow ~15 minutes for the 9.4GB image:

```bash
make plan  CONFIG=implementations/vl-embeddings/config.yml
make apply CONFIG=implementations/vl-embeddings/config.yml
uv run simple-local domain -c implementations/vl-embeddings/config.yml --bind
curl -s https://image-embed.model.aeriumhq.com/health
```

Test real inference against both, not just `/health` — a replica that fails on
its first large input still reports healthy:

```bash
curl -s https://text-embed.model.aeriumhq.com/v1/embeddings \
  -H "Authorization: Bearer <new key>" -H "Content-Type: application/json" \
  -d '{"input": "a realistic query of the length you actually send"}' | head -c 300
```

**Rollback:** `uv run simple-local teardown -c <config>`. The old apps never
stopped serving.

---

## Phase 5 — Confirm the revision workflow

Worth doing once while nothing depends on the new routes yet, so the mechanism
is familiar before you need it under pressure.

Change something small in `implementations/embeddings/config.container.yml` —
`parallel: 8` to `parallel: 6` will do — then:

```bash
make apply CONFIG=implementations/embeddings/config.yml
uv run simple-local revisions -c implementations/embeddings/config.yml
```

The new revision should show `candidate` and **0%** traffic. The route is still
serving the old one. Test the candidate directly:

```bash
uv run simple-local revisions -c implementations/embeddings/config.yml   # revision names
curl -s https://text-embed--<revision-suffix>.redstone-84592858.canadacentral.azurecontainerapps.io/health
```

Only the bound route is on the custom domain, so a candidate is reached at its
`azurecontainerapps.io` revision URL. That is fine — a candidate URL is for you,
not for callers, and the stable route is the part that has to look right.

Promote, then roll back, to prove both directions work:

```bash
uv run simple-local promote  -c implementations/embeddings/config.yml
uv run simple-local rollback -c implementations/embeddings/config.yml
```

Revert the config change when you are done.

---

## Phase 6 — Cutover

Move callers from the old URLs to the new ones. The new routes use **new API
keys** — the old apps' secrets did not come along.

| old | new |
| --- | --- |
| `https://simple-local-embeddings.redstone-….azurecontainerapps.io` | `https://text-embed.model.aeriumhq.com` |
| `https://simple-local-vl.redstone-….azurecontainerapps.io` | `https://image-embed.model.aeriumhq.com` |

Leave both sets running until you are satisfied. This is the last point where
rollback is a one-line config change on the caller.

---

## Phase 7 — Leave the old apps running

**Not part of this weekend.** `simple-local-embeddings` and `simple-local-vl`
stay up. Nothing in Phases 1–6 touches them, and keeping them there is what
makes the whole migration reversible: if anything about the new routes turns out
wrong, callers point back at the old URLs and you have lost nothing.

The cost of that safety is real, though, so it should be a decision rather than
a thing that quietly continues. The old pair floors at roughly **USD 570/month**
combined. Set a reminder to revisit.

When you are ready, the gentle first move is to stop rather than delete — it
finds callers you forgot about, and it is reversible in seconds:

```bash
az containerapp revision deactivate -n simple-local-vl -g simple-local-ca \
  --revision "$(az containerapp show -n simple-local-vl -g simple-local-ca \
    --query properties.latestRevisionName -o tsv)"
```

`activate` puts it straight back. Leave them stopped for a week or two; a
stopped app costs nothing and keeps its URL, secrets and revision history.

Only then, and only once the request log is quiet:

```bash
make mysql-tail CONFIG=implementations/embeddings/config.yml

az containerapp delete -n simple-local-embeddings -g simple-local-ca --yes
az containerapp delete -n simple-local-vl -g simple-local-ca --yes
```

---

## Cost while both are up

Both sets run at once from Phase 4 onward, which roughly doubles spend. At list
price the old pair floors at about **USD 570/month** combined and the new pair
about the same, so this is around **USD 1,140/month** for as long as you keep
both. Deliberate and reversible is worth paying for; forgetting about it is not.

Stopping the old apps (Phase 7) drops their cost to nothing while keeping them
recoverable, which is the cheapest form of the same insurance.

```bash
uv run simple-local cost -c implementations/embeddings/config.yml
uv run simple-local cost -c implementations/vl-embeddings/config.yml
```

## Later: moving DNS hosting off Hostinger

Not needed for anything above, and not urgent. It becomes worth doing when the
console starts creating models on its own, because at that point every new route
needs two DNS records written by a machine rather than a person.

Hostinger does publish an API with DNS endpoints (`developers.hostinger.com`),
so automating four records there is possible. Two things to check before relying
on it: whether record updates are additive or **replace the whole zone**, and
whether a partial failure can leave the zone half-written. A full-zone PUT that
drops `MX` because the payload was built from an incomplete read is a very
expensive bug, and DNS gives you no undo. If you do automate against it, read
the zone first, modify, write back, then assert `MX` survived.

The alternative is to move DNS *hosting* to Azure DNS while leaving the domain
**registered at Hostinger** — you are only changing which nameservers answer,
not who owns the domain. That gets NS support, a proper API with the same
credentials as everything else here, and makes the wildcard certificate route
available again, which removes per-route DNS entirely.

Done in this order it does not risk email:

1. **Lower TTLs first.** Set MX and apex records to 300 seconds at Hostinger and
   wait a day. This is what makes rollback fast if step 5 goes wrong.
2. **Export the current zone.** Hostinger's DNS page has an Export button. Keep
   the file — it is your record of truth and your rollback.
3. **Recreate everything in Azure DNS**, into a zone for `aeriumhq.com`. Miss
   nothing: `MX`, the `SPF` TXT, `DKIM` (TXT or CNAME, often on a selector
   subdomain), `DMARC` at `_dmarc`, and any `autodiscover`/`autoconfig` records.
   Mail breaks quietly when SPF or DKIM go missing — it still sends, it just
   starts landing in spam.
4. **Verify before cutting over.** This is the step that makes it safe: you can
   query the new nameservers directly while they are not yet authoritative.

   ```bash
   dig @ns1-01.azure-dns.com MX aeriumhq.com +short
   dig @ns1-01.azure-dns.com TXT aeriumhq.com +short
   dig @ns1-01.azure-dns.com TXT _dmarc.aeriumhq.com +short
   ```

   Compare each against the same query without `@`ns. They must match exactly
   before you go on.
5. **Switch the nameservers** at Hostinger to the four Azure ones — this is the
   `Nameservers` panel that everything above told you not to touch, and this is
   the one occasion it is correct to.
6. **Verify again and send real mail** both directions. Check a message's
   headers for `spf=pass` and `dkim=pass` rather than trusting that it arrived.
7. **Leave the Hostinger zone in place** for a month. Rollback is switching the
   nameservers back, and that only works if the old zone still exists.

One ongoing consequence worth knowing: if Hostinger hosts your mailboxes, it
currently manages those mail records for you. After the move they are static
copies, so if Hostinger ever changes its mail server hostnames, nothing updates
automatically and mail stops. Rare, but it is now your responsibility rather
than theirs.

## If something goes wrong

| symptom | most likely cause |
| --- | --- |
| Hostinger rejects the nameserver with "Value must be valid IPv4 or IPv6" | you are on the `Child nameservers` tab; NS records live under `Manage DNS records` |
| email stops arriving, or arrives intermittently, after step 1.3 | Azure nameservers were added to the domain-level `Nameservers` panel. Set it back to `ns1.dns-parking.com` / `ns2.dns-parking.com` immediately; the delegation belongs in an NS *record*, not there |
| `dig NS model.aeriumhq.com` returns nothing | delegation not propagated yet, or the NS host was entered as `model.aeriumhq.com` rather than `model` |
| lego fails on DNS-01 | zone not delegated yet, or `AZURE_RESOURCE_GROUP` pointing somewhere other than `aerium-rg` |
| `env update` rejects the certificate | `.pfx` password wrong, or the cert is not a wildcard for `*.model.aeriumhq.com` |
| new route returns 404 | the app exists but the wildcard A record is missing, or the suffix did not register |
| new route returns 502/503 | the app is starting, or died on its first request — `az containerapp logs show -n text-embed -g simple-local-ca --follow` |
| old URLs break after Phase 3 | clear the suffix (rollback in Phase 3) and reassess before continuing |

The saved queries and alert rules in `DEPLOY.md` under **Monitoring** apply to
the new apps too, but `scripts/monitoring.sh` defaults to the old app names —
run it with `APP=text-embed` and `APP=image-embed` once the cutover is done.

# Stealth Social

Stealth Social is a Python 3.12+ safety-first automation bot for X and Reddit. Local Ollama inference creates reviewable candidates; it never publishes content by itself.

## Setup

Install the package in a Python 3.12+ environment:

```bash
python -m pip install .
```

Copy `config.example.json` to `config.json`, then review the account, subreddit, browser-profile, dry-run, and approval settings. Environment overrides are listed in `.env.example`; they use the `STEALTH_BOT_...` prefix. Authenticate an already configured browser profile manually:

```bash
python -m bot login x --account main
python -m bot login reddit --account main
```

The sample keeps `dry_run=true` and `manual_approval=true`. Keep both settings while learning the workflow.

## Ollama

Install Ollama using its official installer, start the local service, and install the model explicitly:

```bash
ollama pull qwen3:8b
ollama list
ollama run qwen3:8b
```

The bot checks the configured local service with:

```bash
python -m bot ollama status
python -m bot ollama models
python -m bot ollama check
```

Generation uses non-streaming structured JSON with `think=false` by default. The bot never stores or logs reasoning, cookies, browser storage, credentials, complete prompts, or complete malformed responses, and it never downloads a model automatically. A missing model reports the exact `ollama pull <model>` guidance.

## Browser-only examples

Examples come only from configured or explicit public X/Reddit pages through the existing authenticated `BrowserManager` profile. The collectors do not use platform APIs, request interception, cookies, session storage, CAPTCHA solving, fingerprint changes, or challenge bypasses.

Configure source accounts, queries, public post URLs, and allowlisted Reddit communities explicitly. Collection is bounded by per-source item/comment limits, score thresholds, useful windows, expiry, and refresh settings. A challenge, login wall, rate limit, or unavailable page stops that platform run and reports the saved count and safe next action. Raw usernames are replaced by platform-scoped hashes. The selector can also use the account's own successful published history when a public external URL exists.

```bash
python -m bot examples collect x --account main --query "local tools"
python -m bot examples collect reddit --account main --subreddit SideProject
python -m bot examples list --active-only
python -m bot examples show EXAMPLE_ID
python -m bot examples disable EXAMPLE_ID
python -m bot examples refresh --platform reddit --account main
```

Public targets and examples are untrusted data. They are framed as data in prompts; embedded instructions, URLs, commands, role claims, and schemas are not executed. High-risk injection findings quarantine an example from selection; lower-risk findings remain visible as structured metadata. Authored paragraphs and wording are preserved.

## Generation and approval

Every request has an explicit purpose: `educational`, `product-update`, `builder-update`, `promotional`, `organic-discussion`, or `customer-support`. Posts default to `educational`; comments and replies default to `organic-discussion`. Account identity, products, verified facts, forbidden claims, disclosures, and Reddit community rules are applied per account. Promotional Reddit requests are rejected unless the destination has an explicit rule with `allow_promotional_content=true`; a disclosure does not create permission to promote.

The five generation commands are:

```bash
python -m bot generate x-post --account main --topic "local-first workflows" --purpose educational
python -m bot generate x-reply --account main --target https://x.com/example/status/123 --purpose organic-discussion
python -m bot generate reddit-post --account main --subreddit SideProject --topic "a small project lesson" --purpose builder-update
python -m bot generate reddit-comment --account main --target https://old.reddit.com/r/SideProject/comments/abc/example/ --purpose organic-discussion
python -m bot generate reddit-reply --account main --target https://old.reddit.com/r/SideProject/comments/abc/example/def/ --purpose customer-support
```

Safe purpose-specific examples (configure the account identity and required disclosures first):

```bash
python -m bot generate x-post --account main --topic "a documented toolkit update" --purpose product-update --product-context "Example Toolkit" --fact "Example Toolkit is a fictional local project used in this sample." --additional-instructions "This is an authorized product update. State the configured affiliation and include every configured required disclosure; use only configured verified facts and make no unsupported results claims."
python -m bot generate x-post --account main --topic "a transparent project announcement" --purpose promotional --product-context "Example Toolkit" --fact "Example Toolkit is a fictional local project used in this sample." --call-to-action "Read the project notes" --additional-instructions "Run only after explicit account-owner permission. State the configured affiliation and include every configured required disclosure; use only configured verified facts and do not invent performance, customer, or endorsement claims."
```

The promotional command is an example of a guarded request, not permission to promote: for Reddit, the destination rule must also set `allow_promotional_content=true`, and every configured disclosure remains mandatory.

Common options include `--goal`, `--product-context`, `--project-context`, `--target-audience`, `--tone`, `--desired-length`, `--call-to-action`, repeatable `--fact`, `--forbidden-claim`, `--forbidden-phrase`, and `--keyword`, plus `--additional-instructions`, `--candidate-count`, `--profile`, `--campaign-id`, `--generate-at`, `--no-review`, and `--bypass-approval`.

An immediate request persists every candidate before review. Interactive review displays one persisted candidate at a time:

```text
[a] Approve  [r] Reject  [e] Edit  [n] Next  [q] Quit
```

Reddit-post edits use `typer.edit` with exactly one `---BODY---` delimiter and a nonblank `Title:` line. Editing creates a new revision; the original text never changes. Noninteractive terminals act as `--no-review`. With normal manual approval, `--no-review` leaves candidates pending. `--bypass-approval` requires `automation.allow_unattended_approval=true` and accepts only the highest-ranked candidate with zero validation errors and zero warnings. Neither path schedules or publishes.

Candidate management is explicit:

```bash
python -m bot candidates list REQUEST_ID
python -m bot candidates show CANDIDATE_ID
python -m bot candidates approve CANDIDATE_ID --note "Reviewed locally"
python -m bot candidates reject CANDIDATE_ID --note "Needs a different angle"
python -m bot candidates regenerate REQUEST_ID
```

The complete lifecycle is:

```text
generate -> candidates -> one approval -> draft action -> explicit schedule -> worker publication
```

Approval creates one unscheduled `DRAFT` social action. Schedule it explicitly, then run the publishing queue:

```bash
python -m bot schedule SOCIAL_ACTION_ID --at 2030-01-01T12:00:00Z
python -m bot worker --once --queue publishing
```

The existing `create`, `preview`, `approve`, `cancel`, `execute`, `list`, and `import-json` commands remain for hand-authored actions. Generated drafts use the authoritative approved-candidate link at scheduling time; action metadata is not proof of approval. Unattended approval is separate from unattended publishing, and `allow_unattended_publishing` remains false in the safe example configuration.

Future generation creates a scheduled request without opening a browser session or calling Ollama:

```bash
python -m bot generate x-post --account main --topic "queued notes" --generate-at 2030-01-02T12:00:00Z --no-review
python -m bot worker --once --queue generation
```

## Topics

Topic discovery uses active, recent, non-quarantined examples. It groups transparent keyword overlap, requires repeated support across distinct author/source identities, caps one source's contribution, and stores support IDs, labels, recency, and counts. Engagement is only a capped tiebreaker; discovery makes no popularity or outcome prediction.

```bash
python -m bot topics discover --platform reddit --since-hours 720
python -m bot topics discover --platform x
python -m bot topics list --platform reddit
python -m bot topics generate TOPIC_ID --action-type x-post --account main --no-review
```

Topic generation uses the same request, validation, approval, candidate, and scheduling path. Reddit posts still require `--subreddit`; every comment/reply still requires `--target`.

## Worker queues

The persistent worker supervises independent publishing and scheduled-generation queues. `all` is the default, and `--once` applies to the selected queue:

```bash
python -m bot worker --once --queue all
python -m bot worker --once --queue publishing
python -m bot worker --once --queue generation
```

Generation claims never claim social actions, publishing claims never claim generation requests, and a global pause blocks both. `dry_run` blocks publication only; local generation and permitted read-only collection can still produce pending candidates/drafts. Ctrl-C and SIGTERM stop both internal workers and release active leases safely.

## Autopost campaigns and systemd

The one-shot autopost command runs one configured campaign occurrence:

```bash
python -m bot autopost daily-x
python -m bot autopost weekly-reddit
```

It is intentionally configuration-only: the command accepts only a campaign ID. Before it can generate or open browser-backed publishing resources, the campaign, account, global pause, dry-run mode, content_generation.enabled, unattended-approval, and unattended-publishing gates must all allow the operation. The checked-in campaigns are safe examples: dry-run and both unattended capabilities remain disabled. Set content_generation.enabled=true only in a reviewed local configuration when Ollama is available.

Install the example Linux user units from the repository checkout:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/stealth-autopost@.service.example ~/.config/systemd/user/stealth-autopost@.service
cp deploy/systemd/stealth-autopost-daily-x.timer.example ~/.config/systemd/user/stealth-autopost-daily-x.timer
cp deploy/systemd/stealth-autopost-weekly-reddit.timer.example ~/.config/systemd/user/stealth-autopost-weekly-reddit.timer
systemctl --user daemon-reload
systemctl --user enable --now stealth-autopost-daily-x.timer
systemctl --user enable --now stealth-autopost-weekly-reddit.timer
systemctl --user list-timers 'stealth-autopost*'
journalctl --user -u 'stealth-autopost@daily-x.service'
```

The service example uses %h/stealth-social for WorkingDirectory, configuration, and the virtualenv. Change every %h/stealth-social path if the checkout lives elsewhere. Install it as the Linux user who owns the authenticated browser profiles and configuration; do not run it as root. If the machine must run user timers without an interactive login, an administrator can perform the one-time operation loginctl enable-linger USERNAME.

Use a headless-capable browser configuration and authenticate each configured browser profile manually before enabling a timer. Confirm the same saved profiles work in the unattended environment. Start Ollama at boot and install the configured model before enabling generation. Keep config.json and any environment files readable only by the owning user (for example, chmod 600 config.json). The service uses bounded restart and timeout settings; status 75 is retryable, while statuses 2 and 3 are not automatically retried.

Exit statuses are stable: 0 means published or safely skipped by cooldown, 2 means invalid configuration or a disabled policy capability, 3 means manual attention is required, and 75 means a bounded temporary failure. For status 3, inspect the JSON result and persisted request/action state, then check the service journal and re-authenticate or resolve the named safety condition before rerunning. A retry never creates replacement content while resumable persisted work exists.

Operator rollout checklist:

- [ ] Replace example campaign topics and instructions with real content policy.
- [ ] Choose the X and Reddit `OnCalendar` schedules.
- [ ] Authenticate each browser profile once on the Linux laptop.
- [ ] Confirm each saved profile works in the unattended headless environment.
- [ ] Configure Ollama to start at boot and confirm the selected model is installed.
- [ ] Complete the existing manual dry-run generation and publishing checks.
- [ ] Back up and confirm the intended SQLite database path.
- [ ] Set `dry_run=false` only after the dry-run checks succeed.
- [ ] Enable unattended approval and unattended publishing explicitly.
- [ ] Install and enable one low-frequency timer first.
- [ ] Inspect its published URL, persisted records, and journal output.
- [ ] Enable the remaining campaign timers after the first campaign is verified.
- [ ] Define an operator routine for exit status `3`, paused accounts, and expired sessions.

Deferred enhancements:

- [ ] Add a manual content queue as an alternative campaign source.
- [ ] Add discovered-topic selection as an alternative campaign source.
- [ ] Add external notifications for manual-attention outcomes.
- [ ] Add X media publication after the existing media support work is complete.
- [ ] Add X replies only with their existing target and policy constraints.
- [ ] Add Reddit comments and replies only with their existing target, subreddit, and policy constraints.
- [ ] Consider independent campaign concurrency only after resource and browser-session isolation is demonstrated.

## Privacy, platform obligations, and schema reset

Ollama inference stays on the configured local service, but browser automation still must follow X and Reddit rules, permissions, community rules, applicable terms, and rate limits. Local inference does not exempt automation from platform policies. Do not configure credentials or private account data in example files.

The schema is disposable. Before pre-production use, back up data and confirm the target is this repository's `data/stealth.db`, then delete it when an incompatible schema change requires a reset:

```powershell
Remove-Item -LiteralPath 'data\stealth.db'
```

```bash
rm -- data/stealth.db
```

Reset deletes drafts, schedules, action history, examples, candidates, and topics. It is unrecoverable unless backed up. The application uses `Base.metadata.create_all`; it does not migrate old incompatible databases.

## Troubleshooting and verification

- Ollama unavailable: start the local service and run `ollama status`, then `python -m bot ollama status`.
- Model missing: run `python -m bot ollama check` and follow its exact `ollama pull qwen3:8b` guidance; the bot will not pull it.
- Timeout or malformed structured output: lower the context budget or inspect the bounded safe error; the generator makes at most one structure-only repair attempt.
- Browser authentication: run `login` for the configured profile. Do not bypass a challenge or login wall.
- Challenge/rate limit: stop, follow the reported safe next action, and wait for any permitted retry interval.
- No relevant examples: generation continues with an empty persisted selection outcome; it does not invent provenance.
- No safe unattended candidate: the request remains without automatic approval and needs manual review.
- Broken Python environment: verify Python 3.12+ and dependencies before running compile/import/help checks. Unavailable browser, Ollama, schema, or runtime smoke checks must be reported as unavailable, never as passing.

Safe verification is based on compilation, imports, configuration loading, fresh disposable-schema inspection, CLI help, static safety scans, and manual dry-run scenarios. This project intentionally adds no automated tests, fixtures, snapshots, or CI workflows.

## Remaining project TODO

- [ ] Implement media uploads for X.

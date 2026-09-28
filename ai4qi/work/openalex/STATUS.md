# OpenAlex harvest — status

**Not started.** No OpenAlex API key was found in the environment (checked
for any variable with `OPENALEX` in its name, e.g. `OPENALEX_API_KEY`).

Per task instructions, the harvest requires the key be passed as the
`api_key` query parameter on every request to `https://api.openalex.org/works`.
Without it, no queries were made and `ai4qi/openalex_harvest.py` was not
written (no point authoring an untestable script).

## To unblock

Set an environment variable containing `OPENALEX` (e.g. `OPENALEX_API_KEY`)
with a valid OpenAlex API key in this session's environment, then re-run
this task.

Note: OpenAlex's public API is usable without a key (via the "polite pool"
with a `mailto` parameter, subject to lower rate limits), but the task
explicitly specifies passing a key as `api_key`, so this harvester defers
to that requirement rather than substituting an unauthenticated approach.

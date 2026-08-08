# Security policy

## Supported versions

Only the latest release line receives security fixes. Older versions are
out of scope.

| Version | Supported |
| ------- | --------- |
| 0.3.x   | yes       |
| < 0.3   | no        |

## Reporting a vulnerability

Please do not open a public issue for security problems. Email the
maintainer (see the GitHub profile of `naivestud`) with:

- a description of the issue and its impact,
- a minimal reproduction or proof of concept,
- the affected version(s).

You should receive an acknowledgement within a few days. Once the issue is
understood, a fix will be prepared and published in the next release, with
credit to the reporter unless they prefer to stay anonymous.

## Scope notes

roomkit processes local audio files and does not make network calls at
runtime; it has no privilege escalation surface beyond that of the Python
process it runs in. The most likely risks come from parsing untrusted media
files (via `soundfile`/`libsndfile`) and loading untrusted `.npz` token
bundles with `numpy.load`. Treat both as untrusted input; in particular,
never load `.npz` files from unknown sources with `allow_pickle=True`.

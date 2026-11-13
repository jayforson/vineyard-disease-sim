# Optional GitHub Actions activation

The repository publishes workflow templates under `docs/ci/templates/`, not active `.github/workflows/` files. Local tests and simulation commands work without GitHub Actions.

The GitHub CLI credential used during initial publication had repository scope but no `workflow` scope. GitHub rejected the active workflow-file push; no alternate credential route was used. Activation is a separate authorized action.

If you want Actions enabled, the account owner can authorize the scope interactively:

```bash
gh auth refresh --hostname github.com --scopes workflow
```

After authorization, copy the templates into `.github/workflows/` on a review branch and publish that change. The unit template runs pytest on pushes/pull requests. The heavier simulation template is `workflow_dispatch` only and builds/renders the container; it is not launched automatically.

No hosted CI result is claimed for the initial prototype. See `docs/VERIFICATION.md` for local checks actually completed.

# mash-site — design

Status: approved by user. No code has been written yet.

## What

A single static marketing page for the two just-published open-source projects, telling
the extraction story: book-mash → mash-core. One combined site, not two — the natural
narrative is "we built book-mash, found the judge-core was reusable, extracted
mash-core," and splitting it into two sites would fragment that story for no benefit.

## Why

Both repos are fresh (v0.1.0, zero stars, no announcement) and currently have no web
presence beyond their GitHub READMEs. A short, deployed page gives the marketing push
(GitHub topics + awesome-list PRs, already done this session) something concrete to
link to besides bare repo URLs, and is a natural companion to the planned "why we
extracted mash-core" writeup.

## Approach

Plain static HTML + CSS, no framework, no build step. This is a one-page site with
entirely static content (project descriptions, install snippets, links) — a framework
(Next.js, Astro) would add dependency management and a build pipeline for zero
functional benefit at this scope. Responsive layout is CSS grid/flexbox with a single
mobile breakpoint; no JavaScript is required.

## Structure

New sibling repo, `/Users/timur_isachenko/Dev/LifeOS/mash-site/`, matching the pattern
already established for book-mash and mash-core (own git repo, MIT license, pushed to
GitHub).

```
mash-site/
  index.html
  style.css
  LICENSE
  README.md
```

## Page content (single page, top to bottom)

1. **Hero** — the story in 2-3 sentences: editorial review doesn't scale past a few
   read-throughs; book-mash was built to measure prose against six judge dimensions;
   the judge-core underneath turned out to be generic, so it was extracted into
   mash-core for reuse. Links to both GitHub repos.
2. **book-mash section** — what it does (six judges: humanness, voice, usefulness,
   evidence density, claim defensibility, redundancy), the install snippet from its
   README, link to the repo.
3. **mash-core section** — what it does (JudgeDim/JudgeScore contract, provider-agnostic
   model routing, retry-with-backoff), the install snippet (path-dependency note, since
   it isn't on PyPI yet), link to the repo.
4. **Footer** — MIT license note, author link.

Content is drawn from the existing READMEs — no new copywriting invented beyond
connective narrative sentences in the hero.

## Responsive behavior

Single breakpoint at 640px: sections stack vertically on narrow viewports (already the
natural flow for a single-column page), font sizes and section padding scale down,
no layout that requires JS to reflow.

## Deploy

Vercel CLI (`vercel --prod`) against the static directory — no framework preset needed,
Vercel serves the directory as-is. Repo is also pushed to GitHub, so a git-connected
auto-deploy can be wired up later if desired; not set up as part of this pass (CLI
deploy now is faster and doesn't require dashboard configuration).

## Non-goals

- No JavaScript framework, no build step, no bundler.
- No analytics/tracking wired up in this pass.
- No custom domain configuration — ships on Vercel's default `*.vercel.app` URL unless
  requested otherwise.
- No blog/writeup content on this page — the planned "why we extracted mash-core"
  writeup is a separate piece of content (marketing idea #2), not part of this site.

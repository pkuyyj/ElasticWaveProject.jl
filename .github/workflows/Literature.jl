name: Run Literate.jl
# adapted from https://lannonbr.com/blog/2019-12-09-git-commit-in-actions

on: push

jobs:
  lit:
    runs-on: ubuntu-latest
    steps:
      # Checkout the branch
      - uses: actions/checkout@v2

      - uses: julia-actions/setup-julia@v1
        with:
          version: '1.9'
          arch: x64

      - uses: julia-actions/cache@v1
      - uses: julia-actions/julia-buildpkg@latest

      - name: run Literate
        run: QT_QPA_PLATFORM=offscreen julia --color=yes --project -e 'cd("src/"); include("literate-script.jl")'; cd -

      - name: setup git config
        run: |
          # setup the username and email. I tend to use 'GitHub Actions Bot' with no email by default
          git config user.name "GitHub Actions Bot"
          git config user.email "<>"

      - name: commit
        run: |
          # Stage the file, commit and push
          git add PorousConvection/docs/*
          git commit -m "Commit markdown files fom Literate"
          git push
# CausalLens

CausalLens is a local-first causal inference lab for estimating the effect of a binary intervention from tabular data. It uses a doubly robust estimator, propensity overlap diagnostics, inverse-probability balance checks, and a bootstrap interval over estimated influence values.

This is a learning and analysis tool. It does not establish causality from observational data by itself. Results depend on study design, measured confounders, treatment consistency, and input quality.

## Quick start with Docker

Build the analysis image:

```sh
docker build -t causallens .
```

Run the included synthetic example:

```sh
mkdir -p reports
docker run --rm -v "$PWD/samples:/data:ro" -v "$PWD/reports:/reports" causallens /data/synthetic_effect.csv --treatment treatment --outcome outcome --output /reports/report.json
```

The sample is synthetic and its known generating effect is documented in `samples/README.md`. It is a demonstration fixture, not a benchmark.

## Run locally

Requires Python 3.11+ and the packages in `requirements.txt`:

```sh
python3 -m pip install -r requirements.txt
python3 causallens.py samples/synthetic_effect.csv --treatment treatment --outcome outcome --output reports/report.json
```

Feature columns are inferred as every numeric column except treatment and outcome. Override that behavior with `--features age,income,prior_score`.

## Method

1. Fit a logistic propensity model for treatment assignment.
2. Fit separate ridge outcome models for treated and control observations.
3. Combine both pieces into a doubly robust pseudo-outcome.
4. Report its mean, a bootstrap interval, propensity overlap, and standardized mean differences before and after weighting.
5. Add an illustrative omitted-confounding stress table. The stress table is a scenario tool, not a formal sensitivity bound.

## Input contract

- `treatment`: binary 0/1 column
- `outcome`: numeric column
- features: numeric columns with missing values allowed; medians are used for model fitting

The report includes warnings for small groups, poor overlap, and unbalanced covariates. Inspect those warnings before interpreting an estimate.

## License

MIT. See [LICENSE](LICENSE).

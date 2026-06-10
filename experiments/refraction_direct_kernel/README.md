# Direct refraction kernel

This folder is an independent experiment for the A-type refraction destination
layer. It is deliberately separated from `main_ou_medium_response`.

The main model estimates medium response proportions:

```text
F_external = F_refraction + F_reflection + F_absorption
```

This experiment starts after that response layer and estimates where the
refraction part goes:

```text
P(D | O, U, refraction)
```

## Method

Only A-type records are used:

```text
O != U, D != O, D != U
```

The model is a direct destination scoring kernel:

```text
score(D | O, U) =
  target cross-section
  - U-D / O-D distance attenuation
  + geographic channel terms
  + destination level and socioeconomic terms
```

Training uses negative-sampled legal destination candidates. Evaluation
normalizes scores over all legal D cities for each O-U beam using softmax, so
the predicted refraction distribution sums to one within each beam.

## Run

```powershell
D:\APP\programming\Anaconda\python.exe .\run_direct_refraction_kernel.py --negatives 3 --epochs 1
```

Outputs are written to `outputs/`.

## Current first run

The first runnable version uses 3 negative destinations per positive O-U-D row
and 1 training epoch. It should be treated as a baseline direct kernel, not as
the final refraction model.

Key test metrics from the first run:

- `test_top1_dominant_accuracy`: 0.2948
- `test_top10_dominant_coverage`: 0.7915
- `test_weighted_tv_distance`: 0.6228
- observed mean U-D distance: 622.63 km
- predicted mean U-D distance: 821.27 km
- observed mean D level: 3.88
- predicted mean D level: 5.11

Interpretation: after removing the direction-change feature and completing
manual coordinates for Qianjiang and Wanning, the direct kernel covers the full
A-type flow. It can identify many plausible destinations in the top 10, but its
top-1 prediction is still weak and it now over-predicts higher-level /
longer-distance destinations. This is useful evidence for trying the later
two-stage layered model: first choose the socioeconomic height layer, then
choose the geographic landing point inside that layer.

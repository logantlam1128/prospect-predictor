# MLB Prospect Career Tier Predictor

A PyTorch neural network that projects minor league baseball prospects into career tiers, from never reaching the majors to Hall of Fame caliber, based on their minor league performance.

> **Status: in progress.** The full modeling pipeline is built and tested on synthetic data. Integration of real minor league and MLB data is the next phase (see [Roadmap](#roadmap)).

## The problem

Most prospects never reach the majors, and only a handful become stars. This project asks: given a prospect's most recent minor league season, how likely is he to become a fringe major leaguer, a starter, an All-Star, or more?

Each player is assigned one of six tiers based on his MLB career:

| Tier | Meaning |
|---|---|
| 0 | Never reaches MLB |
| 1 | Fringe / replacement-level major leaguer |
| 2 | MLB starter |
| 3 | All-Star caliber |
| 4 | All-MLB caliber |
| 5 | Hall of Fame caliber |

Tiers are defined by WAR (wins above replacement) thresholds rather than awards, so labels reflect performance rather than popularity or voting.

## Approach

**One row per player per minor league season.** Each season is a snapshot labeled with the player's eventual tier, so the model learns what future stars look like at every stage of development, from Rookie ball to Triple-A. Predicting a current prospect uses his most recent season in the same format.

**Ordinal outputs.** Instead of six unrelated classes, the network answers five ordered questions: "Is this player above tier 0? Above tier 1? ... Above tier 4?" This teaches the model that tiers have an order, so its mistakes tend to be off by one tier rather than wildly wrong.

**Masked loss for unfinished careers.** An active player's current tier is a lower bound, since he can still improve. Rather than discarding active players, the loss only grades the answers that are already known (the tiers he has reached) and masks out the rest. Finished careers are graded on all five questions.

**Leakage-safe evaluation.** Data is split into train, validation, and test sets **by player**, so no player's seasons appear in more than one set. Features are standardized using statistics from the training set only.

**Class imbalance.** Most prospects never reach the majors. An optional class-weighted loss up-weights rare "yes" answers so the model is less likely to default to predicting the most common tier.

## Model

```
14 inputs -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(5) -> 5 logits
```

Inputs: age, minor league level, strikeout rate, isolated power, walk rate, BABIP, and position (as share of games at each position). Trained with Adam and a masked binary cross-entropy loss.

## Current results (synthetic data)

The pipeline is currently validated on a synthetic dataset of 1,000 generated players, built so the true relationships between stats and tiers are known. These results test that the pipeline works; they are **not** claims about real prospect prediction.

| Model | Exact tier accuracy | Mean tier error |
|---|---|---|
| Always predict tier 0 | 0.685 | 0.510 |
| Logistic regression | 0.756 | 0.275 |
| Neural network (unweighted) | 0.754 | 0.264 |
| Neural network (class-weighted) | 0.710 | 0.337 |

Takeaways so far:
- Both learned models clearly beat the majority-class baseline.
- The network matches logistic regression on accuracy but makes smaller mistakes, consistent with its ordinal design. Because the synthetic data has linear structure, a large gap over logistic regression isn't expected here; the real-data comparison is the meaningful test.
- Class weighting shifts predictions upward, catching more upper-tier players at the cost of overall accuracy, a tradeoff that depends on how the tool is used.

## Project structure

```
src/
  fake_data.py    # generates the synthetic player-season dataset
  dataset.py      # loading, player-level split, standardization, Dataset/DataLoader
  model.py        # the neural network
  targets.py      # ordinal targets, masks, masked loss, class weights
  train.py        # training loop, saves the trained model
  evaluate.py     # accuracy, mean tier error, confusion matrix
  baselines.py    # majority-class and logistic regression baselines
```

## How to run

```
pip install -r requirements.txt
python src/fake_data.py     # create the synthetic dataset
python src/train.py         # train and save the model
python src/evaluate.py      # evaluate on the validation set
python src/baselines.py     # compare against baselines
```

## Roadmap

- [ ] Build a real data pipeline: minor league stats and MLB WAR-based career tier labels
- [ ] Add stronger features: age relative to level, league-adjusted stats, year-over-year trends, prospect rankings
- [ ] Add a star-recall metric that directly measures how often future stars are identified
- [ ] Early stopping and regularization (dropout, weight decay)
- [ ] Consistent ordinal probabilities (CORAL-style shared outputs)
- [ ] Player-level evaluation using each player's most recent season
- [ ] Interactive Streamlit app: prospect lookup, what-if sliders, and historical comparisons
- [ ] Final evaluation on the held-out test set

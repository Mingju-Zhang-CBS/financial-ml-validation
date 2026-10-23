# Interview discussion guide

## Two-minute explanation

The question is whether a financial classification improvement survives a separate evaluation stage. I used a public task definition to build an independent, AI-assisted study with three classifiers and three fixed feature ablations. The runnable demonstration uses synthetic panel data. Date groups stay together, preprocessing learns only from training rows, and the final choice is determined before confirmation scores are evaluated. The most useful outcome is a clear statement of what the validation protocol does and does not establish.

## Questions to be ready to answer

1. **Why not shuffle individual rows?** Allocations on the same date share market information. Mixing them can overstate the number of independent observations. Whole-date isolation addresses that dependence, although it does not solve all temporal dependence.
2. **Why do anonymous dates need a different protocol?** Their suffixes are not a calendar. A hash-based date split is reproducible but cannot support a forward-performance claim. Only the synthetic generator supplies a known order here.
3. **Why purge 20 dates in the synthetic study?** Inputs contain 20 observations of history and a next-period outcome. The purge separates histories across known chronological boundaries. The exact gap should follow the information horizon and data dependence, rather than being a universal constant.
4. **Where can preprocessing leak?** Medians, scales and category vocabularies can learn about validation distributions if fitted globally. The ColumnTransformer is inside each estimator pipeline and fits only training rows.
5. **How are features and labels aligned?** Unique ROW_ID sets must match exactly; labels are joined to a canonical row order. Permutation and duplicate/missing-key tests exercise the contract. The downloaded lowercase `target` spelling is accepted explicitly, while ambiguous target columns are rejected.
6. **What is selected and what is fixed?** The model/ablation pair is selected from nine predeclared candidates. Parameters and the probability threshold are fixed. Confirmation is not used for fallback or retuning.
7. **What does the bootstrap interval mean?** Complete dates are sampled with replacement; candidate and baseline remain paired. It measures descriptive uncertainty under that resampling model, not all serial dependence or the full selection history.
8. **Why not quote Sharpe?** Classification accuracy does not specify investable position sizes, returns, costs or execution. It is not a P&L statistic. An additional trading model would be needed.
9. **What if a complex model loses?** Retain the result, inspect error slices as diagnostics and design a genuinely new evaluation before tuning further. Reopening confirmation to improve its score changes its role.
10. **What did you contribute?** Review the independently written package, schema checks, protocol, tests and reporting; disclose AI assistance. Do not claim authorship of QRT's task, library estimators or supplied course/company frameworks.

## Suggested live demonstration

Run the tests; show keyed-row permutation invariance and the training-only imputer check. Open the model comparison, then the confirmation interval. Explain why the selected model need not win on confirmation and why the synthetic example carries no leaderboard claim.

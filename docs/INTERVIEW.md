# Interview guide: a baseline must be taken seriously

## A 90-second explanation

“I compared probability classifiers for future Treasury-yield direction using official public-domain H.15 data. I separated the feature observation, publication allowance, decision and both future target endpoints; then purged training labels by when their outcomes were actually available. Three expanding validation windows compared nine fixed model/feature configurations and two simple probability baselines. The baseline already led development. CatBoost was the best classifier comparator, but its confirmation probability loss was worse. Logistic regression looked better in confirmation, with block-sensitive uncertainty, so I kept the frozen selection and published the qualified result rather than switching winners. The main contribution is defensible experiment design, financial data judgment and reproducible evidence.”

## Questions to prepare

1. **Why yields instead of ETF returns?** The selected official government series have a clear source-specific reuse policy and known measurement dates. They support a rates-classification question, not a claim about bond total returns.
2. **What is the subtle target mistake?** Starting the target at the stale feature quote would include pre-decision movement. This label starts with the later measurement on the future decision day.
3. **Why not just purge twenty rows?** A training label needs the future endpoint and its publication allowance. Eligibility is tested from timestamps, including the development-selection/confirmation boundary.
4. **Are overlapping feature windows leakage?** Shared known past observations can be legitimate. Unavailable future training labels are leakage; dependence among test outcomes is a separate inference problem.
5. **Why does the baseline's accuracy fall below 50%?** It uses the training class tendency. Confirmation's class balance changed; it cannot learn that future majority in advance.
6. **Why keep a classifier that loses development to baseline?** It is the fixed research comparator, not a deployment recommendation. Both the classifier-only winner and the overall development leader are recorded.
7. **Why not select logistic after confirmation?** That would use the evaluation period for another search. Its two block intervals also differ; the honest conclusion is conditional and sensitive, not universal significance.
8. **What would a real trading study require?** A specified bond or instrument, price/coupon/carry/funding conventions, duration/convexity, market quotes, costs and a fresh prospective evaluation.
9. **What residual leakage risk remains?** Current-vintage revisions and missing historical release archives. A conservative delay is not an archived point-in-time dataset.

For QR emphasize information boundaries, baseline quality and dependent uncertainty. For DS emphasize the keyed data contract, train-only transformations, calibration and reproducibility. For rates/markets roles connect a yield change to risk mechanics without treating classification accuracy as desk P&L.

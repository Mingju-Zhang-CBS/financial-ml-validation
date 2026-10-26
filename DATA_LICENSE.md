# Data license, attribution and exclusions

## Selected public-domain government series

The packaged CSV contains dates and five U.S. Treasury nominal constant-maturity yields: 3 months, 1 year, 2 years, 5 years and 10 years. Source: **Board of Governors of the Federal Reserve System (US), H.15 Selected Interest Rates; underlying Treasury constant-maturity yields: U.S. Treasury**.

The [Board's copyright/reuse policy](https://www.federalreserve.gov/disclaimer.htm) states that, unless an exception is indicated, information on its website is in the public domain and may be copied and distributed, with the Board cited as source. The [H.15 release](https://www.federalreserve.gov/releases/h15/) identifies these nominal government-yield series and their Treasury source. No third-party copyright exception is identified for the five selected series. This is a source-specific assessment, not a claim that every financial dataset or every linked website is public domain.

The [official download page](https://www.federalreserve.gov/datadownload/Choose.aspx?rel=H15) provides the Treasury Constant Maturities CSV package. [source.json](ml_validation/datasets/source.json) records the exact download URL, original response SHA-256, retrieval time, selected series identifiers, units, normalized snapshot hash and filtering rules. The snapshot covers 2 January 1990–31 December 2025. It contains 9,005 dates with an observed 10-year yield; 388 missing/non-observation dates are excluded, and no dates or yields are filled in. All five tenors are present on the retained dates.

## Scope

- Distributed: numerical government yield observations, minimal identifiers/attribution and independently authored code, analysis and figures.
- Excluded: government seals/logos, website HTML, full policy text, proprietary commercial-paper series, vendor security databases, original coursework, employer data and challenge files.
- Code and original authored material: [MIT](LICENSE). The code license does not relicense government marks or third-party material. This research is independent and carries no government endorsement.

## Vintage and economic meaning

This October 2026 retrieval is a **current historical vintage**. It is not a release-by-release archive and may contain revisions to historical values. Exact historical publication times are not reconstructed. The current H.15 schedule informs a conservative publication allowance, which is documented as an assumption. A lag cannot repair unknown historical revisions.

The observations are interpolated constant-maturity reference yields, not transaction prices or total returns. The [Treasury methodology](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/treasury-yield-curve-methodology) changed on 6 December 2021; pre-change official values are retained. The snapshot is not backfilled into a single constant methodology. Any future refresh requires a new rights/source record, fingerprint and protocol; do not silently update data beneath old results.

The Board has announced changes to its Data Download Program. Checked-in data support offline reproduction even if the download interface later changes. See the original source URLs and retrieval record before designing a new collection route.

## Separate licensed equity extension

The equity workflow accepts the reader's own CRSP CIZ extract and French factor archive. Its bundled fixture is generated locally and is not CRSP data. Institutional WRDS access is distinct from rights to redistribute records or derived results. Licensed inputs, security-level predictions and real-data aggregate outputs remain private pending the applicable rights assessment; none is bundled with the public fixture. Current historical factors do not provide a release-by-release archive. See [Equity Research](docs/EQUITY_RESEARCH.md), [WRDS policy](https://wrds-www.wharton.upenn.edu/pages/about/data-download-and-analysis-policy/) and the [French library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). The Treasury reuse assessment above does not apply to these separate sources.

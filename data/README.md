# Benchmark Data

The current study uses all original rows of four credit-data representations.
German Numeric and South German derive from the same underlying German data.

| Representation | Local filename | Rows | Positive outcome |
| --- | --- | ---: | --- |
| German Numeric | `german-data-numeric` | 1000 | Original class 2 |
| South German | `SouthGermanCredit.asc` | 1000 | `kredit == 0` |
| HMEQ | `hmeq.csv` | 5960 | `BAD == 1` |
| Taiwan | `default of credit card clients.xls` | 30000 | Next-month default 1 |

The frozen acquisition implementation is `code/download_data.py`. Its default
selection is these four representations. Acquisition is a separate network
operation; it is not required for the aggregate-statistics replay.

```bash
python code/download_data.py --datasets german_numeric south_german hmeq taiwan
```

The German and Taiwan sources are their UCI distributions. HMEQ uses the
[fixed SAS repository file](https://github.com/sassoftware/sas-viya-dmml-pipelines/blob/cf9f13a0ce15c8968c879e9e6dcba11deeff200b/data/hmeq.csv).
The acquisition script records source URLs and hashes. Dataset files are kept
in the local download directory and remain subject to their providers' terms.

All data-dependent imputation, encoding, scaling, PCA and risk estimation are
fitted within their relevant training subset. The final repeated-holdout
protocols, rather than older dataset-specific runners, define the current
splits and model-selection procedure.

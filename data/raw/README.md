# Raw data source

- Dataset: California Housing
- Original URL: <https://ndownloader.figshare.com/files/5976036>
- Original archive: `cal_housing.tgz`
- Original SHA256: `aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681`
- Prepared CSV: `../california_housing.csv`

The official archive was converted with
`sklearn.datasets.fetch_california_housing`. Scikit-learn stores the cached
prepared array as `cal_housing_py3.pkz`; the original archive is no longer
needed after conversion.

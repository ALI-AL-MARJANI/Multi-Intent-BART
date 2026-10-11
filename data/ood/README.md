# Out-of-distribution evaluation data

`clinc150_oos_test.txt` holds the 1,000 out-of-scope test queries of CLINC150
(`data_full.json`, key `oos_test`), one per line, unmodified. It is produced by
`scripts/prepare_ood_data.py`.

Source: Larson et al., "An Evaluation Dataset for Intent Classification and
Out-of-Scope Prediction", EMNLP 2019, <https://github.com/clinc/oos-eval>,
licensed CC BY 3.0.

At load time `gemis.ood.data.load_ood_utterances` lowercases the text, removes
sentence punctuation and deduplicates, so that the queries match the
formatting of MixATIS / MixSNIPS.

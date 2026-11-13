# Data access and privacy

The raw questionnaire workbook is **not included in this public repository**. It contains direct identifiers (email addresses), mental-health responses, and records from participants younger than 18 years. Publishing the raw workbook on GitHub would create avoidable privacy and re-identification risks.

The public repository therefore contains only:

- analysis code;
- model-selection result tables;
- model metadata and integrity hashes;
- a data dictionary; and
- reproducibility instructions.

For exact reruns, authorized researchers should place the original private workbook locally and pass its path through the `--data` argument. The repository `.gitignore` excludes raw anxiety workbooks and `data/raw/`.

## Recommended data-availability wording

> Participant-level data are not publicly deposited because the source dataset contains personal identifiers and sensitive mental-health information. De-identified data or a controlled-access extract may be made available to qualified researchers subject to applicable ethical approval, participant-consent constraints, and institutional data-governance requirements. The analysis code, model-selection documentation, and reproducibility metadata are publicly available in the accompanying repository.

Do not create or publish a participant-level de-identified file unless the ethics approval and consent language explicitly permit public release. Removing email alone does not eliminate re-identification risk because age, sex, education, employment, and mental-health outcome remain potentially identifying in combination.

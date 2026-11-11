# FMA metadata attribution

The real-track catalog is derived from:

Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, and Xavier Bresson.
**FMA: A Dataset For Music Analysis.** International Society for Music Information Retrieval Conference (ISMIR), 2017.

- Dataset documentation: https://github.com/mdeff/fma
- Paper: https://arxiv.org/abs/1612.01840
- Metadata license: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
- Official metadata archive: https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
- Published archive SHA-1: `f0df49ffe5f2a6008d7dc83c6915b31835dfe733`

Changes: joined track and artist metadata with published Echo Nest tempo and energy fields; filtered incomplete/out-of-range rows; selected a deterministic genre-balanced subset; added provenance fields and an explicit `unknown` mood. Artist and track names remain as published. The adjacent manifest records the exact selection rules and counts.

This attribution applies to metadata. It does not replace the individual licenses of recordings. No recordings, album artwork, or previews are bundled. A source link can refer to the dataset rather than a currently playable track page. Dataset entries are historical and link availability is not guaranteed.

Echo Nest energy is a published proprietary audio descriptor, not a measurement of RMS amplitude. The imported metadata does not provide validated mood labels. Uploaded-audio features must not be compared directly to Echo Nest energy without a justified mapping or consistent re-extraction.

`development_catalog.csv` remains entirely fictional and is retained only for isolated software tests. It is excluded from real-data evaluation.

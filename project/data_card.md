# Data card

**Source:** [FMA](https://github.com/mdeff/fma), FMA-small audio and `tracks.csv` metadata from its official download links.

FMA-small contains 8,000 30-second music clips in eight balanced genres: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop and Rock. Clips can include vocals. Genre labels come from the dataset; this project does not independently verify every label.

This first experiment samples 50 training, 10 validation and 10 test tracks per genre, with seed 42, from the original `set.split` assignments within `set.subset == small`. `results/manifest.csv` records all 560 selected IDs and original splits. Track IDs are unique. Artist IDs have no overlaps between these three pilot splits.

The middle five seconds of each file are decoded to mono float32 and resampled to 22,050 Hz. Short files are zero-padded. The STFT uses a 1,024-sample Hann window, a 512-sample hop and centered frames; power is mapped to 64 triangular HTK mel bands, then log10 with a floor of 1e-10. The output is 1 × 64 × 216. Mean and standard deviation are computed on training spectrograms only.

Audio ZIP members are checked against the member size and CRC from the official archive. Downloads and feature caches are ignored by Git. The subset selection is deterministic and independent of predictions or performance.

FMA audio has track-specific Creative Commons licenses; consult each track's metadata before redistributing or reusing it. The FMA repository documents the separate licenses for metadata and code. Downloaded audio is not included in the repository.

This small balanced pilot does not represent a user's entire music collection. Mixed genres, silence, speech, unknown genres and other recording styles can produce misleading predictions. The test has only 80 tracks; its metrics are preliminary.

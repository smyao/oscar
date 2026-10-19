# GPQA Diamond calibration input

`gpqa_diamond.csv.gz` contains the **unmodified CSV bytes**, losslessly compressed
to avoid publishing benchmark examples as plain text. This is calibration input,
not a generated rotation checkpoint and not a benchmark quality result.

- Source: [OpenAI simple-evals GPQA Diamond CSV](https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv), retrieved 2026-10-19 over verified HTTPS.
- Dataset: [GPQA](https://github.com/idavidrein/gpqa), by David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman.
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/), as declared by the [dataset authors](https://huggingface.co/datasets/Idavidrein/gpqa).
- Paper: [GPQA: A Graduate-Level Google-Proof Q&A Benchmark](https://arxiv.org/abs/2311.12022).
- CSV SHA256: `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`.
- Uncompressed bytes: 1,373,492; question rows: 198.

The default preparation path verifies the CSV SHA256 before model loading and
records it in the capture manifest and rotation cache identity. It does not
download data or read an unverified global Hugging Face data cache. The official
seed-0 option shuffle, chat template, whole-prompt selection and 30,000-token
budget remain unchanged. Explicit `rotation_calibration_prompts` still selects
the caller's local CSV/JSONL and binds its own SHA256.

Please preserve the authors' request not to reveal dataset examples in plain
text or images online, to reduce contamination of future training corpora.

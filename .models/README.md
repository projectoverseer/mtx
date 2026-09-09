# Local model weights

Not in git (see `.gitignore`).  `faster-whisper-small` lives here because the
HuggingFace download from this machine resets more often than it succeeds:
`model.bin` came down, the three small JSON files each took several attempts,
and `preprocessor_config.json` never did (ctranslate2 does not need it).

Point mtx at it with `MTX_WHISPER_MODEL`, which `mtx.env` already does.

To rebuild:

    for f in config.json model.bin tokenizer.json vocabulary.txt; do
      curl -sSLf --retry 5 --retry-all-errors -O \
        "https://huggingface.co/Systran/faster-whisper-small/resolve/main/$f"
    done

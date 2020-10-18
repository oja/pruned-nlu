# Accelerating NLU in Task-Oriented Dialog

Code for ACL ConvAI workshop paper "Accelerating Natural Language Understanding in Task-Oriented Dialog".

## Requirements
- \>= Python 3.7
- \>= PyTorch 1.4.0
- [seqeval](https://github.com/chakki-works/seqeval)

## Setup
Download GloVe embeddings:

```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d glove
```

## Files
- `train.py`, `test.py`, `distill.py`, `timer.py`, `prune.py`: runnable scripts, check each file's argparse for options and details
- `models.py`: intent detection, slot-filling, and multi-task (joint intent detection and slot filling) CNN models
- `dataset.py`: dataset loading abstractions
- `util.py`: common code
- `models/`: pretrained models, 5 duplicates of each
- `datasets/`: prepared ATIS and Snips datasets

## Citation
```bibtex
@inproceedings{ahuja-desai-2020-accelerating,
    title = "Accelerating Natural Language Understanding in Task-Oriented Dialog",
    author = "Ahuja, Ojas  and
      Desai, Shrey",
    booktitle = "Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI",
    year = "2020",
    publisher = "Association for Computational Linguistics"
}
```

## Contact
[ojas@utexas.edu](mailto:ojas@utexas.edu)

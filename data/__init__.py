from .Squad import SquadDataset
from .NaturalInstructions import NIDataset
from .SuperNI import SuperNIDataset
from .Xsum import XsumDataset
from .LongBench import *

TRAIN_DATASETS = ["SQuAD", "Natural-Instructions", "XSum", "Super-Natural-Instructions"]
TEST_DATASETS = {
    "hotpotqa": HotpotQADataset,
    "multi_news": MultiNewsDataset,
    "samsum": SAMSumDataset,
    "musique": MuSiQueDataset,
    "2wikimqa": WikiMQADataset,
}
DATASET_NAME_PROJ = {
    "hotpotqa": "HotpotQA",
    "multi_news": "MultiNews",
    "samsum": "SAMSum",
    "musique": "MuSiQue",
    "2wikimqa": "2WikiMQA",
}

DATASET_DICT = {
    "SQuAD": SquadDataset,
    "Natural-Instructions": NIDataset,
    "XSum": XsumDataset,
    "Super-Natural-Instructions": SuperNIDataset
}
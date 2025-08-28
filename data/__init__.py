from .Squad import SquadDataset
from .NaturalInstructions import NIDataset
from .Xsum import XsumDataset
from .LongBench import *

TRAIN_DATASETS = ["SQuAD", "Natural-Instructions", "XSum"]
TEST_DATASETS = [HotpotQADataset, MultiNewsDataset, SAMSumDataset,
                MuSiQueDataset, WikiMQADataset]

DATASET_DICT = {
    "SQuAD": SquadDataset,
    "Natural-Instructions": NIDataset,
    "XSum": XsumDataset
}
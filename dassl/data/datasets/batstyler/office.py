import os.path as osp
from dassl.utils import listdir_nohidden
from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase, SFDatum
import glob

@DATASET_REGISTRY.register()
class SFOffice(DatasetBase):
    dataset_dir = "OfficeHome"
    domains = ["none", "art", "clipart", "product", "real_world"]

    def __init__(self, cfg, train_data):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        test_datasets = []
        self.train_data = train_data
        self.cfg = cfg

        train = self.init_train_data()

        for domain in cfg.DATASET.TARGET_DOMAINS:
            test_datasets.append(self.read_data(self.dataset_dir, [domain], "all"))

        super().__init__(train_x=train, test=test_datasets)
    
    def read_data(self, dataset_dir, input_domains, split):
        train_data = self.train_data
        classnames = train_data["classnames"]
        def _load_data_from_directory(directory):
            folders = listdir_nohidden(directory)
            folders = sorted(folders, key=str.lower)
            items_ = []

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(directory, folder, "*.jpg"))

                for impath in impaths:
                    items_.append((impath, label))

            return items_

        items = []

        for domain, dname in enumerate(input_domains):
            if split == "all":
                train_dir = osp.join(dataset_dir, dname)
                impath_label_list = _load_data_from_directory(train_dir)

            for impath, lbl in impath_label_list:
                item = Datum(
                    impath=impath,
                    label=lbl,
                    domain=dname,
                    classname=classnames[lbl]
                )
                items.append(item)

        return items

    def init_train_data(self):
        cfg = self.cfg
        train_data = self.train_data
        items = []
        classnames = train_data["classnames"]
        n_cls = train_data["n_cls"]
        n_style = train_data["n_style"]

        for c in range(n_cls):
            for s in range(n_style):
                item = SFDatum(
                    style=s, 
                    label = c, 
                    classname=classnames[c], 
                )
                items.append(item)
        return items
from argparse import ArgumentParser

class Config(object):
    def __init__(self):
        self.model = "resnet32"
        self.num_epoch = 200
        self.batch_size = 128
        self.no_eval = False
        # data condig
        self.data_dir = "."
        # learning rate config
        self.lr = 0.1
        self.lr_decay = 0.01
        self.lr_decay_step = [160, 180]
        self.lr_warmup = True
        self.lr_warmup_method = "linear"
        self.lr_warmup_epoch = 5
        # class-balanced loss
        self.weight_decay = 2e-4  # decay for regularization loss
        self.loss_type = "softmax"  # base loss type
        self.beta = 0.9
        self.gamma = 1.0  # hyper parameter for focal loss
        # save config
        self.save_path = "."
        # checkpoint
        self.checkpoint = ""

        parser = self.setup_parser()
        args = vars(parser.parse_args())
        self.__dict__.update(args)

        if self.loss_type not in ["softmax", "sigmoid", "focal"]:
            raise ValueError("Loss type should be 'softmax' or 'sigmoid' or 'focal'.")
        if self.lr_warmup_method not in ["linear", "constant"]:
            raise ValueError("Warmup method should be 'linear' or 'constant'.")
        if self.lr_decay_step != sorted(self.lr_decay_step):
            raise ValueError("Decay steps must be sorted.")

    def setup_parser(self):
        parser = ArgumentParser()
        parser.add_argument("-model", help="resnet18/32/44/56/110/1202", default="resnet32", type=str)
        parser.add_argument("-num_epoch", help="epoch number", default=200, type=int)
        parser.add_argument("-batch_size", help="batch size", default=128, type=int)
        parser.add_argument("-no_eval", help="whether evaluating after training", action="store_true")
        parser.add_argument("-data_dir", help="data directory", default=".", type=str)
        parser.add_argument("-lr", help="value of learning rate", default=0.1, type=float)
        parser.add_argument("-lr_decay", help="decay rate for learning rate", default=1e-2, type=float)
        parser.add_argument("-lr_decay_step", help="decay epoch", default=[160, 180], type=list)
        parser.add_argument("-lr_warmup", action="store_true")
        parser.add_argument("-lr_warmup_method", help="linear/constant", default="linear", type=str)
        parser.add_argument("-lr_warmup_epoch", help="wramup duration", default=5, type=int)
        parser.add_argument("-weight_decay", help="decay rate for regularization loss", default=2e-4, type=float)
        parser.add_argument("-loss_type", help="softmax/sigmoid/focal", default="softmax", type=str)
        parser.add_argument("-beta", help="beta for class-balanced loss", default=0, type=float)
        parser.add_argument("-gamma", help="gamma for focal loss", default=1.0, type=float)
        parser.add_argument("-save_path", help="save file path", default=".", type=str)
        parser.add_argument("-checkpoint", help="checkpoint path", default="", type=str)
        return parser
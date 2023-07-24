from libs.HyperParameters import get_hparams
from libs.model import get_model
from libs.optimizer import get_optim
from libs.dataset import HPA
from libs.trainer import Trainer
from libs.loss import get_loss_func


def main():

    hparams = get_hparams()
    model = get_model(hparams)
    optimizer = get_optim(model, **hparams.train.optimizer)
    train_dataset = HPA(**hparams.train.dataset)
    val_dataset = HPA(**hparams.val.dataset)
    loss_func = get_loss_func(**hparams.train.loss)

    trainer = Trainer(model, optimizer, loss_func, train_dataset, val_dataset, hparams)
    trainer.run()

if __name__ =='__main__':
    main()


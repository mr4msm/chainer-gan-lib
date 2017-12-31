import argparse
import os
import sys

import chainer
from chainer import training
from chainer.training import extension
from chainer.training import extensions

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.dirname(__file__)) +
                os.path.sep + os.path.pardir)

from evaluation import sample_generate, sample_generate_light, calc_inception, calc_FID
from common.record import record_setting

from net import Discriminator, Generator
from updater import Updater

from common.misc import copy_param

import numpy as np
import chainer.functions as F
from chainer import Variable

from chainer.datasets.image_dataset import ImageDataset


class MyImageDataset(ImageDataset):

    def __init__(self, paths, flip=False, root='.', dtype=np.float32):
        super(MyImageDataset, self).__init__(
            paths, root=root, dtype=dtype
        )

        self._flip = flip

    def get_example(self, i):
        image = super(MyImageDataset, self).get_example(i)
        image = (image / 127.5) - 1.
        if self._flip:
            if np.random.randint(2) == 0:
                image = image[:, :, ::-1]
        return image


try:
    x = Variable(np.asarray([1, 2, 3], dtype='f'))
    y = F.sum(1.0 / x)
    y.backward(enable_double_backprop=True, retain_grad=True)
    (F.sum(x.grad_var)).backward()
except:
    print('This code uses double-bp of DivFromConstant (not yet merged).')
    print('Please merge this PR: https://github.com/chainer/chainer/pull/3615 to chainer.')
    print('    (in chainer repository)')
    print('    git fetch origin pull/3615/head:rdiv')
    print('    git merge rdiv')
    print('    (reinstall chainer')
    exit(0)

try:
    x = Variable(np.asarray([1, 2, 3], dtype='f'))
    y = F.sum(F.sqrt(x))
    y.backward(enable_double_backprop=True, retain_grad=True)
    (F.sum(x.grad_var)).backward()
except:
    print('This code uses double-bp of Sqrt (not yet merged).')
    print('Please merge this PR: https://github.com/chainer/chainer/pull/3581 to chainer')
    print('    (in chainer repository)')
    print('    git fetch origin pull/3581/head:sqrt')
    print('    git merge sqrt')
    print('    (reinstall chainer')
    exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Train script')
    parser.add_argument(
        'dataset',
        help='a text file in which image file paths are listed'
    )
    parser.add_argument(
        '--flip', action='store_true',
        help='if specified, randomly flip image horizontally'
    )
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='default=16')
    parser.add_argument('--max_iter', '-m', type=int, default=400000,
                        help='default=400000')
    parser.add_argument('--max_ch', '-c', type=int, default=512,
                        help='default=512')
    parser.add_argument('--n_hidden', type=int, default=512,
                        help='default=512')
    parser.add_argument('--size', type=int, default=128,
                        help='default=128')
    parser.add_argument('--max_stage', '-s', type=int, default=10,
                        help='default=10')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU) (default=-1)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result (default=result)')
    parser.add_argument('--snapshot_interval', type=int, default=10000,
                        help='Interval of snapshot (default=10000)')
    parser.add_argument('--evaluation_interval', type=int, default=50000,
                        help='Interval of evaluation (default=50000)')
    parser.add_argument('--out_image_interval', type=int, default=10000,
                        help='Interval of evaluation (default=10000)')
    parser.add_argument('--stage_interval', type=int, default=400000,
                        help='Interval of stage progress (default=400000)')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console (default=100)')
    parser.add_argument('--n_dis', type=int, default=1,
                        help='number of discriminator update per generator update (default=1)')
    parser.add_argument('--lam', type=float, default=10,
                        help='gradient penalty (default=10)')
    parser.add_argument('--gamma', type=float, default=750,
                        help='gradient penalty (default=750)')
    parser.add_argument('--pooling_comp', type=float, default=1.0,
                        help='compensation (default=1.0)')
    parser.add_argument('--pretrained_generator', type=str, default='')
    parser.add_argument('--pretrained_discriminator', type=str, default='')
    parser.add_argument('--initial_stage (default=0.0)',
                        type=float, default=0.0)
    parser.add_argument(
        '--generator_smoothing (default=0.999)', type=float, default=0.999)

    args = parser.parse_args()
    record_setting(args.out)

    report_keys = ['stage', 'loss_dis', 'loss_gp', 'loss_gen',
                   'g', 'inception_mean', 'inception_std', 'FID']
    max_iter = args.max_iter

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()

    generator = Generator(
        n_hidden=args.n_hidden,
        ch=args.max_ch,
        max_stage=args.max_stage,
        size=args.size
    )
    generator_smooth = Generator(
        n_hidden=args.n_hidden,
        ch=args.max_ch,
        max_stage=args.max_stage,
        size=args.size
    )
    discriminator = Discriminator(pooling_comp=args.pooling_comp)

    # select GPU
    if args.gpu >= 0:
        generator.to_gpu()
        generator_smooth.to_gpu()
        discriminator.to_gpu()
        print('use gpu {}'.format(args.gpu))

    if args.pretrained_generator != '':
        chainer.serializers.load_npz(args.pretrained_generator, generator)
    if args.pretrained_discriminator != '':
        chainer.serializers.load_npz(
            args.pretrained_discriminator, discriminator)
    copy_param(generator_smooth, generator)

    # Setup an optimizer
    def make_optimizer(model, alpha=0.001, beta1=0.0, beta2=0.99):
        optimizer = chainer.optimizers.Adam(
            alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        # optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
        return optimizer

    opt_gen = make_optimizer(generator)
    opt_dis = make_optimizer(discriminator)

    train_dataset = MyImageDataset(args.dataset, flip=args.flip)
    train_iter = chainer.iterators.SerialIterator(
        train_dataset, args.batchsize)

    # Set up a trainer
    updater = Updater(
        models=(generator, discriminator, generator_smooth),
        iterator={
            'main': train_iter},
        optimizer={
            'opt_gen': opt_gen,
            'opt_dis': opt_dis},
        device=args.gpu,
        n_dis=args.n_dis,
        lam=args.lam,
        gamma=args.gamma,
        smoothing=args.generator_smoothing,
        initial_stage=args.initial_stage,
        stage_interval=args.stage_interval,
        size=args.size
    )

    trainer = training.Trainer(updater, (max_iter, 'iteration'), out=args.out)

    trainer.extend(extensions.snapshot_object(
        generator, 'generator_{.updater.iteration}.npz'),
        trigger=(args.snapshot_interval, 'iteration')
    )
    trainer.extend(extensions.snapshot_object(
        generator_smooth, 'generator_smooth_{.updater.iteration}.npz'),
        trigger=(args.snapshot_interval, 'iteration')
    )
    trainer.extend(extensions.snapshot_object(
        discriminator, 'discriminator_{.updater.iteration}.npz'),
        trigger=(args.snapshot_interval, 'iteration')
    )
    trainer.extend(
        extensions.LogReport(keys=report_keys,
                             trigger=(args.display_interval, 'iteration'))
    )
    trainer.extend(extensions.PrintReport(report_keys),
                   trigger=(args.display_interval, 'iteration'))
    trainer.extend(
        sample_generate(generator_smooth, args.out),
        trigger=(args.out_image_interval, 'iteration'),
        priority=extension.PRIORITY_WRITER
    )
    trainer.extend(
        sample_generate_light(generator_smooth, args.out),
        trigger=(args.evaluation_interval // 10, 'iteration'),
        priority=extension.PRIORITY_WRITER
    )
    trainer.extend(
        calc_inception(generator_smooth),
        trigger=(args.evaluation_interval, 'iteration'),
        priority=extension.PRIORITY_WRITER
    )
    trainer.extend(
        calc_FID(generator_smooth),
        trigger=(args.evaluation_interval, 'iteration'),
        priority=extension.PRIORITY_WRITER
    )
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()

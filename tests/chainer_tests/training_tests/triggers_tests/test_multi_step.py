import unittest

from chainer import testing
from chainer import training
from chainer.training import triggers


class DummyUpdater(training.Updater):

    def __init__(self, iterations_per_epoch):
        self.iteration = 0
        self.iterations_per_epoch = iterations_per_epoch

    def finalize(self):
        pass

    def get_all_optimizers(self):
        return {}

    def update(self):
        self.iteration += 1

    @property
    def epoch(self):
        return self.iteration // self.iterations_per_epoch

    @property
    def is_new_epoch(self):
        return self.iteration > 0 and \
            self.iteration % self.iterations_per_epoch == 0


def _test_trigger(self, trigger, expected):
    updater = DummyUpdater(2)
    trainer = training.Trainer(updater)
    for expected_value in expected:
        updater.update()
        self.assertEqual(trigger(trainer), expected_value)


class TestMultiStepTrigger(unittest.TestCase):

    def test_multi_step_trigger(self):
        trigger = triggers.MultiStepTrigger([1, 4, 5], 'iteration')
        expected = [True, False, False, True, True, False, False, False]
        _test_trigger(self, trigger, expected)
        trigger = triggers.MultiStepTrigger([1, 3, 4], 'epoch')
        expected = [False, True, False, False, False, True, False, True]
        _test_trigger(self, trigger, expected)


testing.run_module(__name__, __file__)

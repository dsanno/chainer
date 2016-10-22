class MultiStepTrigger(object):

    """Trigger based on multiple steps.

    This trigger accepts specific steps. There are two ways to specify the
    steps: based on iterations and epochs. `Iteration` means the
    number of updates, while `epoch` means the number of sweeps over the
    training dataset. Both values are defined by the updater.

    For the description of triggers, see :func:`~chainer.training.get_trigger`.

    Args:
        steps (list of int): Steps when the trigger fires.
        unit (str): Unit of the ``steps``. It must be either ``'iteration'`` or
        ``'epoch'``.

    """

    def __init__(self, steps, unit):
        self.steps = steps
        assert unit == 'epoch' or unit == 'iteration'
        self.unit = unit

    def __call__(self, trainer):
        """Decides whether the extension should be called on this iteration.

        Args:
            trainer (Trainer): Trainer object that this trigger is associated
                with. The updater associated with this trainer is used to
                determine if the trigger should fire.

        Returns:
            bool: True if the corresponding extension should be invoked in this
                iteration.

        """
        updater = trainer.updater
        if self.unit == 'epoch':
            return updater.is_new_epoch and updater.epoch in self.steps
        else:
            iteration = updater.iteration
            return iteration > 0 and iteration in self.steps

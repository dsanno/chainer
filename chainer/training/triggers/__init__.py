from chainer.training.triggers import interval
from chainer.training.triggers import minmax_value_trigger
from chainer.training.triggers import multi_step


IntervalTrigger = interval.IntervalTrigger
MaxValueTrigger = minmax_value_trigger.MaxValueTrigger
MinValueTrigger = minmax_value_trigger.MinValueTrigger
MultiStepTrigger = multi_step.MultiStepTrigger

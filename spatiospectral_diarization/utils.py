import dataclasses
import numpy as np


@dataclasses.dataclass
class ABCKernel1D:
    kernel_size: int
    # stride: int
    # dilation: int
    padding_mode: ['edge'] = 'edge'
    pad_position: ['pre', 'post', None] = 'pre'

    def __post_init__(self):
        assert self.kernel_size % 2 == 1, (self.kernel_size, 'must be odd.')
        assert self.pad_position in ['pre', 'post', None], self.pad_position

    def kernel_fn(self, x):
        """
        Do an operation that removes the last axis of x.
        e.g. `np.mean(x, axis=-1)`
        """
        raise NotImplementedError()

    def __call__(self, x):
        if self.pad_position == 'pre':
            shift = self.kernel_size // 2
            padding = [(0, 0)] * (x.ndim - 1) + [(shift, shift)]
            x = np.pad(x, padding, 'edge')

        y = self.kernel_fn(
            pb.array.segment_axis(x, self.kernel_size, 1, end='pad')
        )

        if self.pad_position == 'post':
            shift = self.kernel_size // 2
            padding = [(0, 0)] * (y.ndim - 1) + [(shift, shift)]
            y = np.pad(y, padding, 'edge')
        return y


@dataclasses.dataclass
class Kernel1D(ABCKernel1D):
    kernel: callable = np.mean

    def kernel_fn(self, x):
        return self.kernel(x, axis=-1)


@dataclasses.dataclass
class MaxThresholdKernel1D(ABCKernel1D):
    threshold: float = 0.2

    def kernel_fn(self, x):
        """
        >>> w2a = MaxThresholdKernel1D(3, threshold=5)
        >>> x = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 3, 2, 1, 0, 4, 5])
        >>> x.shape
        (15,)
        >>> y = w2a(x)
        >>> y.shape
        (15,)
        >>> print(x, y.astype(int), sep='\\n')
        [0 0 1 2 3 4 5 6 7 3 2 1 0 4 5]
        [0 0 0 0 0 1 1 1 1 1 0 0 0 1 1]
        """
        x_max = np.max(x, axis=-1)
        return np.where(x_max >= self.threshold, True, False)


def reduction_max_threshold(x, axis=-1, threshold=0.2):
    assert axis == -1, axis
    x_max = np.max(x, axis=axis)
    return np.where(x_max < threshold, False, True)


def smooth(noisy, window=25, reduction=reduction_max_threshold, pre_pad=True):
    assert (window % 2) == 1, (window, 'must be odd.')

    if pre_pad:
        shift = window // 2
        padding = [(0, 0)] * (noisy.ndim - 1) + [(shift, shift)]
        noisy = np.pad(noisy, padding, 'edge')

    smoothed = reduction(
        pb.array.segment_axis(noisy, window, 1, end='pad'),
        axis=-1)

    if not pre_pad:
        shift = window // 2
        padding = [(0, 0)] * (noisy.ndim - 1) + [(shift, shift)]
        smoothed = np.pad(smoothed, padding, 'edge')
    return smoothed

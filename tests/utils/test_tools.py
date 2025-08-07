import unittest
import torch

from utils import set_random_state, set_cuda_device, set_torch_dtype


class TestTools(unittest.TestCase):

    def test_set_random_seed(self):
        """test the function `set_random_seed`"""
        for seed in range(10):
            set_random_state(seed)

    def test_set_cuda_device(self):
        """test the function `set_cuda_device`"""
        for cpu in [False, True]:
            for cuda_index in range(4):
                set_cuda_device(cpu=cpu, cuda_index=cuda_index)

    def test_set_torch_dtype(self):
        """test the function `set_torch_dtype`"""
        for dtype in [torch.float32, torch.float64]:
            set_torch_dtype(dtype)
            tensor = torch.randn(5, 5, requires_grad=True)
            self.assertTrue(dtype, tensor.dtype)
            

if __name__ == '__main__':
    unittest.main()
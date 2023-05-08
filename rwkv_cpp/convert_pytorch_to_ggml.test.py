import os
import struct
import torch
import convert_pytorch_to_ggml
from typing import Dict

def test() -> None:
    test_file_path = 'convert_pytorch_rwkv_to_ggml_test.tmp'

    try:
        state_dict: Dict[str, torch.Tensor] = {
            'emb.weight': torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32),
            'blocks.0.ln1.weight': torch.tensor([1], dtype=torch.float32)
        }

        convert_pytorch_to_ggml.write_state_dict(state_dict, dest_path=test_file_path, data_type='float32')

        with open(test_file_path, 'rb') as input:
            actual_bytes: bytes = input.read()

        expected_bytes: bytes = struct.pack(
            '=iiiiii' + 'iiiii10sffffff' + 'iiii19sf',
            0x67676d66,
            100,
            3,
            2,
            1,
            0,
            # emb.weight
            2,
            10,
            0,
            2, 3,
            'emb.weight'.encode('utf-8'),
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            # blocks.0.ln1.weight
            1,
            19,
            0,
            1,
            'blocks.0.ln1.weight'.encode('utf-8'),
            1.0
        )

        assert list(actual_bytes) == list(expected_bytes), f'\nActual: {list(actual_bytes)}\nExpected: {list(expected_bytes)}'

        print('All tests pass')
    finally:
        if os.path.isfile(test_file_path):
            os.remove(test_file_path)

if __name__ == "__main__":
    test()

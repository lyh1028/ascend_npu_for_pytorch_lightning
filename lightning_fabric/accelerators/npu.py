# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import lru_cache
from typing import List, Optional, Union

import torch
import torch_npu
from typing_extensions import override

from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.accelerators.registry import _AcceleratorRegistry
from lightning_fabric.utilities.rank_zero import rank_zero_info


class NPUAccelerator(Accelerator):
    """Accelerator for NVIDIA CUDA devices."""

    @override
    def setup_device(self, device: torch_npu.npu.device) -> None:
        """
        Raises:
            ValueError:
                If the selected device is not of type NPU.
        """
        if device.type != "npu":
            raise ValueError(f"Device should be NPU, got {device} instead.")
        _check_cuda_matmul_precision(device)
        torch.npu.set_device(device)

    @override
    def teardown(self) -> None:
        _clear_npu_memory()

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[List[int]]:
        """Accelerator device parsing logic."""
        from lightning_fabric.utilities.device_parser import _parse_npu_ids

        return _parse_npu_ids(devices)

    @staticmethod
    @override
    def get_parallel_devices(devices: List[int]) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        return [torch.device("npu", i) for i in devices]

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return num_npu_devices()

    @staticmethod
    @override
    def is_available() -> bool:
        return num_npu_devices() > 0

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            "npu",
            cls,
            description=cls.__name__,
        )


def find_usable_cuda_devices(num_devices: int = -1) -> List[int]:
    """Returns a list of all available and usable Ascend NPU devices.
    
    A GPU is considered usable if we can successfully move a tensor to the device, and this is what this function
    tests for each GPU on the system until the target number of usable devices is found.

    A subset of GPUs on the system might be used by other processes, and if the GPU is configured to operate in
    'exclusive' mode (configurable by the admin), then only one process is allowed to occupy it.

    Args:
        num_devices: The number of devices you want to request. By default, this function will return as many as there
            are usable CUDA GPU devices available.

    Warning:
        If multiple processes call this function at the same time, there can be race conditions in the case where
        both processes determine that the device is unoccupied, leading into one of them crashing later on.

    """
    if num_devices == 0:
        return []
    visible_devices = _get_all_visible_npu_devices()
    if not visible_devices:
        raise ValueError(
            f"You requested to find {num_devices} devices but there are no visible Ascend devices on this machine."
        )
    if num_devices > len(visible_devices):
        raise ValueError(
            f"You requested to find {num_devices} devices but this machine only has {len(visible_devices)} NPUs."
        )

    available_devices = []
    unavailable_devices = []

    for npu_idx in visible_devices:
        try:
            torch.tensor(0, device=torch.device("npu", npu_idx))
        except RuntimeError:
            unavailable_devices.append(npu_idx)
            continue

        available_devices.append(npu_idx)
        if len(available_devices) == num_devices:
            # exit early if we found the right number of GPUs
            break

    if num_devices != -1 and len(available_devices) != num_devices:
        raise RuntimeError(
            f"You requested to find {num_devices} devices but only {len(available_devices)} are currently available."
            f" The devices {unavailable_devices} are occupied by other processes and can't be used at the moment."
        )
    return available_devices


def _get_all_visible_npu_devices() -> List[int]:
    """Returns a list of all visible Ascend NPU devices.

    """
    return list(range(num_npu_devices()))


def num_npu_devices() -> int:
    """Returns the number of available CUDA devices."""
    return torch.npu.device_count()


def is_npu_available() -> bool:
    """Returns a bool indicating if CUDA is currently available."""
    # We set `PYTORCH_NVML_BASED_CUDA_CHECK=1` in lightning_fabric.__init__.py
    return torch.npu.is_available()



@lru_cache(1)  # show the warning only ever once
def _check_cuda_matmul_precision(device: torch.device) -> None:
    if not torch.npu.is_available():
        return
    # check that the user hasn't changed the precision already, this works for both `allow_tf32 = True` and
    # `set_float32_matmul_precision`
    if torch.get_float32_matmul_precision() == "highest":  # default
        rank_zero_info(
            f"You are using a CUDA device ({torch.npu.get_device_name(device)!r}) that has Tensor Cores. To properly"
            " utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off"
            " precision for performance. For more details, read https://pytorch.org/docs/stable/generated/"
            "torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision"
        )
    # note: no need change `torch.backends.cudnn.allow_tf32` as it's enabled by default:
    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices


def _clear_npu_memory() -> None:
    # strangely, the attribute function be undefined when torch.compile is used
    torch.npu.empty_cache()

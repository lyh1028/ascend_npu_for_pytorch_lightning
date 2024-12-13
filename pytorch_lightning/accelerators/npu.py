#coding=gbk
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
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Union
from contextlib import nullcontext
import torch
import torch_npu
from typing_extensions import override

import pytorch_lightning as pl
from lightning_fabric.accelerators import _AcceleratorRegistry
from lightning_fabric.accelerators.npu import _check_cuda_matmul_precision, _clear_npu_memory, num_npu_devices
from lightning_fabric.utilities.device_parser import _parse_npu_ids
from lightning_fabric.utilities.types import _DEVICE
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException

_log = logging.getLogger(__name__)


class NPUAccelerator(Accelerator):
    """Accelerator for Ascend NPU devices."""

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not GPU.
        """
        if device.type != "npu":
            raise MisconfigurationException(f"Device should be NPU, got {device} instead")
        _check_cuda_matmul_precision(device)
        #torch.cuda.set_device(device)
        torch.npu.set_device(device)

    @override
    def setup(self, trainer: "pl.Trainer") -> None:
        # TODO refactor input from trainer to local_rank @four4fish
        self.set_ascend_flags(trainer.local_rank)
        _clear_npu_memory()

    @staticmethod
    def set_ascend_flags(local_rank: int) -> None:
        # set the correct cuda visible devices (using pci order)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        all_npu_ids = ",".join(str(x) for x in range(num_npu_devices()))
        devices = os.getenv("CUDA_VISIBLE_DEVICES", all_npu_ids)
        _log.info(f"LOCAL_RANK: {local_rank} - NPU_VISIBLE_DEVICES: [{devices}]")

    @override
    def get_device_stats(self, device: _DEVICE) -> Dict[str, Any]:
        """Gets stats for the given GPU device.

        Args:
            device: GPU device for which to get stats

        Returns:
            A dictionary mapping the metrics to their values.

        Raises:
            FileNotFoundError:
                If nvidia-smi installation not found

        """
        return torch.npu.memory_stats(device)

    @override
    def teardown(self) -> None:
        _clear_npu_memory()

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[List[int]]:
        """Accelerator device parsing logic."""
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

    @override
    def get_distribute_name(self) -> str:
        return "hccl"
    
    @override
    def get_stream_context(self, device_id: List[int]) -> Any:
        return torch.npu.stream(torch.npu.Stream()) if device_id is not None else nullcontext()


def _get_npu_id(device_id: int) -> str:
    """Get the unmasked real GPU IDs."""
    # All devices if `CUDA_VISIBLE_DEVICES` unset
    default = ",".join(str(i) for i in range(num_npu_devices()))
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default=default).split(",")
    return cuda_visible_devices[device_id].strip()

# ascend_npu_for_pytorch_lightning
Modified pytorch_lightning packages that adapt to Huawei's Ascend NPU environment. pytorch and pytorch_lightning version is 2.4.0

# Main Modifications:

Added npu.py under the accelerator folder in the pytorch_lightning directory.

Added npu_parallel.py under the strategies folder in the pytorch_lightning directory.

Added npu.py under the accelerators folder in the lightning_fabric directory.

# Other Modifications:

Can be compared with the original pytorch_lightning library for additional trivial changes.

# Note:

This repository is based on the older version of pytorch_lightning 2.4.0, which is different from the lightning library installed directly via `pip install lightning`. However, the overall functionality remains consistent. For more details, please refer to https://github.com/Lightning-AI/pytorch-lightning/discussions/16688.

# Usage Example

Replace pytorch_lightning, lightning_fabric directory in your conda env such as "xxx/python3.9/site-packages" with this repository's.

## Single device

```python
trainer = pl.Trainer(max_epochs=5, accelerator='npu', devices=1)
```

```shell
python test.py
```

## Multi-device
Assume your python code `test.py` like this:
```python
trainer = pl.Trainer(max_epochs=5, accelerator='npu', devices=4, strategy='ddp_npu')
# you may specify the accelerator flag as 'npu'.
# using strategy=ddp may also work.
trainer.fit(model, datamodule=datamodule)
trainer.validate(model, datamodule=datamodule)
```

make sure -nproc_per_node equals to devices num. Also, Ascend NPU doesn't support devices=3, 5, 6, 7
```shell
torchrun --nnodes=1 --nproc_per_node=4 test.py
```
Some weird bugs: If use 'python' instead of 'torchrun', it'll throw an error like 'segmentation fault' when setting num_workers>0


Some useful references:
https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/ptmoddevg/trainingmigrguide/PT_LMTMOG_0021.html?sub_id=/zh/Pytorch/60RC3/ptmoddevg/trainingmigrguide/PT_LMTMOG_0080.html
https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/quickstart/useguide/useguide_0001.html


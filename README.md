# Multiple Choice

Based on the script [`run_multiple_choice.py`]().

### Fine-tuning on Arc Challenge 
```bash
#using google corpus
export ARC_DIR=/home/tsinghuaee08/Team21/Arc-Challenge/ARC-V1-Feb2018-2/google_corpus
python run_multiple_choice.py \
--model_type roberta \
--task_name arc \
--model_name_or_path roberta-base \
--do_train \
--do_eval \
--do_lower_case \
--data_dir $ARC_DIR \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--max_seq_length 512 \
--output_dir models_output/arc_roberta_base \
--per_gpu_eval_batch_size=16 \
--per_gpu_train_batch_size=16 \
--gradient_accumulation_steps 2 \
--overwrite_output
```
Training with the defined hyper-parameters yields the following results:

```
***** Eval results *****
eval_acc = XXXXXXXXXX
eval_loss = XXXXXXXXX
```

- --model_name_or_path roberta-base  给出的只是**roberta-base**这个模型名称，需要从hugging face 下载预训练实际的预训练模型
- output_dir 命名规则:**models_output/+your model name**

### GPU Memory 爆掉了怎么办

一个可以用来计算GPU占用量的工具[gpu_mem_track](https://github.com/Oldpan/Pytorch-Memory-Utils)  
```
from gpu_mem_track import  MemTracker

gpu_tracker = MemTracker(frame)
gpu_tracker.track()
#your code here for GPU memory usage 
gpu_tracker.track()
```

### 各个文件的作用

| 文件名     | 作用     |
| ------- | -------------------- |
| ARC-V1-Feb2018-2 | 训练数据 |
| distillation | 最后做消融研究可能会用到的轮子 |
| gpu_mem_track.py    | 看GPU显存占用情况的轮子 |
|  test.md|可以在这个上面熟练git的操作|
| run_bertology.py    | 官方给的如何用BERT的示例 |
| run_multiple_choice.py | 可以跑的模型的示例   |
| utils_multiple_choice.py | 里面有Race,arc数据集的处理方法和实际tokenize的方式 |

### Schdule(11.27-12.4)

- [ ] 一种预训练模型的搭建，在 run_multiple_choice.py 基础上做修改，代码命名为**your mode_multiple_choice.py**, pretraining model应该存在**model/your model**文件夹下面。

- [ ] 一种预训练模型的搭建，代码命名规则和模型文件存放如上所示

- [ ] **Race数据集**的下载，能够在run_multiple_choice.py上跑通；实现根据question_id就可找到对应问题的功能的函数。代码写在utils文件夹下。

- [ ] 根据arc_corpus的csv文件，实现根据question_id就可找到对应问题基本信息( 如类别年级)的函数；修改run_multiple_choice.py的evaluate部分，查找出

- [ ] emsemble方法的调研和初步实现

### 目前已经遇到的bug
- transformers 2.2.0里面已经把WarmupLinearSchedule方法替换成了get_linear_schedule_with_warmup

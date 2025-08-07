import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_HOME'] = r'G:\model\hf'

os.environ['MODELSCOPE_CACHE'] = r'G:\model\modelscope'

import pandas as pd
import torch
from datasets import load_dataset, Dataset
from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from swanlab.integration.transformers import SwanLabCallback
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig

# 设置CUDA内存分配器
# torch.cuda.set_per_process_memory_fraction(1.0)  # 设置使用100%的显存
torch.cuda.empty_cache()  # 清空缓存
torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优

# DS_CONFIG = "ds_z0_config.json"

"""
多卡训练
deepspeed --include "localhost:0,1" train.py
"""

batch_size = 4
epochs = 3
ds_size = 20000

hf_home = os.environ.get("HF_HOME")

model_name = "Qwen/Qwen3-1.7B"
ds_name = "FreedomIntelligence/medical-o1-reasoning-SFT"

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# device_map = {"": int(os.environ.get("LOCAL_RANK")) or 0}
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

# 开启梯度检查点时，需要执行此方法
model.enable_input_require_grads()

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
)

model = get_peft_model(model, config)

ds = load_dataset(ds_name, "zh", cache_dir="mydata/reasoning", split=f"train")

def generate_conversation(examples):
    questions = examples["Question"]
    cots = examples["Complex_CoT"]
    solutions = examples["Response"]
    conversations = []
    for question, cot, solution in zip(questions, cots, solutions):
        conversations.append([
            {"role": "user", "content": question},
            {"role": "assistant", "content": f'<think>{cot}</think>{solution}'},
        ])
    return {"conversations": conversations}


reasoning_conversations = tokenizer.apply_chat_template(
    ds.map(generate_conversation, batched=True)["conversations"],
    tokenize=False
)
print(type(reasoning_conversations))
df = pd.DataFrame({"text": reasoning_conversations})
train_ds = Dataset.from_pandas(df).shuffle(seed=42)

swanlab_callback = SwanLabCallback(
    project="Qwen3-1.7B-finetune",
    experiment_name=model_name,
    description="基于Qwen3-1.7B模型微调medical-o1-reasoning-SFT数据集",
    config={
        "model": model_name,
        "dataset": "FreedomIntelligence/medical-o1-reasoning-SFT",
        "train_data_number": len(train_ds),
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    }
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,
    eval_dataset=None,
    callbacks=[swanlab_callback],
    args=SFTConfig(
        output_dir="lora_model",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=16,
        warmup_steps=5,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
        fp16=True,
        max_grad_norm=1.0,
        logging_first_step=True,
        save_steps=100,
    )
)

# 显示当前内存统计
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# 显示最终内存和时间统计
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


class CaptureStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt: bool = False, **kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **kwargs)
        self.generated_text = ""  # 用于存储完整输出

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """重写方法捕获最终文本"""
        self.generated_text += text  # 累积输出
        super().on_finalized_text(text, stream_end=stream_end)  # 保持原样输出到终端

    def get_output(self) -> str:
        """获取完整生成内容"""
        return self.generated_text.strip()

def ask(question, is_thinking=True, save_to_file=None):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=is_thinking
    )

    # 使用自定义的 CaptureStreamer
    streamer = CaptureStreamer(tokenizer, skip_prompt=True)

    # 生成响应
    model.eval()  # 确保模型在推理模式
    with torch.no_grad():
        _ = model.generate(
            **tokenizer(text, return_tensors="pt").to("cuda"),
            max_new_tokens=1024,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            streamer=streamer,  # 关键：使用自定义的 streamer
        )

    # 获取完整输出
    full_output = streamer.get_output()

    # 保存到文件
    if save_to_file:
        try:
            with open(save_to_file, "w", encoding="utf-8") as f:
                f.write(full_output)
            print(f"成功写入文件: {save_to_file}")
        except Exception as e:
            print(f"写入文件失败: {e}")

    return full_output


ask("根据描述，一个1岁的孩子在夏季头皮出现多处小结节，长期不愈合，且现在疮大如梅，溃破流脓，口不收敛，头皮下有空洞，患处皮肤增厚。这种病症在中医中诊断为什么病？",
    save_to_file="../law/output.txt")

print("##########")

model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

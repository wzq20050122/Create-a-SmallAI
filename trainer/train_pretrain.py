import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from model.model_SmallAI import SmallAIConfig, SmallAIForCausalLM
from dataset.lm_dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def manage_checkpoints(save_dir, pattern, keep_last_n):
    """管理检查点文件，只保留最近的N个"""
    if keep_last_n <= 0:
        return
    
    import glob
    import re
    
    # 找到所有匹配的检查点文件
    checkpoint_files = glob.glob(os.path.join(save_dir, pattern))
    if len(checkpoint_files) <= keep_last_n:
        return
    
    # 从文件名中提取epoch和步数进行排序
    def extract_sort_key(filename):
        # 优先尝试匹配epoch (包括half标记)
        epoch_match = re.search(r'epoch(\d+)(_half)?', filename)
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            is_half = epoch_match.group(2) is not None
            # 半epoch的排序值为: epoch * 2 - 1, 完整epoch为: epoch * 2
            return epoch_num * 2 - (1 if is_half else 0)
        
        # 然后尝试匹配步数
        step_match = re.search(r'step(\d+)', filename)
        if step_match:
            return int(step_match.group(1))
        
        return 0

    # 按提取的键值排序
    checkpoint_files.sort(key=extract_sort_key)
    
    # 删除旧的检查点
    files_to_delete = checkpoint_files[:-keep_last_n]
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            Logger(f'删除旧检查点: {os.path.basename(file_path)}')
        except OSError:
            Logger(f'无法删除文件: {file_path}')


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb, tb_writer):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})
            
            # TensorBoard记录
            if tb_writer is not None and (not ddp or dist.get_rank() == 0):
                global_step = epoch * iter_per_epoch + step + 1
                tb_writer.add_scalar("train/loss", loss.item() * args.accumulation_steps, global_step)
                tb_writer.add_scalar("train/lr", optimizer.param_groups[-1]['lr'], global_step)
                tb_writer.add_scalar("train/epoch", epoch + 1, global_step)

        # 每半个epoch保存检查点
        half_epoch_step = iter_per_epoch // 2
        if args.save_every_half_epoch and (not ddp or dist.get_rank() == 0):
            if step + 1 == half_epoch_step:  # 半个epoch
                model.eval()
                moe_path = '_moe' if lm_config.use_moe else ''
                half_filename = f'pretrain_{lm_config.hidden_size}{moe_path}_epoch{epoch+1}_half.pth'
                half_ckp = f'{args.save_dir}/{half_filename}'
                
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                
                state_dict = {k: v.half() for k, v in state_dict.items()}
                torch.save(state_dict, half_ckp)
                Logger(f'半个epoch检查点已保存: {half_filename}')
                
                # 管理检查点数量
                if args.keep_last_n_checkpoints > 0:
                    pattern = f'pretrain_{lm_config.hidden_size}{moe_path}_epoch*_half.pth'
                    manage_checkpoints(args.save_dir, pattern, args.keep_last_n_checkpoints)
                model.train()

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            step_filename = f'pretrain_{lm_config.hidden_size}{moe_path}_step{epoch * iter_per_epoch + step + 1}.pth'
            ckp = f'{args.save_dir}/{step_filename}'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            Logger(f'步数检查点已保存: {step_filename}')
            
            # 管理检查点数量
            if args.keep_last_n_checkpoints > 0:
                pattern = f'pretrain_{lm_config.hidden_size}{moe_path}_step*.pth'
                manage_checkpoints(args.save_dir, pattern, args.keep_last_n_checkpoints)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    model = SmallAIForCausalLM(lm_config).to(args.device)
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmallAI Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="SmallAI-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    # 新增参数
    parser.add_argument("--use_tb", action="store_true", help="使用TensorBoard记录训练过程")
    parser.add_argument("--save_every_half_epoch", action="store_true", help="每半个epoch保存检查点")
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=3, help="保留最新的N个检查点")
    args = parser.parse_args()

    lm_config = SmallAIConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"SmallAI-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    # 初始化wandb
    wandb = None
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb as wandb_module
        wandb = wandb_module
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    
    # 初始化TensorBoard
    tb_writer = None
    if args.use_tb and (not ddp or dist.get_rank() == 0):
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir)
        Logger(f"TensorBoard日志将保存到: {log_dir}")
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb, tb_writer)
    
    # 训练结束后保存最终模型
    if not ddp or dist.get_rank() == 0:
        model.eval()
        moe_path = '_moe' if lm_config.use_moe else ''
        final_filename = f'pretrain_{lm_config.hidden_size}{moe_path}_final.pth'
        final_ckp = f'{args.save_dir}/{final_filename}'
        
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        
        state_dict = {k: v.half() for k, v in state_dict.items()}
        torch.save(state_dict, final_ckp)
        Logger(f'训练完成，最终模型已保存: {final_filename}')
        
        # 关闭TensorBoard writer
        if tb_writer is not None:
            tb_writer.close()
            Logger("TensorBoard writer已关闭")

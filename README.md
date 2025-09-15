# SmallAI
从零开始搭建一个小参数量的LLM（maybe可以称作SLM，hh）

## 0、克隆项目

```bash
git clone https://github.com/wzq20050122/SmallAI.git
```

## 1、环境配置
<details>
<summary>环境配置 cuda</summary>

* 我是在 autodl 上租用一个 RTX 4090D 单卡跑的
* PyTorch 2.3.0
* Python 3.12 (ubuntu22.04)
* CUDA 12.1

</details>

然后下载requirement.txt
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2、训练tokenizer
你可以运行scripts\train_tokenizer.py来得到model\tokenizer_config.json和model\tokenizer.json，这将作为tokenizer来为后面的分词起作用。不过直接用我训练的tokenizer就可以省去这一步hh。

## 3、训练pretrain模型
利用trainer\train_pretrain.py文件便可以开始训练pretrain模型了
下面是我训练的loss曲线，我训练了大概三个epoch，模型参数存放在out文件夹下  

<div align="center">
  
![train_loss](https://github.com/user-attachments/assets/d8bd076c-3978-4cf2-bffa-a449552a344c)

</div>  

然后，我们测试一下这个模型的性能，这里我用到了eval_model.py  


<div align="center">
  
https://github.com/user-attachments/assets/c8a74c1e-38a2-4a6d-bca1-68da7dee005d

</div>  
其实在这里可以看得到模型性能并不是很好，在回答问题时会有很多错误信息并且出现语无伦次的情况。可能的原因时其中华南理工大学的信息其实可能在训练语料中并未出现，如果想要回答的比较准确的话可以做进一步微调或者RAG检索。

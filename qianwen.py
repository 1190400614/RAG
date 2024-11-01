from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from datetime import datetime
from FlagEmbedding import BGEM3FlagModel
import pandas as pd
import re
import numpy as np
from nltk.tokenize import sent_tokenize

# 受计算能力限制，每次只处理一行数据，指定数据行号
prepareParam = 0

model = BGEM3FlagModel('BAAI/bge-m3',use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

print("加载分词BGE-M3模型成功")
df = pd.read_json(r'D:\shorttimeexe\communicationBot\art.jsonl', lines=True)
prompt = df['input'].iloc[prepareParam]
#prompt = "What role does Bruno Latour's sociology play in Alworth's methodology?"
messages = df['context'].tolist()[prepareParam]

def split_sentences(text):
    # 按标点和换行分割，保留句子
    # [.!?;]：这个部分匹配句号 (.)、问号 (?) 或感叹号 (!) 中的任意一个。它表示句子的结束。
    # \s*：这个部分匹配零个或多个空白字符（如空格、制表符等）。它用于处理标点后面的空格，使得在分割时不会留下多余的空白。
    # |：这个符号表示“或”，意味着可以匹配前面的模式或后面的模式。
    # \n+：这个部分匹配一个或多个换行符。这使得在文本中换行的地方也会进行分割。
    text = re.sub(r',\s*[^A-Za-z0-9]*,', ',', text)#减少逗号之间没有内容
    sentences = sent_tokenize(text)
    #sentences = re.split(r'[.!?;]\s*|\n', text)#分割句子
    # 清理和过滤句子
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 10 and s.count(' ') >= 3 and '**' not in s]
    print("未降重的句子总数：")
    print(len(all_sentences))
    print(len(sentences))
    return sentences

# 处理每个 message
all_sentences = []

# 句子降重函数
for message in messages:
    # 存储唯一句子的集合
    unique_sentences_hash = set()
    # 分割内容
    sentences = split_sentences(message)
    for sentence in sentences:
        # 计算句子的哈希值, 自动过滤重复的句子,在前10行中过滤了1331个句子，得到48983条
        sentence_hash = hash(sentence)
        if sentence_hash in unique_sentences_hash:
            sentences.remove(sentence)
            continue
        unique_sentences_hash.add(sentence_hash)
        
    all_sentences.extend(sentences) 
print("得到context句子总数：")
print(len(all_sentences))
print("分割句子，4000条句子计算太慢，可取一部分演示")
all_sentences = all_sentences[100:120]

# 向量化 prompt
prompt_embedding = model.encode(prompt, batch_size=12, max_length=4096)['dense_vecs']
# 向量化 sentences
# message_embeddings = model.encode(sentence, batch_size=12, max_length=4096)['dense_vecs']
message_embeddings = []
for sentence in all_sentences:
    embedding = model.encode(sentence, batch_size=12, max_length=4096)['dense_vecs']
    message_embeddings.append(embedding)

# 计算相似度
message_embeddings = np.array(message_embeddings)
similarity = prompt_embedding @ message_embeddings.T
print("得到input与context句子的相似度矩阵")
#print(similarity)

# 遍历所有句子并扩展相似度
num_sentences = len(all_sentences)
extended_similarity = np.zeros(similarity.shape)
for i in range(num_sentences):
    if i>1:
        extended_similarity[i] = 3*similarity[i]  # 当前句子的相似度
    elif i==1:
        extended_similarity[i] = 2*similarity[i]
    else:
        extended_similarity[i] = similarity[i]
    # 扩展前两个句子的相似度
    if i - 1 >= 0:  # 前一个句子
        extended_similarity[i] += similarity[i - 1]
    if i - 2 >= 0:  # 前两个句子
        extended_similarity[i] += similarity[i - 2]
    # 扩展后两个句子的相似度
    if i + 1 < num_sentences:  # 后一个句子
        extended_similarity[i] += similarity[i + 1]
    if i + 2 < num_sentences:  # 后两个句子
        extended_similarity[i] += similarity[i + 2]

top_n = 1000
top_indices = np.argsort(similarity)[-top_n:][::-1]  # 从大到小排序索引


print("得到similar_messages若干")
# 获取最相似的 message
similar_messages = ""
current_length = 0
max_chars = 8192
similar_messages = ""
for index in top_indices:
    sentence_length = len(all_sentences[index])        
    # Check if adding the next sentence would exceed the character limit
    if current_length + sentence_length  <= max_chars:  # + len(current_window) to account for spaces or punctuation
        similar_messages += all_sentences[index]
        current_length += sentence_length
    else:
        break

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="D:/shorttimeexe/communicationBot/qianwen",
    torch_dtype="auto",
    device_map="auto"
)
print("加载Qwen模型成功")

#从Hugging Face Hub 加载指定模型的 AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

#打印数据集给的结果
print("the answers of dateset")
print(messages['answers'].iloc[prepareParam])

#prompt是提示词，类似于question；messages是上下文信息，类似于knowledge，一般不显示
messages_with_prompt = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud, Changed by Li Haihan for learning. You are a helpful assistant."},
    similar_messages,
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages_with_prompt,
    tokenize=False,
    add_generation_prompt=True#添加生成提示
)

#生成输入的token
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
#根据model_inputs生成token序列，生成输出内容，generated_ids 是模型生成的输出
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=8192
)

#可能会包含与输入相同的 token（即输入的内容），也可以不切割直接使用
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
#将回答编码到可显示的内容，过滤特殊字符
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("the answers of RAG")
print(response)

#空白对照组：
messages_is_nothing = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud, Changed by Li Haihan for learning. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages_is_nothing,
    tokenize=False,
    add_generation_prompt=True#添加生成提示
)
#生成输入的token
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
#根据model_inputs生成token序列，生成输出内容，generated_ids 是模型生成的输出
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024
)
#可能会包含与输入相同的 token（即输入的内容），也可以不切割直接使用
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
#将回答编码到可显示的内容，过滤特殊字符
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("the answers of LLM without messages")
print(response)
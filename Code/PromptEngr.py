### 這是正式的沒有加載RAG的版本，可以用以基礎的讀取提示詞文本
### 0328 2025 By Leqi

from volcenginesdkarkruntime import Ark

# 初始化客户端（使用 AK/SK）
client = Ark(
    #ak="xxx",  # 替换真实 AK
    #sk="xxx",  # 替换真实 SK
    api_key="62943a2c-7dc1-4492-ae60-f4de1c6a98b0",
    timeout=1800,
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

def read_prompt_file(file_path):
    """读取本地提示词文件（支持txt/json）"""
    try:
        if file_path.endswith('.json'):
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f).get('prompt', '')
        else:  # 默认为文本文件
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except Exception as e:
        print(f"读取文件错误: {str(e)}")
        exit(1)


# 调用Ark API时添加LLM参数
# R1: ep-20250313174439-fcjxh; model='deepseek-r1-250120'
# V3: ep-20250325163212-9lxdg; model='deepseek-v3-241226'

def generate_summary(prompt_text):
    messages = [
        {"role": "system",
         "content": "You are an expert in pharmaceutical engineering specializing in solid dosage design and development."},
        {"role": "user", "content": prompt_text}  # 动态插入本地读取内容
    ]

    return client.chat.completions.create(
        model='deepseek-r1-250120',
        messages=messages,
        # ===== 新增参数区 =====
        temperature=0.7,  # 控制随机性 (0~1, 默认0.7)
        max_tokens=1000,  # 最大生成长度
        top_p=0.9,  # 核采样阈值 (0~1)
        stream=False,  # 关闭流式传输
        frequency_penalty=0.2,  # 减少重复词 (-2.0~2.0)
        presence_penalty=0.1  # 避免新话题 (-2.0~2.0)
        # =====================
    )

# 主程序流程
if __name__ == "__main__":
    # 1. 读取提示词
    user_prompt = read_prompt_file("zeroshot_CoT.txt")

    # 2. 调用生成函数
    completion = generate_summary(user_prompt)

    # 3. 输出结果
    print("----- Recipe -----")
    print(completion.choices[0].message.content)
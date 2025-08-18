from volcenginesdkarkruntime import Ark


client = Ark(
    api_key="xxxxxxx",
    timeout=1800,
    base_url="xxxxxx"
)

def read_prompt_file(file_path):
 
    try:
        if file_path.endswith('.json'):
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f).get('prompt', '')
        else:  
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except Exception as e:
        print(f"读取文件错误: {str(e)}")
        exit(1)

def generate_summary(prompt_text):
    messages = [
        {"role": "system",
         "content": "You are an expert in pharmaceutical engineering specializing in solid dosage design and development."},
        {"role": "user", "content": prompt_text}  # 动态插入本地读取内容
    ]

    return client.chat.completions.create(
        model='deepseek-r1-250120',
        messages=messages,
       
        temperature=0.7,  
        max_tokens=1000,  
        top_p=0.9,  
        stream=False,  
        frequency_penalty=0.2,  
        presence_penalty=0.1  
        # =====================
    )


if __name__ == "__main__":
 
    user_prompt = read_prompt_file("0710-ZS-CoT.txt")

    completion = generate_summary(user_prompt)

    print("----- Recipe -----")
    print(completion.choices[0].message.content)
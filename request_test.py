import requests
import json

def curl_way_ds():
  try:
    url = "https://api.deepseek.com/v1/chat/completions"
    api_key = "sk-25e969cec8f7407b9ad1ddd7686b940c"  # 替换为你的实际 API Key

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ],
        "stream": False
    }

    response = requests.post(url, headers=headers, json=data)
    print(response.content.decode('gbk'))

    if response.status_code == 200:
        result = response.json()
        print(result["choices"][0]["message"]["content"])
    else:
        print(f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}")
    pass
  except Exception as e:
    print(f"对话链执行错误: {str(e)}")
    raise

if __name__ == "__main__":
    curl_way_ds()
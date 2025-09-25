import requests

def main():
    url = "http://localhost:9696/predict"
    
    # 让用户输入图片URL
    img_url = input("请输入要分析的图片URL: ").strip()
    
    if not img_url:
        print("错误: URL不能为空")
        return
    
    # 构造请求数据
    data = {"query": img_url}
    
    try:
        # 发送POST请求
        response = requests.post(url, json=data, timeout=30)
        
        # 检查HTTP状态码
        if response.status_code == 200:
            score = response.json()
            print("预测成功:")
            print(score)
        elif response.status_code == 400:
            print("错误: 请求参数不正确")
            print(response.json())
        elif response.status_code == 404:
            print("错误: 服务端点不存在")
        else:
            print(f"错误: 服务器返回状态码 {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到服务器，请检查服务是否启动")
    except requests.exceptions.Timeout:
        print("错误: 请求超时")
    except requests.exceptions.RequestException as e:
        print(f"网络错误: {e}")
    except ValueError:
        print("错误: 服务器返回的数据格式不正确")

if __name__ == "__main__":
    main()

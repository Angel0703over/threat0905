import json
import re


def response_extractor(response):
    """
    提取大模型返回的json格式的结果
    :param response: 大模型返回的json格式的回复
    :return:
    """
    regex = r'```json\n(.*?)\n```'
    json_s = re.search(regex, response, re.DOTALL).group(1)
    result = json.loads(json_s)
    return result

def result_print(desp, result: list):
    if result[0] == "200":
        print(desp+"调用成功!")
        return result[1]
    else:
        print(desp+"调用失败!")
        return


if __name__ == '__main__':
    pass


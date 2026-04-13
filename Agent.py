"""
AI代理基础类模块

该模块定义了一个通用的AI代理基础类，用于与OpenAI API进行交互。
支持多种AI模型、重试机制、并发API调用等功能。
"""

# 标准库导入
import json
import os
import random
import time
from urllib import error as urllib_error
from urllib import request as urllib_request

# 第三方库导入
import yaml

try:
    from tenacity import retry, wait_fixed, stop_after_attempt  # 重试机制库
except ImportError:  # pragma: no cover - 依赖缺失时退化为无重试
    def retry(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def wait_fixed(*args, **kwargs):
        return None

    def stop_after_attempt(*args, **kwargs):
        return None

try:
    from openai import OpenAI  # OpenAI官方客户端库
except ImportError:  # pragma: no cover - 依赖缺失时退化为运行时告警
    OpenAI = None

# ============================ 全局配置 ============================
# 默认系统提示词
sys_default_prompt = "You are a helpful assistant."


class BaseAgent:
    """
    AI代理基础类

    该类封装了与OpenAI API的交互逻辑，提供了统一的接口来调用各种语言模型。
    支持多个API密钥的随机选择、重试机制、参数可配置等特性。

    Attributes:
        config: 从配置文件加载的配置信息
        api_keys: API密钥列表
        model_name: 使用的模型名称
        base_url: API基础URL
        default_system_prompt: 默认系统提示词
        client: OpenAI客户端实例
    """

    def __init__(
        self, system_prompt=sys_default_prompt, config_path=None
    ):
        """
        初始化AI代理

        Args:
            system_prompt (str): 系统提示词，定义AI的角色和行为模式
            config_path (str): 配置文件路径，包含API密钥、模型名称等信息
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        resolved_config_path = config_path
        if resolved_config_path is None:
            default_config_path = os.path.join(current_dir, "config", "api.yaml")
            if os.path.exists(default_config_path):
                resolved_config_path = "config/api.yaml"

        # 如果没有提供配置文件，使用默认的ollama配置
        if resolved_config_path is None:
            self.config = {
                "api_key": ["ollama"],
                "model_name": "qwen2.5:7b",
                "base_url": "http://localhost:11434/v1"
            }
        else:
            # 获取配置文件的绝对路径
            config_file_path = os.path.join(current_dir, resolved_config_path)

            # 加载配置文件
            with open(config_file_path, "r") as config_file:
                self.config = yaml.load(config_file, Loader=yaml.FullLoader)

        # 提取配置信息
        self.api_keys = self.config["api_key"]  # API密钥列表
        self.model_name = self.config["model_name"]  # 模型名称
        self.base_url = self.config["base_url"]  # API基础URL
        self.default_system_prompt = system_prompt  # 默认系统提示词
        self.ollama_config = self.config.get("ollama", {}) or {}
        self.is_ollama_compatible = (
    self.api_keys == ["ollama"]
            or self.base_url.startswith("http://localhost:11434")
            or self.base_url.startswith("http://127.0.0.1:11434")
        )
        self.use_native_ollama_api = bool(
            self.is_ollama_compatible and self.ollama_config.get("use_native_api", False)
        )
        self.ollama_native_base_url = str(
            self.ollama_config.get("native_base_url")
            or self.base_url.replace("/v1", "")
        ).rstrip("/")

        self.client = None
        if self.is_ollama_compatible:
            self.client = None
        elif OpenAI is not None:
            # 初始化OpenAI客户端，随机选择一个API密钥
            self.client = OpenAI(
                api_key=random.choice(self.api_keys),  # 从多个密钥中随机选择
                base_url=self.base_url,  # 设置API基础URL
            )
        elif not self.is_ollama_compatible:
            raise ImportError("未安装 openai 依赖，无法创建 OpenAI 模式 BaseAgent 实例")

    def __post_process(self, response):
        """
        处理OpenAI API的响应数据

        从原始API响应中提取有用信息，包括生成的文本内容和使用的token数量。

        Args:
            response: OpenAI API返回的原始响应对象

        Returns:
            dict: 包含响应内容和token使用情况的字典
        """
        if isinstance(response, dict):
            choices = response.get("choices") or []
            content = ""
            if choices:
                message = choices[0].get("message") or {}
                content = message.get("content", "")
            elif isinstance(response.get("message"), dict):
                content = response["message"].get("content", "")

            usage = response.get("usage") or {}
            total_tokens = (
                usage.get("total_tokens")
                or usage.get("eval_count")
                or usage.get("completion_tokens")
                or 0
            )
            return {
                "response": content,
                "total_tokens": total_tokens,
            }

        return {
            "response": response.choices[0].message.content,  # 提取AI生成的文本内容
            "total_tokens": response.usage.total_tokens,  # 提取总的token使用数量
        }

    def __post_json_request(self, url, payload, headers=None, timeout=300):
        headers = headers or {"Content-Type": "application/json"}
        request_body = json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(
            url=url,
            data=request_body,
            headers=headers,
            method="POST",
        )

        with urllib_request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")

        return json.loads(response_body)

    def __call_ollama_native_api(
        self,
        messages,
        temperature=0.9,
        max_tokens=8192,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        **kwargs,
    ):
        """
        直接通过 HTTP 调用 Ollama 原生接口。
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }
        payload.update(kwargs)

        return self.__post_json_request(
            url=f"{self.ollama_native_base_url}/api/chat",
            payload=payload,
            headers={"Content-Type": "application/json"},
        )

    def __call_ollama_compatible_api(
        self,
        messages,
        temperature=0.9,
        max_tokens=8192,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        **kwargs,
    ):
        """
        通过 HTTP 调用 Ollama 的 OpenAI 兼容接口。
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False,
        }
        payload.update(kwargs)

        headers = {"Content-Type": "application/json"}
        api_key = random.choice(self.api_keys) if self.api_keys else ""
        if api_key and api_key != "ollama":
            headers["Authorization"] = f"Bearer {api_key}"

        return self.__post_json_request(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            payload=payload,
            headers=headers,
        )

    def __call_ollama_api(
        self,
        messages,
        temperature=0.9,
        max_tokens=8192,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        **kwargs,
    ):
        """
        统一处理 Ollama 调用，优先使用配置指定的原生接口。
        """
        try:
            if self.use_native_ollama_api:
                return self.__call_ollama_native_api(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    **kwargs,
                )

            return self.__call_ollama_compatible_api(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                **kwargs,
            )
        except urllib_error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="ignore").strip()
            message = error_body or str(e)
            if self.use_native_ollama_api and e.code == 404:
                try:
                    return self.__call_ollama_compatible_api(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        **kwargs,
                    )
                except urllib_error.HTTPError as compat_exc:
                    compat_body = compat_exc.read().decode(
                        "utf-8", errors="ignore"
                    ).strip()
                    compat_message = compat_body or str(compat_exc)
                    raise RuntimeError(
                        f"Ollama 原生接口和兼容接口均请求失败: {compat_message}"
                    ) from compat_exc
            raise RuntimeError(f"Ollama 接口请求失败: {message}") from e
        except urllib_error.URLError as e:
            target_url = (
                self.ollama_native_base_url if self.use_native_ollama_api else self.base_url
            )
            raise RuntimeError(
                f"无法连接 Ollama 服务 {target_url}: {e.reason}"
            ) from e
        except json.JSONDecodeError as e:
            raise RuntimeError("Ollama 返回了无法解析的 JSON 响应") from e

    @retry(wait=wait_fixed(1000), stop=stop_after_attempt(10))
    def __call_api(
        self,
        messages,
        temperature=0.9,
        max_tokens=8192,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        **kwargs,
    ):
        """
        调用OpenAI API并获取响应

        使用tenacity库实现重试机制，在API调用失败时会自动重试。
        重试间隔为1秒，最多重试10次。

        Args:
            messages (list): 对话消息列表
            temperature (float): 温度参数，控制输出的随机性
            max_tokens (int): 最大生成token数量
            top_p (float): 核采样参数
            frequency_penalty (float): 频率惩罚参数
            presence_penalty (float): 存在惩罚参数
            **kwargs: 其他可选参数

        Returns:
            response: OpenAI API的原始响应对象

        Raises:
            Exception: 当API调用失败时抛出异常
        """
        try:
            if self.client is None:
                return self.__call_ollama_api(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    **kwargs,
                )

            # 使用OpenAI客户端发送聊天完成请求
            response = self.client.chat.completions.create(
                model=self.model_name,  # 模型名称
                messages=messages,  # 对话消息
                temperature=temperature,  # 控制输出的创意性
                max_tokens=max_tokens,  # 最大生成长度
                top_p=top_p,  # 核采样参数
                frequency_penalty=frequency_penalty,  # 频率惩罚
                presence_penalty=presence_penalty,  # 存在惩罚
                **kwargs,  # 其他参数
            )
            return response
        except Exception as e:
            # 记录API错误信息
            print(f"[API错误] {str(e)}")
            raise  # 重新抛出异常，触发重试机制

    def get_response(
        self,
        user_input=None,
        system_prompt=None,
        temperature=0.9,
        max_tokens=4096,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        debug=False,
        messages=None,  # 新增 messages 参数
        **kwargs,
    ):
        """
        获取AI的响应，支持多种输入模式

        该方法是主要的对外接口，支持传入单个用户输入或完整的对话消息列表。
        具有灵活的参数配置和错误处理机制。

        Args:
            user_input (str, optional): 用户输入的文本内容
            system_prompt (str, optional): 系统提示词，默认使用初始化时的提示词
            temperature (float): 温度参数，控制输出的随机性 (0-1)
            max_tokens (int): 最大生成token数量
            top_p (float): 核采样参数 (0-1)
            frequency_penalty (float): 频率惩罚参数 (-2.0-2.0)
            presence_penalty (float): 存在惩罚参数 (-2.0-2.0)
            debug (bool): 是否开启调试模式，会打印响应内容
            messages (list, optional): 完整的对话消息列表
            **kwargs: 其他传递给API的参数

        Returns:
            dict: 包含响应内容和token使用情况的字典，
                  或在出错时返回包含error字段的字典
        """
        try:
            # 使用默认系统提示词（如果未提供）
            if system_prompt is None:
                system_prompt = self.default_system_prompt

            # 初始化消息列表（如果未提供）
            if messages is None:
                messages = []

            # 检查并添加系统提示词（如果不存在）
            if not any(msg["role"] == "system" for msg in messages):
                messages.insert(0, {"role": "system", "content": system_prompt})

            # 添加用户输入到消息列表（如果提供了user_input）
            if user_input is not None:
                messages.append({"role": "user", "content": user_input})

            # 清理kwargs中的messages参数，避免参数冲突
            if "messages" in kwargs:
                kwargs.pop("messages")

            # 调用底层API接口
            response = self.__call_api(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                **kwargs,
            )

            # 处理API响应，提取有用信息
            result = self.__post_process(response)

            # 调试模式：以绿色文字打印响应内容
            if debug:
                print("\033[92m" + f"[响应] {result['response']}" + "\033[0m")

            # 返回处理后的结果
            return result

        except Exception as e:
            # 错误处理：以红色文字打印错误信息
            print("\033[91m" + f"[错误] {str(e)}" + "\033[0m")
            return {"error": f"Error: {str(e)}"}

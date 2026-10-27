"""
LLM API 连接测试脚本
用于验证 API Key、Base URL 和模型配置是否正确
"""
import os
import sys
from openai import OpenAI
from main import llm_request_options

# 配置（优先从环境变量读取，否则使用默认值）
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api-inference.modelscope.cn/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3.8-27B")

def test_llm_connection():
    print("=" * 50)
    print("🔧 LLM API 连接测试")
    print("=" * 50)
    print(f"📍 Base URL: {LLM_BASE_URL}")
    print(f"🤖 Model: {LLM_MODEL}")
    print(f"🔑 API Key: {LLM_API_KEY[:8]}...{LLM_API_KEY[-4:]}" if len(LLM_API_KEY) > 12 else "⚠️ API Key 太短或未设置")
    print("=" * 50)
    
    if not LLM_API_KEY or LLM_API_KEY == "your-api-key-here":
        print("❌ 错误: 请设置 LLM_API_KEY 环境变量")
        print("   示例: $env:LLM_API_KEY = 'sk-xxx'")
        return False
    
    try:
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, max_retries=0)
        
        print("\n📤 发送测试请求...")
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "请用一句话介绍你自己"}],
            temperature=0.1,
            **llm_request_options(LLM_BASE_URL, LLM_MODEL),
        )
        
        print("\n📥 响应详情:")
        print(f"   - Choices 数量: {len(response.choices)}")
        
        if response.choices:
            choice = response.choices[0]
            print(f"   - Finish Reason: {choice.finish_reason}")
            print(f"   - Content: {choice.message.content}")
            if choice.finish_reason == "length":
                print("❌ 输出达到 token 上限，回答可能被截断。")
                return False
            if not (choice.message.content or "").strip():
                print("❌ 未返回最终回答；请检查输出 token 限制和思考模式配置。")
                return False
            print("\n✅ LLM API 连接正常!")
            return True
        else:
            print("\n❌ 错误: API 返回空的 choices")
            print("   可能原因:")
            print("   1. 模型名称错误")
            print("   2. 账户余额不足")
            print("   3. API 请求被拒绝")
            return False
            
    except Exception as e:
        print(f"\n❌ 连接失败: {e}")
        print("\n   可能原因:")
        print("   1. API Key 无效或过期")
        print("   2. Base URL 错误")
        print("   3. 网络连接问题")
        return False

def test_json_mode():
    """测试 JSON 模式是否正常工作"""
    print("\n" + "=" * 50)
    print("🔧 JSON 模式测试")
    print("=" * 50)
    
    try:
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, max_retries=0)
        
        print("\n📤 发送 JSON 格式请求...")
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "返回一个JSON对象，包含name和age字段"}],
            response_format={"type": "json_object"},
            temperature=0.1,
            **llm_request_options(LLM_BASE_URL, LLM_MODEL),
        )
        
        if response.choices:
            if response.choices[0].finish_reason == "length":
                print("❌ JSON 输出达到 token 上限，结果可能被截断。")
                return False
            content = response.choices[0].message.content
            print(f"   - Raw Response: {content}")
            
            import json
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError("响应不是 JSON 对象。")
            print(f"   - Parsed: {result}")
            print("\n✅ JSON 模式正常!")
            return True
        else:
            print("\n❌ JSON 模式测试失败: 空响应")
            return False
            
    except Exception as e:
        print(f"\n❌ JSON 模式测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_llm_connection() and test_json_mode()
    sys.exit(0 if success else 1)

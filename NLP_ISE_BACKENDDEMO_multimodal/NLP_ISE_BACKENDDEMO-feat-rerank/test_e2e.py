"""
端到端测试脚本

测试完整的 Agent 工作流：
1. 意图识别/路由决策
2. 策略执行（local/web/hybrid）
3. 证据聚合
4. 答案生成
5. 响应构建

使用方法：
1. 确保服务已启动：uvicorn backend.app:app --reload
2. 运行此脚本：python test_e2e.py
"""

import json
import requests
import time
from typing import Dict, List, Any
from datetime import datetime

# API 基础URL
BASE_URL = "http://127.0.0.1:8000"

# 测试用例：覆盖不同的路由策略和场景
TEST_CASES = [
    {
        "name": "本地知识库问题 - Local策略",
        "question": "Who is Dr. Elara Vance?",
        "expected_policy": "local",
        "expected_fields": ["answer", "sources", "routing", "latency_ms", "confidence"],
    },
    {
        "name": "本地知识库问题 - Sereleia",
        "question": "Tell me about Sereleia",
        "expected_policy": "local",
        "expected_fields": ["answer", "sources", "routing", "latency_ms", "confidence"],
    },
    {
        "name": "实时问题 - Web策略",
        "question": "What's the weather today?",
        "expected_policy": "web",
        "expected_fields": ["answer", "sources", "routing", "latency_ms", "confidence"],
        "skip_if_no_tavily": True,  # 如果没有 Tavily API，跳过此测试
    },
    {
        "name": "混合问题 - Hybrid策略",
        "question": "Explain the Vance Protocol and give the latest real-world impact",
        "expected_policy": "hybrid",
        "expected_fields": ["answer", "sources", "routing", "latency_ms", "confidence"],
        "skip_if_no_tavily": True,
    },
    {
        "name": "模糊问题 - LLM判断",
        "question": "What is machine learning?",
        "expected_policy": None,  # 由 LLM 判断，不固定
        "expected_fields": ["answer", "sources", "routing", "latency_ms", "confidence"],
        "skip_if_no_tavily": True,
    },
]

# 多模态测试用例：图像+文本
MULTIMODAL_TEST_CASES = [
    {
        "name": "图像内容描述 - 基础场景",
        "image_filename": "hkust.png",
        "question": "请详细描述这张图片的内容，包括场景、物体和氛围",
        "expected_fields": ["answer", "image_path", "query", "latency_ms", "confidence"],
        "min_answer_length": 50,
    },
    {
        "name": "图像对象识别 - 多物体检测",
        "image_filename": "snack.png",
        "question": "列出图片中所有可见的物体",
        "expected_fields": ["answer", "image_path", "query", "latency_ms", "confidence"],
        "min_answer_length": 30,
    },
    {
        "name": "图像文字提取 - OCR能力",
        "image_filename": "error_info.png",
        "question": "请提取并整理图片中的所有文字内容",
        "expected_fields": ["answer", "image_path", "query", "latency_ms", "confidence"],
        "min_answer_length": 10,
    },
]

# 测试图像目录
TEST_IMAGES_DIR = "test_images"

class Colors:
    """终端颜色"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """打印标题"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}\n")


def print_success(text: str):
    """打印成功信息"""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")


def print_error(text: str):
    """打印错误信息"""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")


def print_warning(text: str):
    """打印警告信息"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")


def print_info(text: str):
    """打印信息"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")


def test_health_check() -> bool:
    """检查服务是否正常运行"""
    try:
        response = requests.get(f"{BASE_URL}/healthz", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def test_full_workflow(question: str) -> Dict[str, Any]:
    """测试完整工作流"""
    url = f"{BASE_URL}/api/agent/answer"
    payload = {"q": question}
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=60)  # 增加超时时间
        elapsed_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        response.raise_for_status()
        result = response.json()
        result["_test_elapsed_ms"] = elapsed_time
        return result
    except requests.exceptions.Timeout:
        return {"error": "请求超时（>60秒）"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def test_multimodal_workflow(image_path: str, question: str) -> Dict[str, Any]:
    """测试多模态（图像+文本）工作流"""
    url = f"{BASE_URL}/api/agent/multimodal"
    payload = {
        "q": question,
        "image_path": image_path
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=120)  # 视觉模型需要更长时间
        elapsed_time = (time.time() - start_time) * 1000
        
        response.raise_for_status()
        result = response.json()
        result["_test_elapsed_ms"] = elapsed_time
        return result
    except requests.exceptions.Timeout:
        return {"error": "请求超时（>120秒）"}
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return {"error": "多模态接口不存在，请检查是否已实现 /api/agent/multimodal 端点"}
        return {"error": f"HTTP错误 {e.response.status_code}: {e.response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def validate_multimodal_response(response: Dict, expected_fields: List[str], 
                                  min_answer_length: int = 0) -> tuple[bool, List[str]]:
    """验证多模态响应结构"""
    errors = []
    
    # 检查必需字段
    for field in expected_fields:
        if field not in response:
            errors.append(f"缺少字段: {field}")
    
    # 验证答案长度
    if "answer" in response:
        answer = response["answer"]
        if not isinstance(answer, str):
            errors.append("answer 必须是字符串")
        elif len(answer.strip()) < min_answer_length:
            errors.append(f"答案过短: 期望至少 {min_answer_length} 字符，实际 {len(answer.strip())} 字符")
    
    # 验证置信度
    if "confidence" in response:
        confidence = response["confidence"]
        if not isinstance(confidence, (int, float)):
            errors.append("confidence 必须是数字")
        elif not (0.0 <= confidence <= 1.0):
            errors.append(f"confidence 超出范围 [0.0, 1.0]: {confidence}")
    
    # 验证延迟
    if "latency_ms" in response:
        latency = response["latency_ms"]
        if not isinstance(latency, (int, float)) or latency < 0:
            errors.append(f"latency_ms 无效: {latency}")
    
    return len(errors) == 0, errors


def print_multimodal_response_summary(response: Dict, test_case: Dict):
    """打印多模态响应摘要"""
    print(f"\n{Colors.BOLD}📋 多模态响应摘要{Colors.RESET}")
    print(f"{'-'*80}")
    
    # 图像信息
    if "image_path" in response:
        import os
        image_name = os.path.basename(response["image_path"])
        print(f"{Colors.BOLD}🖼️  图像:{Colors.RESET} {image_name}")
    
    # 问题
    if "query" in response:
        print(f"{Colors.BOLD}❓ 问题:{Colors.RESET} {response['query']}")
    
    # 延迟
    if "latency_ms" in response:
        latency = response["latency_ms"]
        latency_color = Colors.GREEN if latency < 5000 else Colors.YELLOW if latency < 10000 else Colors.RED
        print(f"\n{Colors.BOLD}⏱️  处理时间:{Colors.RESET} {latency_color}{latency} ms{Colors.RESET}")
        if "_test_elapsed_ms" in response:
            print(f"  实际耗时: {response['_test_elapsed_ms']:.2f} ms")
    
    # 置信度
    if "confidence" in response:
        confidence = response["confidence"]
        conf_color = Colors.GREEN if confidence >= 0.7 else Colors.YELLOW if confidence >= 0.4 else Colors.RED
        print(f"\n{Colors.BOLD}📊 置信度:{Colors.RESET} {conf_color}{confidence:.2f}{Colors.RESET}")
    
    # 答案预览
    if "answer" in response:
        answer = response["answer"]
        preview = answer[:300] + "..." if len(answer) > 300 else answer
        print(f"\n{Colors.BOLD}💬 答案 ({len(answer)} 字符):{Colors.RESET}")
        print(f"  {preview}")


def run_multimodal_test_case(test_case: Dict) -> Dict[str, Any]:
    """运行单个多模态测试用例"""
    import os
    
    print_header(f"多模态测试: {test_case['name']}")
    
    # 构建图像路径
    image_path = os.path.join(TEST_IMAGES_DIR, test_case["image_filename"])
    abs_image_path = os.path.abspath(image_path)
    
    # 检查图像是否存在
    if not os.path.exists(abs_image_path):
        print_error(f"图像文件不存在: {abs_image_path}")
        return {"passed": False, "error": f"图像文件不存在: {test_case['image_filename']}"}
    
    print(f"{Colors.BOLD}图像:{Colors.RESET} {test_case['image_filename']}")
    print(f"{Colors.BOLD}问题:{Colors.RESET} {test_case['question']}")
    
    # 执行测试
    print_info("处理多模态查询...")
    response = test_multimodal_workflow(abs_image_path, test_case["question"])
    
    # 检查错误
    if "error" in response:
        print_error(f"请求失败: {response['error']}")
        return {"passed": False, "error": response["error"]}
    
    # 验证响应
    is_valid, errors = validate_multimodal_response(
        response,
        test_case["expected_fields"],
        test_case.get("min_answer_length", 0)
    )
    
    if not is_valid:
        print_error("响应验证失败:")
        for error in errors:
            print_error(f"  - {error}")
        return {"passed": False, "errors": errors, "response": response}
    
    print_success("响应验证通过")
    
    # 打印摘要
    print_multimodal_response_summary(response, test_case)
    
    # 检查答案质量
    if "answer" in response:
        answer = response["answer"]
        if len(answer.strip()) == 0:
            print_warning("答案为空")
        elif any(keyword in answer for keyword in ["无法", "错误", "抱歉", "暂时"]):
            print_warning("答案可能包含错误信息")
        else:
            print_success(f"答案生成成功（{len(answer)} 字符）")
    
    return {"passed": True, "response": response}

def validate_response_structure(response: Dict, expected_fields: List[str]) -> tuple[bool, List[str]]:
    """验证响应结构"""
    errors = []
    
    for field in expected_fields:
        if field not in response:
            errors.append(f"缺少字段: {field}")
    
    # 验证 routing 结构
    if "routing" in response:
        routing = response["routing"]
        if "policy" not in routing:
            errors.append("routing 缺少 policy 字段")
        if "rationale" not in routing:
            errors.append("routing 缺少 rationale 字段")
        
        # 验证 policy 值
        if "policy" in routing:
            policy = routing["policy"]
            if policy not in ["local", "web", "hybrid"]:
                errors.append(f"无效的 policy 值: {policy}")
    
    # 验证 latency_ms 结构
    if "latency_ms" in response:
        latency = response["latency_ms"]
        required_latency_fields = ["retrieve", "rerank", "generate", "total"]
        for field in required_latency_fields:
            if field not in latency:
                errors.append(f"latency_ms 缺少字段: {field}")
    
    # 验证 sources 结构
    if "sources" in response:
        sources = response["sources"]
        if not isinstance(sources, list):
            errors.append("sources 必须是列表")
        else:
            for i, source in enumerate(sources):
                if "type" not in source:
                    errors.append(f"sources[{i}] 缺少 type 字段")
                elif source["type"] not in ["local", "web"]:
                    errors.append(f"sources[{i}] 无效的 type 值: {source['type']}")
    
    # 验证 confidence 范围
    if "confidence" in response:
        confidence = response["confidence"]
        if not isinstance(confidence, (int, float)):
            errors.append("confidence 必须是数字")
        elif not (0.0 <= confidence <= 1.0):
            errors.append(f"confidence 超出范围 [0.0, 1.0]: {confidence}")
    
    return len(errors) == 0, errors


def print_response_summary(response: Dict, test_case: Dict):
    """打印响应摘要"""
    print(f"\n{Colors.BOLD}📋 响应摘要{Colors.RESET}")
    print(f"{'-'*80}")
    
    # 路由信息
    if "routing" in response:
        routing = response["routing"]
        policy = routing.get("policy", "unknown")
        rationale = routing.get("rationale", "无理由")
        policy_color = Colors.GREEN if policy == test_case.get("expected_policy") else Colors.YELLOW
        print(f"{Colors.BOLD}路由策略:{Colors.RESET} {policy_color}{policy}{Colors.RESET}")
        if test_case.get("expected_policy"):
            expected = test_case["expected_policy"]
            status = "✅" if policy == expected else "⚠️"
            print(f"  期望: {expected} {status}")
        print(f"{Colors.BOLD}决策理由:{Colors.RESET} {rationale}")
    
    # 延迟信息
    if "latency_ms" in response:
        latency = response["latency_ms"]
        print(f"\n{Colors.BOLD}⏱️  延迟统计:{Colors.RESET}")
        print(f"  检索: {latency.get('retrieve', 0)} ms")
        print(f"  重排: {latency.get('rerank', 0)} ms")
        print(f"  生成: {latency.get('generate', 0)} ms")
        print(f"  总计: {Colors.BOLD}{latency.get('total', 0)} ms{Colors.RESET}")
        if "_test_elapsed_ms" in response:
            test_time = response["_test_elapsed_ms"]
            diff = abs(test_time - latency.get("total", 0))
            print(f"  实际: {test_time:.2f} ms (差异: {diff:.2f} ms)")
    
    # 置信度
    if "confidence" in response:
        confidence = response["confidence"]
        conf_color = Colors.GREEN if confidence >= 0.7 else Colors.YELLOW if confidence >= 0.4 else Colors.RED
        print(f"\n{Colors.BOLD}📊 置信度:{Colors.RESET} {conf_color}{confidence:.2f}{Colors.RESET}")
    
    # 来源统计
    if "sources" in response:
        sources = response["sources"]
        local_sources = [s for s in sources if s.get("type") == "local"]
        web_sources = [s for s in sources if s.get("type") == "web"]
        print(f"\n{Colors.BOLD}📚 来源统计:{Colors.RESET}")
        print(f"  本地来源: {len(local_sources)} 个")
        print(f"  网络来源: {len(web_sources)} 个")
        print(f"  总计: {len(sources)} 个")
    
    # 答案预览
    if "answer" in response:
        answer = response["answer"]
        preview = answer[:200] + "..." if len(answer) > 200 else answer
        print(f"\n{Colors.BOLD}💬 答案预览:{Colors.RESET}")
        print(f"  {preview}")


def run_test_case(test_case: Dict, skip_if_no_tavily: bool = False) -> Dict[str, Any]:
    """运行单个测试用例"""
    print_header(f"测试: {test_case['name']}")
    
    print(f"{Colors.BOLD}问题:{Colors.RESET} {test_case['question']}")
    
    # 检查是否需要跳过
    if skip_if_no_tavily:
        print_info("此测试需要 Tavily API，如果未配置将可能失败")
    
    # 执行测试
    print_info("执行完整工作流...")
    response = test_full_workflow(test_case["question"])
    
    # 检查错误
    if "error" in response:
        print_error(f"请求失败: {response['error']}")
        return {"passed": False, "error": response["error"]}
    
    # 验证响应结构
    is_valid, errors = validate_response_structure(response, test_case["expected_fields"])
    
    if not is_valid:
        print_error("响应结构验证失败:")
        for error in errors:
            print_error(f"  - {error}")
        return {"passed": False, "errors": errors, "response": response}
    
    print_success("响应结构验证通过")
    
    # 验证路由策略（如果指定了期望值）
    if test_case.get("expected_policy"):
        actual_policy = response.get("routing", {}).get("policy")
        expected_policy = test_case["expected_policy"]
        if actual_policy != expected_policy:
            print_warning(f"路由策略不匹配: 期望 {expected_policy}, 实际 {actual_policy}")
    
    # 打印响应摘要
    print_response_summary(response, test_case)
    
    # 检查答案是否为空
    if "answer" in response:
        answer = response["answer"]
        if not answer or len(answer.strip()) == 0:
            print_warning("答案为空")
        elif any(keyword in answer for keyword in ["无法", "错误", "抱歉", "暂时"]):
            print_warning("答案可能包含错误或降级信息")
        else:
            print_success(f"答案长度: {len(answer)} 字符")
    
    return {"passed": True, "response": response}


def main():
    """主测试函数"""
    print_header("端到端测试 - NLP Agent")
    
    # 检查服务状态
    print(f"{Colors.BOLD}[1/4] 检查服务状态...{Colors.RESET}")
    if not test_health_check():
        print_error("服务未运行！请先启动服务：")
        print("  uvicorn backend.app:app --reload")
        return
    print_success("服务正常运行")
    
    # 运行常规测试用例
    print(f"\n{Colors.BOLD}[2/4] 运行 {len(TEST_CASES)} 个常规测试用例...{Colors.RESET}")
    results = []
    passed_count = 0
    failed_count = 0
    skipped_count = 0
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n{Colors.BOLD}[{i}/{len(TEST_CASES)}]{Colors.RESET}")
        
        skip = test_case.get("skip_if_no_tavily", False)
        result = run_test_case(test_case, skip_if_no_tavily=skip)
        
        if "error" in result:
            if "Tavily" in str(result.get("error", "")):
                skipped_count += 1
                print_warning("测试被跳过（缺少 Tavily API）")
            else:
                failed_count += 1
        elif result.get("passed"):
            passed_count += 1
        else:
            failed_count += 1
        
        results.append({
            "test_case": test_case["name"],
            "test_type": "regular",
            "result": result
        })
        
        if i < len(TEST_CASES):
            time.sleep(1)
    
    # ========== 新增：运行多模态测试 ==========
    print(f"\n{Colors.BOLD}[3/4] 运行 {len(MULTIMODAL_TEST_CASES)} 个多模态测试用例...{Colors.RESET}")
    
    import os
    if not os.path.exists(TEST_IMAGES_DIR):
        print_warning(f"测试图像目录不存在: {TEST_IMAGES_DIR}")
        print_info("跳过多模态测试。如需测试，请创建目录并添加测试图像。")
        multimodal_skipped = len(MULTIMODAL_TEST_CASES)
    else:
        multimodal_passed = 0
        multimodal_failed = 0
        multimodal_skipped = 0
        
        for i, test_case in enumerate(MULTIMODAL_TEST_CASES, 1):
            print(f"\n{Colors.BOLD}[多模态 {i}/{len(MULTIMODAL_TEST_CASES)}]{Colors.RESET}")
            
            result = run_multimodal_test_case(test_case)
            
            if "error" in result:
                if "不存在" in str(result.get("error", "")):
                    multimodal_skipped += 1
                    print_warning(f"跳过测试（图像文件缺失）")
                else:
                    multimodal_failed += 1
            elif result.get("passed"):
                multimodal_passed += 1
            else:
                multimodal_failed += 1
            
            results.append({
                "test_case": test_case["name"],
                "test_type": "multimodal",
                "image": test_case["image_filename"],
                "result": result
            })
            
            if i < len(MULTIMODAL_TEST_CASES):
                print_info("等待2秒...")
                time.sleep(2)
        
        # 更新总计数
        passed_count += multimodal_passed
        failed_count += multimodal_failed
        skipped_count += multimodal_skipped
    
    # 打印总结
    print_header("测试总结")
    total_tests = len(TEST_CASES) + len(MULTIMODAL_TEST_CASES)
    print(f"{Colors.BOLD}总测试数:{Colors.RESET} {total_tests}")
    print(f"  常规测试: {len(TEST_CASES)} 个")
    print(f"  多模态测试: {len(MULTIMODAL_TEST_CASES)} 个")
    print(f"\n{Colors.GREEN}✅ 通过: {passed_count}{Colors.RESET}")
    if failed_count > 0:
        print(f"{Colors.RED}❌ 失败: {failed_count}{Colors.RESET}")
    if skipped_count > 0:
        print(f"{Colors.YELLOW}⏭️  跳过: {skipped_count}{Colors.RESET}")
    
    # 成功率
    if total_tests > 0:
        success_rate = (passed_count / total_tests) * 100
        color = Colors.GREEN if success_rate >= 80 else Colors.YELLOW if success_rate >= 50 else Colors.RED
        print(f"\n{Colors.BOLD}成功率: {color}{success_rate:.1f}%{Colors.RESET}")
    
    # 详细结果（失败的测试）
    if failed_count > 0:
        print(f"\n{Colors.BOLD}详细结果:{Colors.RESET}")
        for result in results:
            if not result["result"].get("passed") and "error" not in result["result"]:
                test_type = result.get("test_type", "regular")
                icon = "🖼️" if test_type == "multimodal" else "📝"
                print(f"\n{Colors.RED}❌ {icon} {result['test_case']}{Colors.RESET}")
                if "error" in result["result"]:
                    print(f"  错误: {result['result']['error']}")
                if "errors" in result["result"]:
                    for error in result["result"]["errors"]:
                        print(f"  - {error}")
    
    # 保存结果
    print(f"\n{Colors.BOLD}[4/4] 保存测试结果...{Colors.RESET}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_e2e_results_{timestamp}.json"
    try:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "summary": {
                    "total": total_tests,
                    "regular_tests": len(TEST_CASES),
                    "multimodal_tests": len(MULTIMODAL_TEST_CASES),
                    "passed": passed_count,
                    "failed": failed_count,
                    "skipped": skipped_count,
                    "success_rate": success_rate if total_tests > 0 else 0
                },
                "results": results
            }, f, ensure_ascii=False, indent=2)
        print_success(f"详细结果已保存到: {results_file}")
    except Exception as e:
        print_warning(f"保存结果文件失败: {e}")
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.RESET}")


if __name__ == "__main__":
    main()


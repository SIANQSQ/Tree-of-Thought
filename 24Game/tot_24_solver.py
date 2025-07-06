import openai
import json
import time
import itertools
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math


class StrategyType(Enum):
    FACTORIZATION = "factorization"
    ADDITION_PATH = "addition_path"
    MULTIPLICATION_PATH = "multiplication_path"
    DIVISION_PATH = "division_path"
    MIXED_OPERATIONS = "mixed_operations"


@dataclass
class ThoughtStep:
    """Represents a single step in the thinking process"""
    step_number: int
    strategy: StrategyType
    expression: str
    value: float
    confidence: float
    reasoning: str
    is_solution: bool = False
    parent_step: Optional[int] = None

    def __post_init__(self):
        self.is_solution = abs(self.value - 24) < 0.001


@dataclass
class ThoughtNode:
    """Represents a node in the Tree-of-Thought"""
    id: str
    expression: str
    value: float
    confidence: float
    reasoning: str
    depth: int
    strategy: StrategyType
    parent_id: Optional[str] = None
    children: List['ThoughtNode'] = field(default_factory=list)
    is_solution: bool = False
    step_details: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.is_solution = abs(self.value - 24) < 0.001


@dataclass
class SolutionResult:
    """Enhanced result with detailed process tracking"""
    success: bool
    solutions: List[str]
    time_taken: float
    steps_count: int
    reasoning_trace: List[str]
    thought_process: List[ThoughtStep]
    strategy_performance: Dict[str, Dict[str, Any]]


class ChineseCommunicationTreeOfThought24Solver:
    """Tree-of-Thought solver with Chinese AI communication"""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        """Initialize with DeepSeek API credentials for legacy OpenAI"""
        openai.api_key = api_key
        openai.api_base = base_url
        self.model = "deepseek-chat"

    def solve_normal_approach(self, numbers: List[int]) -> SolutionResult:
        """Traditional approach with Chinese communication"""
        start_time = time.time()
        reasoning_trace = []
        thought_process = []

        print(f"\n🔍 普通方法求解 - 逐步分析过程")
        print("=" * 60)

        # Step 1: Initial analysis
        step1 = ThoughtStep(
            step_number=1,
            strategy=StrategyType.MIXED_OPERATIONS,
            expression="初始分析",
            value=0,
            confidence=1.0,
            reasoning=f"分析数字 {numbers}，寻找直接通往24的路径"
        )
        thought_process.append(step1)
        print(f"步骤1: {step1.reasoning}")

        # Chinese prompt for AI
        prompt = f"""
        请解决24点问题，使用数字 {numbers}。

        规则：
        - 每个数字必须且只能使用一次
        - 只能使用 +、-、*、/ 运算
        - 可以使用括号
        - 目标结果必须等于24

        请逐步思考并展示你的推理过程：
        1. 首先分析哪些运算可能有效
        2. 系统性地尝试不同组合
        3. 清楚地展示你的计算过程

        请按以下格式回答：
        步骤1: [你的第一次尝试和推理]
        计算: [表达式] = [结果]

        步骤2: [如果需要，你的第二次尝试]
        计算: [表达式] = [结果]

        最终解答: [有效表达式] = 24
        推理过程: [详细说明你的思考过程]
        """

        try:
            print(f"步骤2: 向AI查询系统性解决方案...")
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )

            content = response.choices[0].message.content
            reasoning_trace.append(f"AI回应: {content}")

            # Parse the Chinese response
            steps = self._parse_chinese_normal_steps(content, numbers)
            thought_process.extend(steps)

            # Display each step in Chinese
            for i, step in enumerate(steps, 2):
                print(f"步骤{i}: {step.reasoning}")
                if step.expression != "初始分析":
                    print(f"   表达式: {step.expression}")
                    print(f"   结果: {step.value}")
                    print(f"   置信度: {step.confidence:.1%}")
                    if step.is_solution:
                        print(f"   ✅ 找到解答!")
                    print()

            # Extract solutions
            solutions = []
            for step in thought_process:
                if step.is_solution:
                    solutions.append(step.expression)

            time_taken = time.time() - start_time

            return SolutionResult(
                success=len(solutions) > 0,
                solutions=solutions,
                time_taken=time_taken,
                steps_count=len(thought_process),
                reasoning_trace=reasoning_trace,
                thought_process=thought_process,
                strategy_performance={"normal": {"steps": len(thought_process), "success": len(solutions) > 0}}
            )

        except Exception as e:
            print(f"❌ 普通方法出错: {e}")
            return SolutionResult(
                success=False,
                solutions=[],
                time_taken=time.time() - start_time,
                steps_count=0,
                reasoning_trace=[f"错误: {str(e)}"],
                thought_process=[],
                strategy_performance={}
            )

    def solve_tree_of_thought(self, numbers: List[int], max_depth: int = 4) -> SolutionResult:
        """Enhanced Tree-of-Thought with Chinese communication"""
        start_time = time.time()
        reasoning_trace = []
        all_thought_steps = []
        strategy_performance = {}

        print(f"\n🌳 树状思维方法 - 多策略并行推理")
        print("=" * 60)

        # Phase 1: Generate strategies with Chinese communication
        print(f"\n📋 阶段1: 策略生成与分析")
        print("-" * 40)

        initial_thoughts = self._generate_chinese_initial_thoughts(numbers)
        reasoning_trace.append(f"生成了{len(initial_thoughts)}种策略方法")

        # Display initial strategies in Chinese
        strategy_names = {
            StrategyType.FACTORIZATION: "因式分解策略",
            StrategyType.ADDITION_PATH: "加法路径策略",
            StrategyType.MULTIPLICATION_PATH: "乘法优先策略",
            StrategyType.DIVISION_PATH: "除法创值策略",
            StrategyType.MIXED_OPERATIONS: "混合运算策略"
        }

        for i, thought in enumerate(initial_thoughts, 1):
            strategy_name = strategy_names.get(thought.strategy, thought.strategy.value)
            print(f"策略{i}: {strategy_name}")
            print(f"   推理: {thought.reasoning}")
            print(f"   置信度: {thought.confidence:.1%}")
            print()

        # Phase 2: Explore each strategy
        print(f"\n🔍 阶段2: 策略探索与发展")
        print("-" * 40)

        all_solutions = []
        explored_nodes = 0

        for strategy_idx, thought in enumerate(initial_thoughts):
            if thought.confidence > 0.3:
                strategy_name = strategy_names.get(thought.strategy, thought.strategy.value)
                print(f"\n🎯 探索策略: {strategy_name}")
                print(f"   初始置信度: {thought.confidence:.1%}")

                # Build detailed tree for this strategy
                tree, strategy_steps = self._build_chinese_thought_tree(thought, numbers, max_depth)
                solutions = self._extract_solutions_from_tree(tree)
                all_solutions.extend(solutions)
                explored_nodes += self._count_nodes(tree)
                all_thought_steps.extend(strategy_steps)

                # Track strategy performance
                strategy_performance[thought.strategy.value] = {
                    "solutions_found": len(solutions),
                    "nodes_explored": self._count_nodes(tree),
                    "confidence": thought.confidence,
                    "steps": strategy_steps,
                    "chinese_name": strategy_name
                }

                # Display strategy results in Chinese
                print(f"   📊 结果: 找到{len(solutions)}个解答，探索了{self._count_nodes(tree)}个节点")
                if solutions:
                    for sol in solutions[:2]:
                        print(f"   ✅ 解答: {sol} = 24")

                reasoning_trace.append(
                    f"策略'{strategy_name}': 找到{len(solutions)}个解答，探索了{self._count_nodes(tree)}个节点"
                )

        # Phase 3: Solution verification
        print(f"\n✅ 阶段3: 解答验证与排序")
        print("-" * 40)

        verified_solutions = []
        for sol in set(all_solutions):
            if self._verify_solution(sol, numbers):
                verified_solutions.append(sol)
                print(f"✅ 验证通过: {sol} = 24")

        time_taken = time.time() - start_time

        # Display strategy comparison in Chinese
        self._display_chinese_strategy_comparison(strategy_performance)

        return SolutionResult(
            success=len(verified_solutions) > 0,
            solutions=verified_solutions,
            time_taken=time_taken,
            steps_count=explored_nodes,
            reasoning_trace=reasoning_trace,
            thought_process=all_thought_steps,
            strategy_performance=strategy_performance
        )

    def _generate_chinese_initial_thoughts(self, numbers: List[int]) -> List[ThoughtNode]:
        """Generate initial strategies with Chinese prompts"""
        strategies = [
            (StrategyType.FACTORIZATION,
             f"因式分解策略：分析24的因数（1×24, 2×12, 3×8, 4×6），看如何用{numbers}构造这些因数"),
            (StrategyType.ADDITION_PATH,
             f"加法路径策略：寻找用{numbers}通过加减法组合达到24的方法"),
            (StrategyType.MULTIPLICATION_PATH,
             f"乘法优先策略：以乘法为主要运算，配合{numbers}达到24"),
            (StrategyType.DIVISION_PATH,
             f"除法创值策略：考虑用除法从{numbers}创造有用的中间值"),
            (StrategyType.MIXED_OPERATIONS,
             f"混合运算策略：巧妙结合多种运算，用{numbers}达到24")
        ]

        thoughts = []
        for i, (strategy, description) in enumerate(strategies):
            prompt = f"""
            数字: {numbers}
            策略: {description}

            请详细分析这个策略：
            1. 这些数字的哪些数学性质支持这个策略？
            2. 在这个策略下，哪些具体方法看起来最有希望？
            3. 为这个策略评估置信度（0.0到1.0）

            请按以下格式回答：
            数学分析: [分析这些数字对于此策略的特点]
            有希望的方法: [列出2-3个具体方法]
            置信度: [0.0到1.0的分数]
            推理过程: [解释为什么这个策略有希望或没希望]
            """

            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.7
                )

                content = response.choices[0].message.content
                confidence = self._extract_chinese_confidence(content)
                reasoning = self._extract_chinese_reasoning(content)
                analysis = self._extract_chinese_field(content, "数学分析")
                approaches = self._extract_chinese_field(content, "有希望的方法")

                full_reasoning = f"{analysis} | 方法: {approaches} | {reasoning}"

                thought = ThoughtNode(
                    id=f"strategy_{i}",
                    expression=f"策略: {strategy.value}",
                    value=0.0,
                    confidence=confidence,
                    reasoning=full_reasoning,
                    depth=0,
                    strategy=strategy,
                    step_details=[analysis, approaches, reasoning]
                )
                thoughts.append(thought)

            except Exception as e:
                print(f"⚠️ 生成策略{i}时出错: {e}")
                thought = ThoughtNode(
                    id=f"strategy_{i}",
                    expression=f"策略: {strategy.value}",
                    value=0.0,
                    confidence=0.5,
                    reasoning=description,
                    depth=0,
                    strategy=strategy
                )
                thoughts.append(thought)

        return thoughts

    def _build_chinese_thought_tree(self, root: ThoughtNode, numbers: List[int], max_depth: int) -> Tuple[
        ThoughtNode, List[ThoughtStep]]:
        """Build tree with Chinese communication"""
        strategy_steps = []
        step_counter = 0

        def build_recursive(node: ThoughtNode, current_depth: int) -> ThoughtNode:
            nonlocal step_counter

            if current_depth >= max_depth or node.confidence < 0.2:
                return node

            # Generate children with Chinese reasoning
            children = self._generate_chinese_child_thoughts(node, numbers, step_counter)
            step_counter += len(children)

            # Convert children to ThoughtSteps for tracking
            for child in children:
                step = ThoughtStep(
                    step_number=step_counter,
                    strategy=child.strategy,
                    expression=child.expression,
                    value=child.value,
                    confidence=child.confidence,
                    reasoning=child.reasoning,
                    is_solution=child.is_solution,
                    parent_step=step_counter - len(children) if current_depth > 0 else None
                )
                strategy_steps.append(step)

                print(f"   步骤{step.step_number}: {child.expression}")
                print(f"      数值: {child.value:.2f}, 置信度: {child.confidence:.1%}")
                if child.is_solution:
                    print(f"      🎉 找到解答!")
                print(f"      推理: {child.reasoning[:80]}...")
                print()

            # Recursively build subtrees
            for child in children:
                if child.confidence > 0.3:
                    build_recursive(child, current_depth + 1)

            node.children = children
            return node

        enhanced_tree = build_recursive(root, 0)
        return enhanced_tree, strategy_steps

    def _generate_chinese_child_thoughts(self, parent: ThoughtNode, numbers: List[int], step_offset: int) -> List[
        ThoughtNode]:
        """Generate children with Chinese reasoning"""
        prompt = f"""
        父策略: {parent.strategy.value}
        父分析: {parent.reasoning}
        数字: {numbers}
        目标: 24
        深度: {parent.depth}

        基于父策略，生成3个具体的数学表达式。
        对于每个表达式：
        1. 显示使用这些数字的数学表达式
        2. 计算精确结果
        3. 解释这如何遵循策略
        4. 基于接近24的程度和数学合理性评估置信度

        请按以下格式回答：
        表达式1: [数学表达式]
        数值1: [计算结果]
        置信度1: [0.0到1.0]
        推理1: [详细推理过程]
        ---
        表达式2: [数学表达式]
        数值2: [计算结果]
        置信度2: [0.0到1.0]
        推理2: [详细推理过程]
        ---
        表达式3: [数学表达式]
        数值3: [计算结果]
        置信度3: [0.0到1.0]
        推理3: [详细推理过程]
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.8
            )

            content = response.choices[0].message.content
            return self._parse_chinese_child_thoughts(content, parent, step_offset)

        except Exception as e:
            print(f"⚠️ 生成子思维时出错: {e}")
            return self._generate_fallback_expressions(parent, numbers, step_offset)

    def _parse_chinese_child_thoughts(self, content: str, parent: ThoughtNode, step_offset: int) -> List[ThoughtNode]:
        """Parse Chinese AI response into ThoughtNode objects"""
        children = []
        sections = content.split('---')

        for i, section in enumerate(sections):
            if not section.strip():
                continue

            try:
                # Extract numbered Chinese fields
                expression = self._extract_chinese_numbered_field(section, "表达式", i + 1)
                value_str = self._extract_chinese_numbered_field(section, "数值", i + 1)
                confidence_str = self._extract_chinese_numbered_field(section, "置信度", i + 1)
                reasoning = self._extract_chinese_numbered_field(section, "推理", i + 1)

                # Calculate value
                try:
                    value = float(value_str) if value_str else self._safe_eval(expression)
                except:
                    value = 0.0

                # Parse confidence
                try:
                    confidence = float(confidence_str) if confidence_str else 0.5
                    confidence = max(0.0, min(1.0, confidence))
                except:
                    confidence = 0.5

                child = ThoughtNode(
                    id=f"{parent.id}_child_{step_offset + i}",
                    expression=expression,
                    value=value,
                    confidence=confidence,
                    reasoning=reasoning,
                    depth=parent.depth + 1,
                    strategy=parent.strategy,
                    parent_id=parent.id
                )
                children.append(child)

            except Exception as e:
                print(f"⚠️ 解析子思维{i}时出错: {e}")
                continue

        return children

    def _parse_chinese_normal_steps(self, content: str, numbers: List[int]) -> List[ThoughtStep]:
        """Parse Chinese normal approach response into steps"""
        steps = []
        lines = content.split('\n')
        step_number = 2  # Start from 2 since step 1 is initial analysis

        current_step = None
        current_expression = ""
        current_reasoning = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('步骤'):
                if current_step:
                    steps.append(current_step)
                current_reasoning = line.split(':', 1)[1].strip() if ':' in line else line
                current_step = None

            elif line.startswith('计算:'):
                calc_part = line.split(':', 1)[1].strip()
                if '=' in calc_part:
                    current_expression = calc_part.split('=')[0].strip()
                    try:
                        value = self._safe_eval(current_expression)
                        confidence = 1.0 if abs(value - 24) < 0.001 else max(0.1, 1.0 - abs(value - 24) / 24)

                        current_step = ThoughtStep(
                            step_number=step_number,
                            strategy=StrategyType.MIXED_OPERATIONS,
                            expression=current_expression,
                            value=value,
                            confidence=confidence,
                            reasoning=current_reasoning
                        )
                        step_number += 1
                    except:
                        continue

            elif line.startswith('最终解答:'):
                solution_part = line.split(':', 1)[1].strip()
                if '=' in solution_part:
                    final_expression = solution_part.split('=')[0].strip()
                    try:
                        value = self._safe_eval(final_expression)
                        if abs(value - 24) < 0.001:
                            final_step = ThoughtStep(
                                step_number=step_number,
                                strategy=StrategyType.MIXED_OPERATIONS,
                                expression=final_expression,
                                value=value,
                                confidence=1.0,
                                reasoning="找到最终解答",
                                is_solution=True
                            )
                            steps.append(final_step)
                    except:
                        continue

        if current_step:
            steps.append(current_step)

        return steps

    def _display_chinese_strategy_comparison(self, strategy_performance: Dict[str, Dict[str, Any]]):
        """Display strategy comparison in Chinese"""
        print(f"\n📊 策略表现对比")
        print("=" * 60)

        for strategy, performance in strategy_performance.items():
            chinese_name = performance.get('chinese_name', strategy)
            print(f"\n🎯 {chinese_name}:")
            print(f"   找到解答数: {performance['solutions_found']}")
            print(f"   探索节点数: {performance['nodes_explored']}")
            print(f"   初始置信度: {performance['confidence']:.1%}")
            print(f"   效率: {performance['solutions_found'] / max(1, performance['nodes_explored']):.3f} 解答/节点")

            if performance['solutions_found'] > 0:
                print(f"   ✅ 状态: 成功")
            else:
                print(f"   ❌ 状态: 未找到解答")

    def _extract_chinese_numbered_field(self, text: str, field: str, number: int) -> str:
        """Extract numbered Chinese field from text"""
        field_name = f"{field}{number}"
        lines = text.split('\n')
        for line in lines:
            if line.strip().startswith(f"{field_name}:"):
                return line.split(':', 1)[1].strip()
        return ""

    def _extract_chinese_field(self, text: str, field: str) -> str:
        """Extract Chinese field value from text"""
        lines = text.split('\n')
        for line in lines:
            if line.strip().startswith(f"{field}:"):
                return line.split(':', 1)[1].strip()
        return ""

    def _extract_chinese_confidence(self, text: str) -> float:
        """Extract confidence score from Chinese text"""
        confidence_str = self._extract_chinese_field(text, "置信度")
        try:
            return max(0.0, min(1.0, float(confidence_str)))
        except:
            return 0.5

    def _extract_chinese_reasoning(self, text: str) -> str:
        """Extract reasoning from Chinese text"""
        reasoning = self._extract_chinese_field(text, "推理过程")
        if not reasoning:
            reasoning = self._extract_chinese_field(text, "数学分析")
        return reasoning or "未提供推理过程"

    def _generate_fallback_expressions(self, parent: ThoughtNode, numbers: List[int], step_offset: int) -> List[
        ThoughtNode]:
        """Generate fallback expressions when AI parsing fails"""
        [a, b, c, d] = numbers
        expressions = []

        if parent.strategy == StrategyType.FACTORIZATION:
            expressions = [
                (f"{a} * {b} + {c} - {d}", "尝试以乘法为基础的因式分解"),
                (f"({a} + {b}) * {c} - {d}", "先加法分组再乘法"),
                (f"{a} * ({b} + {c}) - {d}", "乘以和的形式")
            ]
        elif parent.strategy == StrategyType.MULTIPLICATION_PATH:
            expressions = [
                (f"{a} * {b} * {c} / {d}", "三重乘法配合除法"),
                (f"{a} * {b} + {c} * {d}", "两个乘法项相加"),
                (f"{a} * {b} - {c} * {d}", "乘法项相减")
            ]
        else:
            expressions = [
                (f"{a} + {b} + {c} + {d}", "简单加法"),
                (f"{a} * {b} + {c} + {d}", "混合运算"),
                (f"({a} + {b}) * ({c} - {d})", "分组运算")
            ]

        children = []
        for i, (expr, reasoning) in enumerate(expressions):
            try:
                value = self._safe_eval(expr)
                confidence = max(0.1, 1.0 - abs(value - 24) / 24)

                child = ThoughtNode(
                    id=f"{parent.id}_fallback_{step_offset + i}",
                    expression=expr,
                    value=value,
                    confidence=confidence,
                    reasoning=f"备用方案: {reasoning}",
                    depth=parent.depth + 1,
                    strategy=parent.strategy,
                    parent_id=parent.id
                )
                children.append(child)
            except:
                continue

        return children

    def _extract_solutions_from_tree(self, tree: ThoughtNode) -> List[str]:
        """Extract all valid solutions from the tree"""
        solutions = []

        def traverse(node):
            if node.is_solution:
                solutions.append(node.expression)
            for child in node.children:
                traverse(child)

        traverse(tree)
        return solutions

    def _verify_solution(self, expression: str, numbers: List[int]) -> bool:
        """Verify if expression is valid and equals 24"""
        try:
            # Check if all numbers are used
            expr_clean = expression.replace('(', '').replace(')', '').replace(' ', '')
            used_numbers = []

            i = 0
            while i < len(expr_clean):
                if expr_clean[i].isdigit():
                    num_str = ''
                    while i < len(expr_clean) and expr_clean[i].isdigit():
                        num_str += expr_clean[i]
                        i += 1
                    used_numbers.append(int(num_str))
                else:
                    i += 1

            if sorted(used_numbers) != sorted(numbers):
                return False

            result = self._safe_eval(expression)
            return abs(result - 24) < 0.001

        except:
            return False

    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate mathematical expression"""
        try:
            allowed_chars = set('0123456789+-*/().')
            if not all(c in allowed_chars or c.isspace() for c in expression):
                raise ValueError("Invalid characters in expression")

            return float(eval(expression))
        except:
            return float('inf')

    def _count_nodes(self, tree: ThoughtNode) -> int:
        """Count total nodes in tree"""
        count = 1
        for child in tree.children:
            count += self._count_nodes(child)
        return count

    def compare_approaches_detailed(self, numbers: List[int]) -> Dict[str, Any]:
        """Enhanced comparison with Chinese output"""
        print(f"\n{'=' * 80}")
        print(f"🎯 增强版24点问题求解器对比")
        print(f"📊 数字: {numbers}")
        print(f"{'=' * 80}")

        # Normal approach
        print(f"\n" + "🔍 普通方法".center(80, "="))
        normal_result = self.solve_normal_approach(numbers)

        print(f"\n📈 普通方法总结:")
        print(f"   ⏱️  耗时: {normal_result.time_taken:.2f}秒")
        print(f"   ✅ 成功: {'是' if normal_result.success else '否'}")
        print(f"   🔢 解答数: {len(normal_result.solutions)}")
        print(f"   📝 步骤数: {normal_result.steps_count}")

        # Tree-of-Thought approach
        print(f"\n" + "🌳 树状思维方法".center(80, "="))
        tot_result = self.solve_tree_of_thought(numbers)

        print(f"\n📈 树状思维总结:")
        print(f"   ⏱️  耗时: {tot_result.time_taken:.2f}秒")
        print(f"   ✅ 成功: {'是' if tot_result.success else '否'}")
        print(f"   🔢 解答数: {len(tot_result.solutions)}")
        print(f"   🔍 探索节点数: {tot_result.steps_count}")
        print(f"   🎯 使用策略数: {len(tot_result.strategy_performance)}")

        # Detailed comparison in Chinese
        print(f"\n" + "📊 详细对比".center(80, "="))

        print(f"\n🏆 性能指标:")
        print(f"   成功率:")
        print(f"      普通方法: {'✅ 成功' if normal_result.success else '❌ 失败'}")
        print(f"      树状思维: {'✅ 成功' if tot_result.success else '❌ 失败'}")

        print(f"\n   解答数量:")
        print(f"      普通方法: {len(normal_result.solutions)} 个解答")
        print(f"      树状思维: {len(tot_result.solutions)} 个解答")

        if len(tot_result.solutions) > len(normal_result.solutions):
            print(f"      🎉 树状思维多找到了 {len(tot_result.solutions) - len(normal_result.solutions)} 个解答!")

        print(f"\n   计算复杂度:")
        print(f"      普通方法: {normal_result.steps_count} 步")
        print(f"      树状思维: {tot_result.steps_count} 个节点")

        print(f"\n   时间效率:")
        print(f"      普通方法: {normal_result.time_taken:.2f}秒")
        print(f"      树状思维: {tot_result.time_taken:.2f}秒")
        if tot_result.time_taken > normal_result.time_taken:
            print(f"      树状思维耗时是普通方法的 {tot_result.time_taken / normal_result.time_taken:.1f} 倍")
            print(f"      但探索了 {len(tot_result.strategy_performance)} 种不同策略")

        # Strategy effectiveness in Chinese
        if tot_result.strategy_performance:
            print(f"\n🎯 策略效果排名:")
            strategies = sorted(
                tot_result.strategy_performance.items(),
                key=lambda x: (x[1]['solutions_found'], x[1]['confidence']),
                reverse=True
            )

            for i, (strategy, perf) in enumerate(strategies, 1):
                effectiveness = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                chinese_name = perf.get('chinese_name', strategy)
                print(f"   {effectiveness} {chinese_name}:")
                print(f"      解答数: {perf['solutions_found']}")
                print(f"      置信度: {perf['confidence']:.1%}")
                print(f"      效率: {perf['solutions_found'] / max(1, perf['nodes_explored']):.3f}")

        # Tree-of-Thought advantages in Chinese
        print(f"\n🧠 树状思维优势展示:")
        advantages = [
            "🔄 多策略并行探索",
            "🎯 基于置信度的智能剪枝",
            "🧮 数学策略专业化",
            "🔍 系统性解空间覆盖",
            "💡 更高的解答发现率",
            "🛡️ 对局部最优的鲁棒性",
            "📊 详细推理透明度",
            "🎨 策略多样性和创造性"
        ]

        for advantage in advantages:
            print(f"   {advantage}")

        return {
            "numbers": numbers,
            "normal": {
                "success": normal_result.success,
                "solutions": normal_result.solutions,
                "time": normal_result.time_taken,
                "steps": normal_result.steps_count,
                "process": [step.__dict__ for step in normal_result.thought_process]
            },
            "tree_of_thought": {
                "success": tot_result.success,
                "solutions": tot_result.solutions,
                "time": tot_result.time_taken,
                "nodes_explored": tot_result.steps_count,
                "strategies": tot_result.strategy_performance,
                "process": [step.__dict__ for step in tot_result.thought_process]
            },
            "advantages_demonstrated": {
                "more_solutions": len(tot_result.solutions) > len(normal_result.solutions),
                "strategy_diversity": len(tot_result.strategy_performance),
                "detailed_reasoning": len(tot_result.thought_process) > len(normal_result.thought_process)
            }
        }


def main():
    # Configuration
    API_KEY = "Your-API-Key"  # Replace with your actual API key

    print("🚀 树状思维链24点求解器...")
    solver = ChineseCommunicationTreeOfThought24Solver(API_KEY)

    # Test cases with Chinese descriptions
    test_cases = [
        {
            "name": "经典演示",
            "numbers": [1, 2, 3, 8],
            "difficulty": "中等",
            "note": "完美展示策略差异的经典案例"
        }
    ]

    results = []

    for i, test_case in enumerate(test_cases):
        print(f"\n\n{'🧪 测试案例 ' + str(i + 1) + ': ' + test_case['name']:<80}")
        print(f"📊 难度: {test_case['difficulty']}")
        print(f"📝 说明: {test_case['note']}")
        print(f"🔢 数字: {test_case['numbers']}")

        try:
            result = solver.compare_approaches_detailed(test_case['numbers'])
            results.append(result)

            if i < len(test_cases) - 1:
                print(f"\n⏳ 准备下一个测试案例...")
                time.sleep(3)

        except Exception as e:
            print(f"❌ 测试案例{i + 1}出错: {e}")
            continue

    # Final summary in Chinese
    print(f"\n\n{'📈 综合分析总结':<80}")
    print("=" * 80)

    if results:
        normal_successes = sum(1 for r in results if r['normal']['success'])
        tot_successes = sum(1 for r in results if r['tree_of_thought']['success'])

        print(f"\n🎯 总体成功率:")
        print(f"   普通方法: {normal_successes}/{len(results)} ({normal_successes / len(results) * 100:.1f}%)")
        print(f"   树状思维: {tot_successes}/{len(results)} ({tot_successes / len(results) * 100:.1f}%)")

        total_normal_solutions = sum(len(r['normal']['solutions']) for r in results)
        total_tot_solutions = sum(len(r['tree_of_thought']['solutions']) for r in results)

        print(f"\n🔢 总解答发现数:")
        print(f"   普通方法: {total_normal_solutions}")
        print(f"   树状思维: {total_tot_solutions}")

        if total_tot_solutions > total_normal_solutions:
            improvement = ((total_tot_solutions - total_normal_solutions) / max(1, total_normal_solutions)) * 100
            print(f"   🏆 树状思维改进: +{improvement:.1f}% 更多解答!")

    print(f"\n🎓 详细过程分析的关键洞察:")
    insights = [
        "🔍 树状思维在每一步都提供透明的推理过程",
        "🎯 多种策略增加解答发现概率",
        "🧠 置信度评分实现智能探索剪枝",
        "📊 策略专业化改进数学问题求解",
        "🔄 并行探索防止陷入次优路径",
        "💡 详细过程追踪实现学习和改进",
        "🛡️ 通过多样化推理方法提高鲁棒性",
        "🎨 通过策略多样性实现创造性解答发现"
    ]

    for insight in insights:
        print(f"   {insight}")

    print(f"\n✨ 这个增强实现展示了结构化推理和")
    print(f"   战略思维在AI问题求解中的强大力量!")


if __name__ == "__main__":
    main()
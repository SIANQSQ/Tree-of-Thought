"""
Tree of Thought (ToT) implementation for creative writing
Based on the Princeton NLP Tree-of-Thought-LLM paper
Using OpenAI 0.29.0 library with DeepSeek model
"""

import openai
import asyncio
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()

class ThoughtState:
    """Represents a state in the thought tree"""
    def __init__(self, content: str, depth: int, parent=None, score: float = 0.0):
        self.content = content
        self.depth = depth
        self.parent = parent
        self.children = []
        self.score = score
        self.is_evaluated = False
    
    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        return child

class TreeOfThought:
    """Tree of Thought implementation for creative writing"""
    
    def __init__(self, api_key: str, base_url: str = None):
        openai.api_key = api_key
        if base_url:
            openai.api_base = base_url
        self.max_depth = 3
        self.num_thoughts_per_step = 3
        self.num_evaluations = 5
    
    def generate_thoughts(self, prompt: str, current_state: str = "", depth: int = 0) -> List[str]:
        """Generate multiple thought continuations from current state"""
        system_prompt = """你是一个创意写作专家。基于给定的提示和当前内容，生成多个不同的创意续写方向。
每个方向应该：
1. 具有独特的创意角度
2. 保持逻辑连贯性
3. 展现丰富的想象力
4. 适合进一步发展

请生成3个不同的续写方向，每个方向用"---"分隔。"""
        
        user_prompt = f"""
原始提示：{prompt}
当前内容：{current_state}
深度：{depth}

请为这个创意写作任务生成3个不同的续写方向。
"""
        
        try:
            response = openai.ChatCompletion.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=800
            )
            
            content = response.choices[0].message.content
            thoughts = [thought.strip() for thought in content.split("---") if thought.strip()]
            
            # Ensure we have exactly the number of thoughts we want
            while len(thoughts) < self.num_thoughts_per_step:
                thoughts.append(f"续写方向 {len(thoughts) + 1}: 继续发展当前情节...")
            
            return thoughts[:self.num_thoughts_per_step]
            
        except Exception as e:
            print(f"生成思路时出错: {e}")
            return [f"默认续写方向 {i+1}" for i in range(self.num_thoughts_per_step)]
    
    def evaluate_thought(self, prompt: str, thought: str) -> float:
        """Evaluate a single thought's quality and potential"""
        system_prompt = """你是一个专业的创意写作评估专家。请从以下维度评估给定的创意写作片段：
1. 创意性 (0-1分)
2. 逻辑连贯性 (0-1分)  
3. 文学价值 (0-1分)
4. 发展潜力 (0-1分)
5. 情感共鸣 (0-1分)

请给出0-5分的总分，并简要说明评分理由。
格式：分数: X.X
理由: [简要说明]"""
        
        user_prompt = f"""
原始提示：{prompt}
待评估内容：{thought}

请评估这个创意写作片段的质量。
"""
        
        try:
            response = openai.ChatCompletion.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            
            # Extract score from response
            lines = content.split('\n')
            score = 3.0  # default score
            
            for line in lines:
                if '分数:' in line or 'Score:' in line or '得分:' in line:
                    try:
                        score_str = line.split(':')[1].strip()
                        score = float(score_str)
                        break
                    except:
                        continue
            
            return max(0.0, min(5.0, score))
            
        except Exception as e:
            print(f"评估思路时出错: {e}")
            return 3.0
    
    def select_best_thoughts(self, thoughts: List[ThoughtState], k: int = 2) -> List[ThoughtState]:
        """Select top k thoughts based on evaluation scores"""
        evaluated_thoughts = []
        
        for thought in thoughts:
            if not thought.is_evaluated:
                # In a real implementation, you'd evaluate each thought
                # For demo purposes, we'll assign random-ish scores based on content length and creativity indicators
                creativity_indicators = ['突然', '意外', '神秘', '奇妙', '惊讶', '梦境', '幻想', '魔法']
                base_score = 2.5
                
                # Bonus for creativity indicators
                for indicator in creativity_indicators:
                    if indicator in thought.content:
                        base_score += 0.3
                
                # Bonus for length (more developed thoughts)
                if len(thought.content) > 100:
                    base_score += 0.5
                
                thought.score = min(5.0, base_score)
                thought.is_evaluated = True
            
            evaluated_thoughts.append(thought)
        
        # Sort by score and return top k
        evaluated_thoughts.sort(key=lambda x: x.score, reverse=True)
        return evaluated_thoughts[:k]
    
    def tree_of_thought_search(self, prompt: str) -> ThoughtState:
        """Main ToT search algorithm"""
        print("🌳 开始树状思维链搜索...")
        
        # Initialize root
        root = ThoughtState("", 0)
        current_level = [root]
        
        for depth in range(self.max_depth):
            print(f"📊 处理深度 {depth + 1}/{self.max_depth}")
            next_level = []
            
            for state in current_level:
                # Generate thoughts from current state
                current_content = self.get_path_content(state)
                thoughts = self.generate_thoughts(prompt, current_content, depth)
                
                # Create thought states
                thought_states = []
                for thought in thoughts:
                    new_content = current_content + "\n" + thought if current_content else thought
                    thought_state = ThoughtState(new_content, depth + 1, state)
                    state.add_child(thought_state)
                    thought_states.append(thought_state)
                
                # Select best thoughts
                best_thoughts = self.select_best_thoughts(thought_states, k=2)
                next_level.extend(best_thoughts)
            
            current_level = next_level
            
            if not current_level:
                break
        
        # Return the best final thought
        if current_level:
            best_final = max(current_level, key=lambda x: x.score)
            print(f"✅ 最佳路径得分: {best_final.score:.2f}")
            return best_final
        else:
            return root
    
    def get_path_content(self, state: ThoughtState) -> str:
        """Get the full content path from root to current state"""
        if state.parent is None:
            return state.content
        else:
            parent_content = self.get_path_content(state.parent)
            if parent_content:
                return parent_content + "\n" + state.content
            else:
                return state.content

class ChainOfThought:
    """Traditional Chain of Thought implementation for comparison"""
    
    def __init__(self, api_key: str, base_url: str = None):
        openai.api_key = api_key
        if base_url:
            openai.api_base = base_url
    
    def generate_cot_response(self, prompt: str) -> str:
        """Generate response using traditional chain of thought"""
        system_prompt = """你是一个创意写作专家。请基于给定的提示进行创意写作。
使用传统的思维链方法：
1. 分析提示要求
2. 构思基本情节
3. 逐步展开故事
4. 完善细节描述

请生成一个完整的创意文本。"""
        
        user_prompt = f"请基于以下提示进行创意写作：{prompt}"
        
        try:
            response = openai.ChatCompletion.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"生成CoT响应时出错: {e}")
            return "抱歉，生成过程中出现错误。"

class TextEvaluator:
    """Evaluates and compares generated texts"""
    
    def __init__(self, api_key: str, base_url: str = None):
        openai.api_key = api_key
        if base_url:
            openai.api_base = base_url
    
    def evaluate_text(self, prompt: str, text: str, method: str) -> Dict[str, Any]:
        """Evaluate a single text"""
        system_prompt = """你是一个专业的创意写作评估专家。请从以下维度评估创意写作作品：

评估维度：
1. 创意性 (0-5分): 想法的新颖性和独特性
2. 连贯性 (0-5分): 逻辑结构和叙事流畅度  
3. 文学价值 (0-5分): 语言表达和艺术价值
4. 情感共鸣 (0-5分): 能否引起读者情感反应
5. 完整性 (0-5分): 故事的完整度和结构性

请为每个维度打分，并给出总分(0-25分)和详细评价。

输出格式：
创意性: X分
连贯性: X分  
文学价值: X分
情感共鸣: X分
完整性: X分
总分: X分
评价: [详细评价]"""
        
        user_prompt = f"""
原始提示: {prompt}
生成方法: {method}
待评估文本:
{text}

请对这个创意写作作品进行全面评估。
"""
        
        try:
            response = openai.ChatCompletion.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            
            # Parse the evaluation
            evaluation = self.parse_evaluation(content)
            evaluation['method'] = method
            evaluation['raw_response'] = content
            
            return evaluation
            
        except Exception as e:
            print(f"评估文本时出错: {e}")
            return {
                'creativity': 3.0,
                'coherence': 3.0,
                'literary_value': 3.0,
                'emotional_resonance': 3.0,
                'completeness': 3.0,
                'total_score': 15.0,
                'evaluation': '评估过程中出现错误',
                'method': method,
                'raw_response': ''
            }
    
    def parse_evaluation(self, content: str) -> Dict[str, Any]:
        """Parse evaluation response into structured data"""
        lines = content.split('\n')
        evaluation = {
            'creativity': 3.0,
            'coherence': 3.0,
            'literary_value': 3.0,
            'emotional_resonance': 3.0,
            'completeness': 3.0,
            'total_score': 15.0,
            'evaluation': ''
        }
        
        evaluation_text = []
        
        for line in lines:
            line = line.strip()
            if '创意性:' in line or 'creativity:' in line.lower():
                try:
                    score = float(line.split(':')[1].strip().replace('分', ''))
                    evaluation['creativity'] = score
                except:
                    pass
            elif '连贯性:' in line or 'coherence:' in line.lower():
                try:
                    score = float(line.split(':')[1].strip().replace('分', ''))
                    evaluation['coherence'] = score
                except:
                    pass
            elif '文学价值:' in line or 'literary:' in line.lower():
                try:
                    score = float(line.split(':')[1].strip().replace('分', ''))
                    evaluation['literary_value'] = score
                except:
                    pass
            elif '情感共鸣:' in line or 'emotional:' in line.lower():
                try:
                    score = float(line.split(':')[1].strip().replace('分', ''))
                    evaluation['emotional_resonance'] = score
                except:
                    pass
            elif '完整性:' in line or 'completeness:' in line.lower():
                try:
                    score = float(line.split(':')[1].strip().replace('分', ''))
                    evaluation['completeness'] = score
                except:
                    pass
            elif '总分:' in line or 'total:' in line.lower():
                try:
                    score = float(line.split(':')[1].strip().replace('分', ''))
                    evaluation['total_score'] = score
                except:
                    pass
            elif '评价:' in line or 'evaluation:' in line.lower():
                evaluation_text.append(line.split(':', 1)[1].strip())
            elif line and not any(keyword in line for keyword in ['创意性', '连贯性', '文学价值', '情感共鸣', '完整性', '总分']):
                evaluation_text.append(line)
        
        evaluation['evaluation'] = '\n'.join(evaluation_text).strip()
        
        return evaluation
    
    def compare_methods(self, prompt: str, cot_text: str, tot_text: str) -> Dict[str, Any]:
        """Compare CoT and ToT results"""
        print("📊 评估传统思维链结果...")
        cot_eval = self.evaluate_text(prompt, cot_text, "传统思维链 (CoT)")
        
        print("📊 评估树状思维链结果...")
        tot_eval = self.evaluate_text(prompt, tot_text, "树状思维链 (ToT)")
        
        # Calculate improvements
        improvements = {}
        for key in ['creativity', 'coherence', 'literary_value', 'emotional_resonance', 'completeness', 'total_score']:
            improvements[key] = tot_eval[key] - cot_eval[key]
        
        return {
            'cot_evaluation': cot_eval,
            'tot_evaluation': tot_eval,
            'improvements': improvements,
            'tot_better': tot_eval['total_score'] > cot_eval['total_score']
        }
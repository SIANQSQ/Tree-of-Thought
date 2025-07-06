import os
import json
from text_create_engine import TreeOfThought, ChainOfThought, TextEvaluator

def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)

def print_evaluation_results(evaluation: dict):
    """打印评估结果"""
    print(f"📊 评估结果 ({evaluation['method']}):")
    print(f"   创意性: {evaluation['creativity']:.1f}/5.0")
    print(f"   连贯性: {evaluation['coherence']:.1f}/5.0")
    print(f"   文学价值: {evaluation['literary_value']:.1f}/5.0")
    print(f"   情感共鸣: {evaluation['emotional_resonance']:.1f}/5.0")
    print(f"   完整性: {evaluation['completeness']:.1f}/5.0")
    print(f"   总分: {evaluation['total_score']:.1f}/25.0")
    print(f"   评价: {evaluation['evaluation']}")

def main():
    """主函数"""
    print_separator("树状思维链 vs 传统思维链创意写作对比实验")
    print("使用OpenAI 0.29.0库调用DeepSeek大模型")
    
    # 检查环境变量
    api_key = "Your-API-Key"
    base_url = "https://api.deepseek.com/v1"
    
    if not api_key:
        print("❌ 错误: 请设置你的 DEEPSEEK_API_KEY")
        return
    
    print(f"🔑 使用API: {base_url}")
    
    # 创意写作提示词
    prompts = [
        "写一个关于时间旅行者在古代遇到现代科技的奇幻故事",
        "描述一个能够听懂动物语言的小女孩的冒险经历",
        "创作一个关于最后一本实体书的科幻故事",
        "写一个关于会做梦的机器人的温馨故事",
        "以雨夜、圣器、流浪猫为关键词写一篇文章"
    ]
    
    # 初始化组件
    print("\n🔧 初始化系统组件...")
    tot = TreeOfThought(api_key, base_url)
    cot = ChainOfThought(api_key, base_url)
    evaluator = TextEvaluator(api_key, base_url)
    print("✅ 系统初始化完成")
    
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        print_separator(f"实验 {i}: {prompt}")
        
        try:
            # 生成传统思维链结果
            print("🔗 生成传统思维链结果...")
            cot_result = cot.generate_cot_response(prompt)
            print(f"✅ CoT结果生成完成 (长度: {len(cot_result)} 字符)")
            
            # 生成树状思维链结果
            print("\n🌳 生成树状思维链结果...")
            tot_state = tot.tree_of_thought_search(prompt)
            tot_result = tot_state.content
            print(f"✅ ToT结果生成完成 (长度: {len(tot_result)} 字符)")
            
            # 显示生成的内容
            print_separator("生成内容对比")
            print("📝 传统思维链 (CoT) 结果:")
            print("-" * 40)
            print(cot_result)
            
            print("\n📝 树状思维链 (ToT) 结果:")
            print("-" * 40)
            print(tot_result)
            
            # 评估和对比
            print_separator("评估与对比")
            print("🤖 调用AI评估员进行多维度评分...")
            comparison = evaluator.compare_methods(prompt, cot_result, tot_result)
            
            # 显示评估结果
            print_evaluation_results(comparison['cot_evaluation'])
            print()
            print_evaluation_results(comparison['tot_evaluation'])
            
            # 显示改进情况
            print(f"\n📈 改进情况:")
            improvements = comparison['improvements']
            for key, value in improvements.items():
                if key == 'total_score':
                    continue
                key_name = {
                    'creativity': '创意性',
                    'coherence': '连贯性', 
                    'literary_value': '文学价值',
                    'emotional_resonance': '情感共鸣',
                    'completeness': '完整性'
                }.get(key, key)
                
                if value > 0:
                    print(f"   {key_name}: +{value:.1f} ⬆️")
                elif value < 0:
                    print(f"   {key_name}: {value:.1f} ⬇️")
                else:
                    print(f"   {key_name}: {value:.1f} ➡️")
            
            total_improvement = improvements['total_score']
            if total_improvement > 0:
                print(f"   总分提升: +{total_improvement:.1f} 分 🎉")
                improvement_percentage = (total_improvement / comparison['cot_evaluation']['total_score']) * 100
                print(f"   提升幅度: {improvement_percentage:.1f}%")
            else:
                print(f"   总分变化: {total_improvement:.1f} 分")
            
            # 保存结果
            result = {
                'experiment_id': i,
                'prompt': prompt,
                'cot_result': cot_result,
                'tot_result': tot_result,
                'comparison': comparison,
                'timestamp': str(os.popen('date').read().strip())
            }
            results.append(result)
            
            print(f"\n{'🏆 树状思维链获胜!' if comparison['tot_better'] else '🤔 传统思维链表现更好'}")
            
        except Exception as e:
            print(f"❌ 实验 {i} 执行失败: {e}")
            continue
    
    # 总结报告
    print_separator("实验总结报告")
    
    if results:
        tot_wins = sum(1 for r in results if r['comparison']['tot_better'])
        total_experiments = len(results)
        
        print(f"📊 实验统计:")
        print(f"   总实验数: {total_experiments}")
        print(f"   树状思维链获胜: {tot_wins} 次")
        print(f"   传统思维链获胜: {total_experiments - tot_wins} 次")
        print(f"   树状思维链胜率: {tot_wins/total_experiments*100:.1f}%")
        
        # 计算平均改进
        avg_improvements = {}
        for key in ['creativity', 'coherence', 'literary_value', 'emotional_resonance', 'completeness', 'total_score']:
            avg_improvement = sum(r['comparison']['improvements'][key] for r in results) / len(results)
            avg_improvements[key] = avg_improvement
        
        print(f"\n📈 平均改进情况:")
        for key, value in avg_improvements.items():
            if key == 'total_score':
                continue
            key_name = {
                'creativity': '创意性',
                'coherence': '连贯性',
                'literary_value': '文学价值', 
                'emotional_resonance': '情感共鸣',
                'completeness': '完整性'
            }.get(key, key)
            print(f"   {key_name}: {value:+.2f}")
        
        print(f"   总分平均提升: {avg_improvements['total_score']:+.2f}")
        
        # 计算平均提升百分比
        avg_cot_score = sum(r['comparison']['cot_evaluation']['total_score'] for r in results) / len(results)
        if avg_cot_score > 0:
            avg_improvement_percentage = (avg_improvements['total_score'] / avg_cot_score) * 100
            print(f"   平均提升幅度: {avg_improvement_percentage:+.1f}%")
        
        # 保存详细结果到文件
        output_data = {
            'experiment_summary': {
                'total_experiments': total_experiments,
                'tot_wins': tot_wins,
                'cot_wins': total_experiments - tot_wins,
                'tot_win_rate': tot_wins/total_experiments,
                'average_improvements': avg_improvements,
                'openai_version': '0.29.0',
                'model': 'deepseek-chat'
            },
            'detailed_results': results
        }
        
        with open('experiment_results.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 详细结果已保存到 experiment_results.json")
        
        # 结论
        print_separator("结论")
        if avg_improvements['total_score'] > 0:
            print("🎉 实验结果表明：树状思维链在创意写作任务中表现优于传统思维链！")
            print(f"   平均总分提升: {avg_improvements['total_score']:.2f} 分")
            if avg_cot_score > 0:
                print(f"   平均提升幅度: {avg_improvement_percentage:+.1f}%")
            print("   主要优势体现在多路径探索和最优选择机制上。")
            print("\n🔍 优势分析:")
            
            best_improvements = sorted(avg_improvements.items(), key=lambda x: x[1], reverse=True)
            for key, value in best_improvements[:3]:
                if key != 'total_score' and value > 0:
                    key_name = {
                        'creativity': '创意性',
                        'coherence': '连贯性',
                        'literary_value': '文学价值', 
                        'emotional_resonance': '情感共鸣',
                        'completeness': '完整性'
                    }.get(key, key)
                    print(f"   - {key_name}提升最显著: +{value:.2f}分")
        else:
            print("🤔 实验结果显示两种方法表现相近，可能需要更多样本或调整参数。")
            print("   建议增加实验样本数量或调整树搜索深度参数。")
        
        print(f"\n📋 技术细节:")
        print(f"   - OpenAI库版本: 0.29.0")
        print(f"   - 使用模型: deepseek-chat")
        print(f"   - 树搜索深度: {tot.max_depth}")
        print(f"   - 每步候选数: {tot.num_thoughts_per_step}")
        
    else:
        print("❌ 没有成功完成的实验，请检查API配置和网络连接。")
        print("🔧 故障排除建议:")
        print("   1. 检查 .env 文件中的 DEEPSEEK_API_KEY 是否正确")
        print("   2. 确认网络连接正常")
        print("   3. 验证DeepSeek API额度是否充足")

if __name__ == "__main__":
    main()
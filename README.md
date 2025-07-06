# Tree-of-Thought


# 树状思维链创意写作对比实验

基于Princeton NLP的Tree-of-Thought-LLM论文实现，使用OpenAI 0.29.0库调用DeepSeek大模型，对比树状思维链(ToT)与传统思维链(CoT)在创意写作任务中的表现。

## 功能特点

- 🌳 **树状思维链实现**: 基于论文的多路径搜索算法
- 🔗 **传统思维链对比**: 线性思维过程实现
- 📊 **智能评估系统**: AI驱动的多维度评分(0-5分制)
- 📈 **详细对比分析**: 量化展示两种方法的优劣
- 💾 **结果保存**: 自动保存实验数据为JSON格式
- 🔧 **OpenAI 0.29.0兼容**: 使用经典版本OpenAI库

## 技术架构

```
使用技术栈:
├── OpenAI 0.29.0        # 经典版本OpenAI库
├── DeepSeek Chat        # 大语言模型
├── Python 3.7+         # 编程语言
└── Tree-of-Thought      # 核心算法
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置设置

1. 复制环境变量模板：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入您的DeepSeek API密钥：
```
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

## 运行实验

```bash
cd src
python main.py
```

## 实验流程

1. **初始化**: 设置ToT和CoT生成器，以及评估器
2. **内容生成**: 
   - CoT: 使用线性思维链生成创意文本
   - ToT: 使用树状搜索生成多个候选，选择最优路径
3. **智能评估**: AI评估员从5个维度打分：
   - 创意性 (0-5分)
   - 连贯性 (0-5分)  
   - 文学价值 (0-5分)
   - 情感共鸣 (0-5分)
   - 完整性 (0-5分)
4. **对比分析**: 计算改进幅度和胜率统计
5. **结果保存**: 生成详细的JSON报告

## 核心算法

### 树状思维链 (ToT)
- **多路径生成**: 每个节点生成3个不同的续写方向
- **智能评估**: 对每个思路进行质量评分
- **最优选择**: 选择得分最高的路径继续探索
- **深度控制**: 限制搜索深度避免过度复杂

### 传统思维链 (CoT)
- **线性生成**: 按照固定步骤顺序生成内容
- **单一路径**: 不进行多候选比较
- **直接输出**: 一次性生成完整结果

## OpenAI 0.29.0 特性

本实现专门适配OpenAI 0.29.0版本：

```python
# 使用经典的ChatCompletion接口
response = openai.ChatCompletion.create(
    model="deepseek-chat",
    messages=[...],
    temperature=0.8,
    max_tokens=800
)

# 兼容旧版本的API调用方式
openai.api_key = api_key
openai.api_base = base_url
```

## 评估维度

| 维度 | 描述 | 权重 |
|------|------|------|
| 创意性 | 想法的新颖性和独特性 | 20% |
| 连贯性 | 逻辑结构和叙事流畅度 | 20% |
| 文学价值 | 语言表达和艺术价值 | 20% |
| 情感共鸣 | 能否引起读者情感反应 | 20% |
| 完整性 | 故事的完整度和结构性 | 20% |

## 输出结果

实验完成后将生成：
- 控制台实时输出对比结果
- `experiment_results.json`: 详细的实验数据
- 统计报告包括胜率、平均改进幅度等

## 预期结果

根据Tree-of-Thought论文的理论基础，预期ToT在创意写作任务中将表现出：
- 更高的创意性得分
- 更好的整体质量
- 更强的情感共鸣能力

## 技术架构

```
src/
├── tree_of_thought.py    # 核心算法实现
├── main.py              # 主实验程序
├── requirements.txt     # 依赖包列表
└── .env.example        # 环境变量模板
```

## 版本兼容性

- **OpenAI库**: 0.29.0 (经典版本)
- **Python**: 3.7+
- **DeepSeek API**: v1

## 注意事项

- 确保DeepSeek API密钥有效且有足够额度
- 实验过程中会进行多次API调用，请注意费用控制
- 网络连接需要稳定，避免中途中断
- 建议在测试环境中先运行小规模实验

## 扩展功能

可以通过修改参数来调整实验：
- `max_depth`: 树搜索最大深度
- `num_thoughts_per_step`: 每步生成的候选数量
- `temperature`: 生成文本的创意度
- 添加更多评估维度
- 支持更多创意写作类型

## 故障排除

如果遇到问题：

1. **API错误**: 检查`.env`文件中的API密钥
2. **网络问题**: 确认网络连接和防火墙设置
3. **依赖问题**: 确保安装了正确版本的依赖包
4. **模型调用失败**: 验证DeepSeek API额度

## 许可证

本项目基于MIT许可证开源。

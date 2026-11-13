测试：python3 -m unittest discover -s tests/deliberation_cycles -p 'test_*.py'
构建：python3 -m compileall -q deliberation_lake tests/deliberation_cycles
验收：python3 -m deliberation_lake.console.run fixtures/demo.json --database /tmp/deliberation_lake-acceptance.sqlite3
议题发现：python3 -m deliberation_lake.issue_discovery --input fixtures/analysis/input.json --database /tmp/deliberation_lake-acceptance.sqlite3 --output /tmp/issues.json
所有输入均为本地合成数据，运行不需要网络。

议题发现（deliberation_lake.issue_discovery）口径：
- 纯标准库实现：字符 unigram/bigram TF-IDF + 余弦相似度，不下载任何模型或词表。
- 文本与结构联合：综合相似度 = 0.85*文本余弦 + 0.15*同类别加分；结构分只能
  加分不能授权合并，合并须同时越过综合门槛 0.35 与纯文本门槛 0.12。
- 传播与独立支持度分离：forwarded_from 转发树只计一个独立声音，转发件保留在
  议题内体现传播规模；同文案重复提交单独登记为重复传播证据。
- 人工关系按 available_at <= as_of 生效；same 强制合并、different 强制禁止，
  未来复核标签只登记为 pending，不参与本次聚合。
- 议题数：max_topics 内对层次聚类切分做轮廓系数扫描选 k，不为均衡而合并无关诉求。
- 低于 minimum_independent_support 的簇保留为候选议题并在 unassigned 留痕，
  少数群体的独立议题不会被静默丢弃。
- 稳定性：固定种子的传播根 bootstrap、留一法、等权朴素对照（ARI）。
- 谱系：discovery_id 由输入与算法版本指纹决定，重放幂等；旧议题到新议题输出
  anchored/split 关系，承诺随关系锚定；无法解析的旧引用作为未解决证据返回。


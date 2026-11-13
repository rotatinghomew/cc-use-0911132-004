# 业务演练数据口径

submissions 为原意见，不含真实个人信息；forwarded_from 是传播来源，不是删除原文的指令。judgments 只在 available_at 后可用于约束。previous_issues 为既有承诺引用，manual_actions 为后续人工调整请求，不是自动重算的真值。不存在的旧议题引用须作为未解决证据返回。

各 ID 均为本地合成标识，不包含真实人员或生产数据。时刻使用带时区 ISO 8601；as_of 是分析可知信息的截止时刻，不是业务发生时间上界。晚到事实允许保留在输入中，但只能在 available_at 到达之后采用。source_versions 标识输入版本；baseline_case_id 关联同库既有演示运行。空值表示未知，不应未经说明转成零。

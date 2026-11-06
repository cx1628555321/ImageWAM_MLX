# 整理后验证

远端native-w4真实CLI：三图输入、10步、10次预热、输出16×14。action与normalized均同整理前`records/prefix_followup_20261106/cli_native-w4`逐位一致，最大绝对差0。单请求150.539ms仅作功能验证，不更新正式基准。

实际动作配置：W4 affine/group64、65个合格线性层、浮点激活。正式入口不再导入experiments候选。远端完整数据在`records/layout_native_w4_20261106/`及本目录validation.json。

远端正式与归档Markdown的本地Markdown链接检查通过。全量历史实验未重跑，结果及失败证据保留。当前已删除的仅为生成缓存。

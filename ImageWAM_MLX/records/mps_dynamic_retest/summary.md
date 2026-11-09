# M5 Pro RoboTwin 三图 BF16 基线

端到端：三张已解码 RGB 图、状态、指令到 CPU 反归一化动作；不含读视频、加载和写文件。
首次推理发生在任何模型前向之前；模型仅加载一次。稳态无模块钩子，无文本/图像特征缓存。
模块为独立测量阶段的 MPS 同步 wall 时间，父子项不可相加；非纯 GPU kernel 时间。

首次推理：5658.243 ms
稳态：{'count': 20, 'mean': 353.59529999996084, 'median': 350.4840420000619, 'p95': 370.7759708005142, 'std': 8.482555999293835, 'min': 346.253916999558, 'max': 383.2869159996335}
稳定性：{'cv': 0.023989447821548455, 'half_median_relative_drift': 6.205703393249281e-05}

|加载阶段|ms|
|---|---:|
|optimized.policy.map_checkpoint|11.025|
|optimized.visual.construct_and_load|3527.780|
|optimized.action.construct_and_load|1220.305|
|optimized.ae.construct_and_load|92.785|
|optimized.qwen.construct_and_load|7236.616|
|optimized.assemble_and_proprio|89.230|
|model_load_total|12464.398|

|模块|调用数/推理|平均 ms/推理|
|---|---:|---:|
|observation.preprocess|1|2.999|
|text.chat_template|1|0.060|
|text.tokenize|1|0.130|
|text.qwen_forward|1|100.732|
|text.encode|1|102.255|
|proprio.linear|1|0.144|
|proprio.encode_and_pack|1|0.674|
|image.ae_encode|1|21.818|
|image.encode_tokens|1|22.171|
|visual.prepare|1|2.171|
|visual.prefill|1|138.512|
|action.prepare|10|6.248|
|action.transformer|10|148.693|
|action.project|10|5.390|
|action.scheduler|10|4.019|
|action.denoise_step|10|167.894|
|action.denoise_loop|1|167.951|
|policy.model_forward|1|436.769|
|action.denormalize|1|0.016|
|policy.action_chunk|1|439.822|

Saved MPS BF16 baseline only; no CUDA comparison or simulator success evaluation performed.

原始输入 inputs.npz + input_manifest.json；reference_action.npy/CSV 为首次输出；每轮归一化及反归一化输出均独立保存。

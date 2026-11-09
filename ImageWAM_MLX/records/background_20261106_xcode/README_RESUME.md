# Xcode 就绪后恢复适配测试

2026-11-06，Xcode 27.2 (27B52)，Metal Toolchain 27B47，Metal 4 最小编译及链接通过。

后台入口：experiments/background_20261105/resume_xcode.py
构建入口：experiments/background_20261105/setup_xcode.py

独立目录：records/background_20261106_xcode。
不修改生产 Python 环境或默认推理配置。
显式设置 DEVELOPER_DIR 和 MACOSX_DEPLOYMENT_TARGET=26.2、CMAKE_OSX_DEPLOYMENT_TARGET=26.2。
新建构建目录避免继承缺少 Metal 编译器时的 CMake 缓存。

顺序：当前 stock baseline；route4796 构建和 baseline/native_w1/native_w4/qq_mxfp4/qq_mxfp8/qq_nvfp4；affine4802 构建和 baseline/affine_w4/affine_w8/prefix_affine_w4/prefix_affine_w8；符合既有筛选阈值的候选进行 ABBA 确认。
维持 speed、all_int8 prefix、文本缓存1、10步推理；worker 沿用原有模块统计和输出误差验证。
源码构建使用 JIT，不能将相对发布版的全部收益归因于某个 PR；必须同时查看每个构建环境的 baseline。

启动时检测到 PI05 服务 PID 56454，队列等待独占条件，不停止其他服务。
status.json 记录 active/competing/jobs，REPORT.md 自动汇总，runner.log 和各 attempt 日志用于排错。
服务退出并连续通过空闲检查后自动运行，后台等待不等于已完成测试。

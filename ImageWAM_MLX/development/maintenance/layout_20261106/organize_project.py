"""One-time reversible layout maintenance. Run from the ImageWAM_MLX root."""
from pathlib import Path
import json,re,shutil,tarfile,os
root=Path.cwd();assert (root/'imagewam_mlx/runtime.py').is_file()
log=root/'development/maintenance/layout_20261106';log.mkdir(parents=True,exist_ok=True)
assert not (log/'completed.json').exists(),'Already organized; inspect manifest instead of rerunning'
keep={'README.md','README_VERSION_SELECTION.md','README_INFERENCE_VERSIONS.md','README_RETAINED_VERSIONS_20261106.md','README_ROBOTWIN_ACCURACY_20261106.md'}
moves={}
for p in root.glob('README*.md'):
 if p.name not in keep:moves[p.relative_to(root).as_posix()]='docs/optimization/'+p.name
for p in root.glob('*.json'):moves[p.name]='development/results/'+p.name
for p in root.glob('experiments_*'):
 if p.is_dir() and not p.is_symlink():moves[p.name]='development/legacy_experiments/'+p.name
texts=[]
for base in [root,root/'scripts',root/'imagewam_mlx',root/'experiments',root/'tests']:
 if base==root: texts.extend(root.glob('*.md'))
 elif base.exists():texts.extend(x for x in base.rglob('*') if x.is_file() and x.suffix in ('.py','.sh','.md') and '__pycache__' not in x.parts)
for old in moves:
 p=root/old
 if p.is_dir():texts.extend(x for x in p.rglob('*') if x.is_file() and x.suffix in ('.py','.sh','.md'))
texts=list(dict.fromkeys(texts));original={p.relative_to(root).as_posix():p.read_text() for p in texts}
with tarfile.open(log/'source_docs_before.tar.gz','w:gz') as tar:
 for p in texts:tar.add(p,arcname=p.relative_to(root))
(log/'moves.json').write_text(json.dumps(moves,indent=2))
def mapped(rel):
 for old,new in moves.items():
  if rel==old or rel.startswith(old+'/'):return new+rel[len(old):]
 return rel
for old,new in moves.items():
 dest=root/new;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists(),dest;shutil.move(str(root/old),str(dest))
# Rebase every Markdown file link according to the old file's location.
changed=[]
for old,text in original.items():
 dest=root/mapped(old);newtext=text
 if dest.suffix=='.md':
  def link(m):
   target=m.group(2)
   if target.startswith(('http:','https:','mailto:','#','/')) or '://' in target:return m.group(0)
   path,sep,frag=target.partition('#');absold=Path(os.path.normpath(str(root/Path(old).parent/path)))
   try:rel=absold.relative_to(root).as_posix()
   except ValueError:return m.group(0)
   targetpath=root/mapped(rel)
   return m.group(1)+os.path.relpath(targetpath,dest.parent)+(sep+frag if sep else '')+m.group(3)
  newtext=re.sub(r'(!?\[[^\]]*\]\()([^\s)]+)(\))',link,text)
 else:
  for a,b in moves.items():
   if a.startswith('experiments_'):newtext=newtext.replace(a,b)
   elif a.endswith('.json'):
    # Only root-relative string literals, not identically-named nested artifacts.
    newtext=re.sub(r'([\'\"])'+re.escape(a)+r'\1',lambda m:m.group(1)+b+m.group(1),newtext)
 if newtext!=text:dest.write_text(newtext);changed.append(mapped(old))
# Keep historical main text with correctly rebased local links.
oldmain=(root/'README.md').read_text()
archive=root/'docs/optimization/README_LEGACY_PROJECT_OVERVIEW.md'
archive.parent.mkdir(parents=True,exist_ok=True)
oldmain=re.sub(r'(!?\[[^\]]*\]\()([^\s)]+)(\))',lambda m:m.group(0) if m.group(2).startswith(('http:','https:','mailto:','#','/')) else m.group(1)+'../../'+m.group(2)+m.group(3),oldmain)
archive.write_text(oldmain)
(root/'README.md').write_text('''# ImageWAM MLX

M5Pro上的RoboTwin三图推理。项目目录：`/Users/qklj/work/cx/ImageWAM_MLX`。

## 使用入口

- **[版本选型总表](README_VERSION_SELECTION.md)**：一张表查看性能、多样本精度、主要优化与启动命令。优先native-w4；更低平均误差选择action-bf16。
- [版本启动与复测细节](README_INFERENCE_VERSIONS.md)
- [保留版本列表](README_RETAINED_VERSIONS_20261106.md)
- [RoboTwin六版本精度报告](README_ROBOTWIN_ACCURACY_20261106.md)
- [项目目录说明](README_PROJECT_LAYOUT.md)

## 开发与历史记录

- [优化过程文档索引](docs/optimization/README.md)
- [过程代码与配置说明](development/README.md)

正式代码在`imagewam_mlx/`，使用入口在`scripts/infer.py`和`scripts/infer_version.py`。原始权重、数据、环境与测试证据均保留。详细实验结论不在此重复维护。
''')
(root/'README_PROJECT_LAYOUT.md').write_text('''# 项目目录说明

```text
ImageWAM_MLX/
  README.md                         项目导航
  README_VERSION_SELECTION.md       面向使用者的选型与命令
  README_INFERENCE_VERSIONS.md       启动与复测细节
  README_RETAINED_VERSIONS_20261106.md 版本清单
  README_ROBOTWIN_ACCURACY_20261106.md 正式精度总结
  README_PROJECT_LAYOUT.md          本目录说明
  imagewam_mlx/                     正式实现
  scripts/                         正式入口与历史验证工具（见目录README）
  tests/                           检查代码
  docs/optimization/               分阶段优化分析、测试过程与旧总结
  experiments/                     已按实验批次分组的过程代码/配置
  development/legacy_experiments/   原根目录experiments_*历史脚本
  development/results/             原根目录零散JSON结果
  development/maintenance/          整理清单和修改前代码/文档备份
  records/                         实测证据、输出及已验证源码构建环境
  assets/ examples/ weights/        推理资产、示例、权重（远端）
  .venv/                           已验证旧运行时（远端）
```

`records/background_20261106_xcode/builds/.../venv`是实际使用的新MLX环境，不是可删除的临时结果。移动虚拟环境会破坏绝对路径，因此records保持原位。experiments仍为实验复现入口；不再被正式native-w1/native-w4启动路径导入。

仅删除源码目录中的__pycache__/pyc/pyo及.DS_Store。没有删除权重、原始数据、实验输出、失败证据或虚拟环境。根目录JSON与旧实验目录均归档，非删除。历史脚本的旧实验目录字面路径和根目录JSON引用已更新；历史实验没有全部重跑，不将目录整理称为所有旧实验再次验证。

本地是开发镜像，assets/weights/环境和大体积结果可能仅在M5Pro存在。整理清单及备份在`development/maintenance/layout_20261106/`，可按moves.json反向恢复归档路径。正式启动验证见该目录的validation记录。
''')
(root/'development/README.md').write_text('''# 开发过程材料

- `legacy_experiments/`：原根目录experiments_*目录，保留旧实验代码和中间配置。
- `results/`：原根目录零散JSON结果，保留证据不覆盖。
- `maintenance/`：目录整理清单、修改前代码/文档备份和验证记录。
- 根目录`experiments/`：已按批次组织的实验实现，继续作为复现实验入口。
- 根目录`records/`：原始测试证据及实际运行时环境，不能整体当作缓存清理。

运行历史脚本时仍先cd项目根目录。新正式总结放根目录；新过程报告放docs/optimization，过程脚本/配置按批次放experiments，输出放records。
''')
(root/'scripts/README.md').write_text('''# 脚本入口

正式使用：`infer.py`（accuracy等原接口）、`infer_version.py`（保留版本）、`benchmark.py`（性能测试）。

资产准备：`convert_weights.py`、`prepare_assets.py`、`compact_action_weights.py`。核验工具包括`analyze_accuracy.py`、`compare_traces.py`、`capture_environment.py`及validate/test/audit脚本。

其余experiment/diagnose/sweep/run_*工具是历史验证入口，保留原位置以避免改变__file__路径及脚本间导入。阶段性实验实现集中在根目录experiments及development/legacy_experiments。正式推理不从实验目录导入候选。

使用命令与选择建议见[版本总表](../README_VERSION_SELECTION.md)。
''')
index=root/'docs/optimization/README.md';index.write_text('# 优化过程文档索引\n\n正式使用请看[选型总表](../../README_VERSION_SELECTION.md)。本目录保留历史分析、失败方案与各阶段结果；历史数字不能替代最新精度结论。文中命令默认在项目根目录运行。\n\n'+'\n'.join(f'- [{p.stem}]({p.name})' for p in sorted(index.parent.glob('*.md')) if p!=index)+'\n')
removed=[]
for base in [root/'scripts',root/'imagewam_mlx',root/'tests',root/'experiments',root/'development/legacy_experiments']:
 if not base.exists():continue
 for p in list(base.rglob('__pycache__')):
  if p.is_dir():removed.append(p.relative_to(root).as_posix());shutil.rmtree(p)
 for p in list(base.rglob('*.pyc'))+list(base.rglob('*.pyo'))+list(base.rglob('.DS_Store')):
  if p.is_file():removed.append(p.relative_to(root).as_posix());p.unlink()
report=dict(moved=moves,updated=changed,deleted_regenerable=removed,root_readmes=sorted(p.name for p in root.glob('README*.md')))
(log/'completed.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(moved=len(moves),updated=len(changed),deleted_cache_paths=len(removed),root_readmes=report['root_readmes']),indent=2))

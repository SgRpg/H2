# SSH + Git 云端工作方式

这是独立实验仓库，来自已校验的本地实验包。原科研目录未修改。
作者代码以普通文件快照保存在 vendor/evotool，固定原始 commit，
不是子模块，也不要在其目录里直接 git pull。来源与哈希见 baseline-manifest.json。
本仓库的提交表示“作者快照 + 本项目适配层”的版本。

## 首次连接和同步

1. 本地 VSCode 通过 Remote-SSH 连接 DSW；主机信息采用控制台实际配置。
2. 在云端 /mnt/workspace 克隆本私有仓库。私有仓库认证由你本人完成，
   不把令牌写进 URL、脚本或 Git 文件。服务器只拉取时优先使用只读部署密钥。
3. 切换到本地确定的提交，不盲目更新作者仓库。
4. 运行 `bash scripts/cloud_preflight.sh`，先检查 Python、GPU 和原文件哈希。
5. 安装依赖和模型的步骤见 README.md；正式 GPU 实验仍需单独审批。

## 后续迭代

建议在本地修改、提交、推送，在云端拉取；云端发现问题后先回传日志。
拉取前检查 `git status --short`，有未提交改动时停止同步并先保存，
不要使用 reset --hard 或 clean 删除文件。
完成一次运行时记录 `git rev-parse HEAD` 和 `git status --porcelain`。

模型放 models/，结果放 outputs/，环境放仓库外，均不纳入 Git。
不要提交访问令牌、私钥、.env 或账号配置。仓库初始化时已在本地 exclude 中补充防护；
克隆后用下面命令为新克隆设置相同的本地忽略文件：

```bash
git config --local core.excludesFile .gitignore-cloud
```

`.gitattributes` 保留原始文件字节，避免跨平台换行转换破坏哈希校验。
原实验包内 PACKAGE-FILES.json 只覆盖原包文件；新增同步文件不在其中。
本地首次提交使用明确标注的自动化作者身份，不代表人工审批或实验批准。

# Claude Code 常用启动命令速查

> 适用版本：Claude Code 2.1.317（以 `claude --help` 输出为准）。
> 所有命令均可在任意目录下执行；带 `<...>` 的为必填参数，`[...]` 为可选参数。

## 1. 最常用的几条

| 命令 | 作用 |
| --- | --- |
| `claude` | 在当前目录启动交互式会话 |
| `claude "解释这个项目"` | 启动会话并直接发送第一条消息 |
| `claude -c` | 继续**当前目录**最近一次会话 |
| `claude -r` | 打开历史会话选择器 |
| `claude -r <会话ID或名称>` | 恢复指定会话 |
| `claude -p "总结改动"` | 非交互模式：输出结果后退出 |
| `claude update` | 检查并安装更新 |
| `claude doctor` | 检查安装与配置健康状况 |
| `claude --version` | 显示版本号 |

## 2. 启动交互式会话

```bash
claude                                  # 交互式 REPL
claude "帮我重构 single.py"              # 带初始提示
claude --add-dir ../Riemann             # 额外允许访问的目录（可多个）
claude --ide                             # 启动时自动连接唯一的 IDE
claude --desktop                         # 在 Claude Desktop 中打开
claude --chrome                          # 启用 Chrome 集成
claude --model opus                      # 指定模型
claude --agent reviewer                  # 使用指定 agent 作为当前会话
claude --name "euler-refactor"           # 给会话起名，便于 /resume
```

### 会话恢复与分叉

```bash
claude -c                        # 继续最近一次会话
claude -r                        # 交互式选择历史会话
claude -r 3f2a9c                 # 按 ID（完整或前缀）恢复
claude -r euler-refactor         # 按会话名恢复
claude -r <id> --fork-session    # 恢复但创建新的会话 ID（保留原会话）
claude --from-pr 123             # 恢复与某个 PR 关联的会话
claude --session-id <uuid>       # 指定会话 UUID
```

`--no-session-persistence`（仅配合 `-p`）可让会话不落盘、之后无法恢复。

### 后台会话与 worktree

```bash
claude --bg "跑一遍全部算例"        # 后台启动，立即返回会话 ID
claude attach <id|name>            # 在当前终端接入后台会话
claude logs <id|name>              # 查看后台会话输出
claude stop <id|name>              # 停止后台会话（会话内容保留）
claude rm <id>                     # 删除后台会话
claude respawn <id>                # 用当前版本重启后台会话
claude agents                      # 列出后台会话

claude -w                          # 为本次会话新建 git worktree
claude -w euler-fix --tmux         # 指定名称并放入 tmux
```

### 云端 / 远程

```bash
claude --cloud "描述任务"           # 创建云端会话
claude --cloud <session_id|url>    # 接入已有云端会话
claude --teleport [session]        # 恢复 teleport 会话
claude --remote-control [名称]     # 开启 Remote Control
```

## 3. 非交互模式（脚本 / 管道 / CI）

`-p` / `--print` 表示"输出结果后退出"，此时不会弹出工作区信任确认，
因此只应在你信任的目录中使用。

```bash
claude -p "这个函数有什么问题？"

cat error.log | claude -p "分析这些报错"
git diff | claude -p "写一个 commit message"

claude -p "列出 TODO" --output-format json
claude -p "修复 lint" --max-budget-usd 2.0
```

| 参数 | 说明 |
| --- | --- |
| `--output-format text\|json\|stream-json` | 输出格式，仅 `-p` 有效 |
| `--input-format text\|stream-json` | 输入格式，仅 `-p` 有效 |
| `--json-schema '<schema>'` | 按 JSON Schema 校验结构化输出 |
| `--max-budget-usd <金额>` | 限制本次调用的花费 |
| `--permission-prompts host\|none` | 非交互时谁来回答权限询问；`none` 表示需要询问的一律拒绝 |
| `--include-partial-messages` | 流式输出中附带增量消息（stream-json） |
| `--include-hook-events` | 输出 hook 生命周期事件（stream-json） |
| `--replay-user-messages` | 回显 stdin 中的用户消息（双 stream-json） |
| `--forward-subagent-text` | 转发子 agent 的文本/思考块（双 stream-json） |
| `--no-session-persistence` | 不保存会话，无法恢复 |
| `--prompt-suggestions true\|false` | 每轮后给出下一句提示建议 |

## 4. 模型与推理强度

```bash
claude --model sonnet                    # 别名：fable / opus / sonnet
claude --model claude-sonnet-5-5         # 或完整模型名
claude --fallback-model opus,sonnet      # 主模型过载时依次降级
claude --effort high                     # low | medium | high | xhigh | max
claude --betas <beta-name>               # 仅 API Key 用户
```

会话内可用 `/model` 切换模型、`/fast` 切换快速模式。

## 5. 权限与安全

```bash
claude --permission-mode plan          # 只读规划模式
claude --permission-mode acceptEdits   # 自动接受文件编辑，命令仍需确认
claude --allowedTools "Bash(git *) Edit"
claude --disallowedTools "Bash(rm *)"
claude --tools "Bash,Edit,Read"        # 限定可用工具；"" 表示全部禁用
claude --add-dir ../other-project      # 扩大可访问目录
claude --restricted                    # 受限模式：禁用执行命令/代码的工具
claude --safe-mode                     # 临时禁用全部自定义配置，排查问题
claude --dangerously-skip-permissions  # 跳过所有权限检查（仅限沙箱）
```

可选的权限模式：`manual`、`acceptEdits`、`plan`、`bypassPermissions`、
`auto`、`dontAsk`。`--allow-dangerously-skip-permissions` 只是把"跳过权限"
作为一个可选项启用，并不会默认开启。

`--restricted` 会移除 Bash、PowerShell 等执行类工具与 WebFetch（除非在
`--tools` 中显式指定），忽略用户/项目设置，并把文件工具限制在工作目录内。

## 6. 上下文、提示词与配置

```bash
claude --system-prompt "你是一名 CFD 专家"
claude --append-system-prompt "回答使用中文"
claude --system-prompt-snapshot off     # 每轮重新渲染系统提示（调提示词时用）
claude --autocompact auto               # 或指定 100k–1M 的 token 窗口
claude --settings ./settings.json       # 额外加载设置文件或 JSON 字符串
claude --setting-sources user,project,local
claude --bare                           # 极简模式：跳过 hooks/插件/CLAUDE.md 等
```

## 7. MCP、插件与 Agent

### MCP 服务器

```bash
claude mcp add my-server -- npx my-mcp-server
claude mcp add --transport http sentry https://mcp.sentry.dev/mcp
claude mcp add my-server -e API_KEY=xxx -- npx my-mcp-server
claude mcp add-json my-server '{"command":"npx","args":["my-mcp-server"]}'
claude mcp list
claude mcp get my-server
claude mcp login my-server
claude mcp logout my-server
claude mcp remove my-server
claude mcp add-from-claude-desktop      # 从 Claude Desktop 导入（Mac/WSL）
claude mcp serve                        # 把 Claude Code 作为 MCP 服务器运行
```

启动时临时加载 MCP 配置：

```bash
claude --mcp-config ./mcp.json
claude --mcp-config ./mcp.json --strict-mcp-config   # 只用指定配置
```

### 插件

```bash
claude plugin install <plugin>@<marketplace>
claude plugin list
claude plugin enable <plugin>
claude plugin disable <plugin>
claude plugin details <name>
claude plugin configure <plugin>
claude plugin init <name>               # 在 ~/.claude/skills/ 下脚手架
claude --plugin-dir ./my-plugin         # 仅本次会话加载本地插件
claude --plugin-url https://.../p.zip   # 仅本次会话加载远程插件
```

### 自定义 Agent

```bash
claude --agent reviewer
claude --agents '{"reviewer":{"description":"代码审查","prompt":"你是代码审查员"}}'
claude --agents ./agents.json -p "审查这个 diff"
```

## 8. 子命令一览

| 子命令 | 作用 |
| --- | --- |
| `claude agents` | 管理后台 agent |
| `claude attach <id>` | 接入后台会话 |
| `claude auth` | 管理认证 |
| `claude auto-mode` | 查看/重置自动模式配置 |
| `claude doctor` | 安装与配置体检 |
| `claude gateway` | 企业认证/遥测网关 |
| `claude import [source]` | 从其他 AI 编程工具导入配置 |
| `claude install [stable\|latest\|版本]` | 安装原生构建 |
| `claude logs <id>` | 查看后台会话输出 |
| `claude mcp` | 管理 MCP 服务器 |
| `claude plugin` | 管理插件 |
| `claude purge [path]` | 删除某项目的 Claude 数据 |
| `claude respawn [id]` | 用当前版本重启后台会话 |
| `claude rm <id>` | 删除后台会话及其 worktree |
| `claude setup-token` | 生成长期认证 token（需订阅） |
| `claude stop <id>` | 停止后台会话 |
| `claude ultrareview [target]` | 云端多 agent 代码审查 |
| `claude update` | 检查并更新 |

## 9. 会话内常用斜杠命令

在交互式会话中输入 `/` 即可看到完整列表，常用的有：

| 命令 | 作用 |
| --- | --- |
| `/help` | 帮助 |
| `/clear` | 清空当前对话 |
| `/compact` | 压缩上下文 |
| `/config` | 打开设置 |
| `/model` | 切换模型 |
| `/permissions` | 查看/管理权限规则 |
| `/context` | 查看上下文占用 |
| `/cost` | 查看本次会话花费 |
| `/doctor` | 完整诊断并可修复问题 |
| `/init` | 生成项目 `CLAUDE.md` |
| `/memory` | 管理长期记忆 |
| `/mcp` | 管理 MCP 连接 |
| `/resume` | 切换/恢复会话 |
| `/rewind` | 回退到之前的检查点 |
| `/export` | 导出会话 |
| `/review` | 代码审查 |
| `/agents` | 管理 agent |
| `/hooks` | 管理 hooks |
| `/ide` | 连接 IDE |
| `/terminal-setup` | 配置终端快捷键（如 Shift+Enter） |

## 10. 典型使用场景

```bash
# 日常开发：进入项目直接开工
cd ~/githubCode/ComputantialMath
claude

# 接着昨天的活继续
claude -c

# 从多个历史会话中挑一个
claude -r

# 一条命令问完就走
claude -p "解释 CFD/Euler/euler_unified/two_medium.py 的整体流程"

# 让 CI 输出结构化结果
claude -p "检查未提交改动" --output-format json --max-budget-usd 1

# 在隔离的 worktree 中做实验，不污染当前分支
claude -w try-new-limiter

# 后台跑长任务，稍后再看
claude --bg "跑完全部算例并更新报告"
claude logs <id>

# 排查配置问题
claude --safe-mode
claude doctor
```

## 11. 注意事项

- `-p` 模式会跳过工作区信任确认，只在可信目录中使用。
- `--dangerously-skip-permissions` 只建议在无外网的沙箱/容器里使用。
- `-c` 只继续**当前目录**的最近会话；跨目录请用 `-r` 选择。
- `--add-dir` 只是扩大工具可访问范围，不会自动把该目录纳入会话。
- 参数与子命令会随版本变化，遇到不确定的选项先运行 `claude --help` 或
  `claude <子命令> --help`。

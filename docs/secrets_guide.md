# Token 与 Secrets 配置指南

这是一个非常关键的配置步骤。即使您已经上传了代码，如果没有正确配置 Secrets，自动脚本也无法运行。

配置分为两个部分：
1.  **生成 Token**: 在 GitHub 个人账户设置中操作。
2.  **填入 Secrets**: 在具体代码仓库的设置中操作。

---

## 第一步：生成 GH_TOKEN (个人账户设置)

`GH_TOKEN` 是一个个人访问令牌（Personal Access Token），用于授权脚本代表您去读写仓库。

1.  **找到入口**
    点击页面右上角头像 -> **Settings** (设置) -> 左侧最下方 **<> Developer settings**。

2.  **创建 Token**
    *   点击左侧菜单的 **Personal access tokens** -> **Tokens (classic)**。
    *   点击右上角 **Generate new token** -> **Generate new token (classic)**。

3.  **填写配置**
    *   **Note (备注)**: 填个好记的名字，比如 `auto-star-manager`。
    *   **Expiration (过期时间)**: 建议选 **No expiration** (永不过期) 或 **90 days**，避免频繁失效导致脚本报错。
    *   **Select scopes (选择权限 - 重点!)**: 必须勾选以下两项：
        *   ☑️ **repo** (包含所有子选项) - 用于读写仓库代码和 Star 信息。
        *   ☑️ **workflow** - 用于触发 GitHub Actions。

4.  **复制 Token**
    *   点击底部的 **Generate token**。
    *   GitHub 只会显示一次这个以 `ghp_` 开头的字符串。**立刻复制并保存好它**。这就是您的 `GH_TOKEN`。

---

## 第二步：添加 Secrets (代码仓库设置)

拿到 Token 和 DeepSeek API Key 后，现在要把它们存到仓库里。

1.  **进入仓库**
    打开您存放脚本的那个 GitHub 仓库页面（例如 `my-awesome-stars`）。

2.  **进入设置**
    点击仓库顶部导航栏最右侧的 **Settings** (注意是仓库的 Settings，不是头像下的)。

3.  **找到 Secrets 菜单**
    *   在左侧侧边栏向下滚动。
    *   点击 **Secrets and variables** 展开菜单。
    *   选择 **Actions**。

4.  **添加密钥 (GH_TOKEN)**
    *   点击绿色的 **New repository secret** 按钮。
    *   **Name**: 输入 `GH_TOKEN`
    *   **Secret**: 粘贴刚才复制的 `ghp_xxxx...` 字符串。
    *   点击 **Add secret**。

---

## 第三步：添加 LLM 配置

重复上述“添加密钥”的步骤，依次添加以下 Secrets：

| Name (变量名) | Secret (值) | 说明 |
| :--- | :--- | :--- |
| `LLM_API_KEY` | `sk-xxxx...` | **必填**。您的 DeepSeek API Key |
| `LLM_BASE_URL` | `https://api.deepseek.com` | (可选) 如果用 DeepSeek 官方接口，建议填入 |
| `LLM_MODEL` | `deepseek-chat` | (可选) 指定模型版本，默认也为 deepseek-chat |

配置完成后，您的仓库 Secrets 列表中应该包含 `GH_TOKEN` 和 `LLM_API_KEY` 等条目。

# GeminiImage 插件

基于Google Gemini的图像生成插件，使用Gemini 2.0 FlashExp模型生成和编辑图片。

> 本插件fork自sofs2005的插件 [GeminiImage](https://github.com/sofs2005/GeminiImg)，感谢原作者的贡献。

> 在原版基础上增加支持Deno代理服务，解决网络限制问题。具体教程详见另外的文档 https://github.com/Lingyuzhou111/gemini

> 在原版基础上增加提示词自动翻译功能，可通过指令开启翻译或者禁用翻译模式，需要配置对应的LLM api

> 在原版基础上增加自定义参考图编辑功能，通过特定格式的指令触发

## 功能特点

- 根据文本描述生成高质量图片
- 编辑已有图片，调整风格和内容
- 支持参考图片编辑，可以上传自己的图片进行编辑
- 支持会话模式，可以连续对话修改图片
- 支持代理设置，方便国内用户使用
- 支持Deno代理服务，解决网络限制问题
- 支持前置翻译，自动将中文提示词翻译成英文，提高生成质量

## 安装方法

1. 确保你已经安装了dify-on-wechat项目
2. 使用管理员插件安装：
   ```
   #installp https://github.com/Lingyuzhou111/GeminiImage.git
   ```
3. 安装完成后使用 `#scanp` 命令扫描加载插件
4. 配置插件：将 `plugins/GeminiImg/config.json.template` 复制为 `plugins/GeminiImage/config.json` 并编辑

## 配置说明

```json
{
  "enable": true,                                 # 是否启用插件
  "gemini_api_key": "your_api_key_here",          # Google Gemini API密钥
  "model": "gemini-2.0-flash-exp-image-generation", # 使用的模型名称
  "commands": ["$生成图片", "$画图", "$图片生成"],    # 生成图片的命令
  "edit_commands": ["$编辑图片", "$修改图片"],       # 编辑图片的命令
  "reference_edit_commands": ["$参考图", "$编辑参考图"], # 参考图编辑的命令
  "exit_commands": ["$结束对话", "$退出对话", "$关闭对话", "$结束"], # 结束对话的命令
  "enable_points": false,                         # 是否启用积分系统
  "generate_image_cost": 10,                      # 生成图片消耗的积分
  "edit_image_cost": 15,                          # 编辑图片消耗的积分
  "save_path": "temp",                            # 图片保存路径
  "admins": [],                                   # 管理员列表，这些用户不受积分限制
  "enable_proxy": false,                          # 是否启用HTTP代理
  "proxy_url": "",                                # HTTP代理服务器URL
  "use_proxy_service": true,                      # 是否启用Deno代理服务
  "proxy_service_url": "",                        # Deno代理服务URL
  "translate_api_base": "https://open.bigmodel.cn/api/paas/v4", # 翻译API基础URL
  "translate_api_key": "your_translate_api_key",  # 翻译API密钥
  "translate_model": "glm-4-flash",               # 翻译使用的模型
  "enable_translate": true                        # 是否启用前置翻译功能
}
```

## 使用方法

### 生成图片

发送以下命令生成图片：
```
$生成图片 一只可爱的柴犬坐在草地上，阳光明媚
```

### 编辑图片

在生成图片后，可以继续发送命令编辑图片：
```
$编辑图片 将柴犬换成一只猫，保持相同的场景
```

或者直接发送描述继续对话：
```
给猫咪戴上一顶帽子
```

### 参考图片编辑

你可以上传自己的图片，并使用参考图命令对其进行编辑：

1. 发送参考图编辑命令：
```
$参考图 将图片中的人物变成卡通风格
```

2. 插件会提示你上传图片，上传后会自动处理并返回编辑后的图片

3. 上传图片后，你可以继续发送命令对图片进行进一步编辑，就像使用普通的编辑图片功能一样

这个功能允许你编辑任何自己的图片，而不仅限于插件生成的图片。

> **注意**：上传图片后，如果3分钟内没有收到图片，系统会自动结束等待状态。如果遇到图片上传问题，可以尝试以下方法：
> - 确保图片格式为常见的图片格式（JPG、PNG等）
> - 图片大小不要超过10MB
> - 如果微信无法发送原图，可以尝试截图后发送
> - 如果仍然无法上传，可以尝试重新发送参考图命令

### 结束对话

当不需要继续编辑图片时，可以结束对话：
```
g结束对话
```

### 前置翻译功能

当使用以g开头的命令时（如g画图、g改图），插件会自动将中文提示词翻译成英文，然后再调用Gemini API进行图像生成或编辑，这样可以提高生成质量。

示例：
```
g画图 一只可爱的猫咪，手绘风格
```
插件会先将提示词翻译为"A cute kitten, in hand-drawn style"，再调用Gemini API进行生图。

### 控制翻译功能

用户可以随时开启或关闭前置翻译功能：

```
g开启翻译    # 开启前置翻译功能
g关闭翻译    # 关闭前置翻译功能
```

开启或关闭翻译功能的设置是针对每个用户单独生效的，不会影响其他用户。

## 网络配置

插件提供了两种网络配置方式，帮助解决网络限制问题：

1. **HTTP代理**：通过设置 `enable_proxy` 和 `proxy_url` 来使用HTTP代理访问Google API
2. **Deno代理服务**：通过设置 `use_proxy_service` 和 `proxy_service_url` 来使用Deno代理服务访问Google API

对于无法直接访问Google服务的用户，建议启用Deno代理服务。

## 注意事项

1. 需要申请Google Gemini API密钥，可以在[Google AI Studio](https://aistudio.google.com/)申请
2. 国内用户可能需要使用代理才能访问Google API
3. 图片生成和编辑可能需要一些时间，请耐心等待
4. 每个会话的有效期为10分钟，超时后需要重新开始
5. 不支持生成违反内容政策的图片，如色情、暴力等内容
6. 如需更改使用的模型，请直接修改配置文件中的 `model` 值
7. 前置翻译功能需要配置翻译API密钥，否则将使用原始提示词

## 版本历史

- v1.0.0: 初始版本，支持基本的图片生成和编辑功能
- v1.1.0: 添加Deno代理服务支持，优化配置管理
- v1.2.0: 添加前置翻译功能，自动将中文提示词翻译成英文，提高生成质量

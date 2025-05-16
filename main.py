import os
import json
import tomllib
import traceback
import uuid
import time
import base64
import re
import random
import asyncio
import copy
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from collections import defaultdict

# 标准库导入
import aiohttp
from PIL import Image

# 第三方库导入
from loguru import logger

# 框架导入
from WechatAPI import WechatAPIClient
from database.XYBotDB import XYBotDB
from utils.decorators import on_text_message, on_image_message, on_file_message, add_job_safe, schedule
from utils.plugin_base import PluginBase

# 导入增强的系统提示词
from plugins.GeminiImage.enhanced_system_prompt import (
    STANDARD_SYSTEM_PROMPT,
    DETAILED_SYSTEM_PROMPT,
    MULTI_IMAGE_SYSTEM_PROMPT,
    REVERSE_PROMPT,
    EDIT_IMAGE_SYSTEM_PROMPT,
    MERGE_IMAGE_SYSTEM_PROMPT,
    IMAGE_ANALYSIS_PROMPT
)


class GeminiImage(PluginBase):
    """基于Google Gemini的图像生成插件"""

    description = "基于Google Gemini的图像生成插件"
    author = "XYBot"
    version = "2.0.0"

    def __init__(self):
        super().__init__()

        try:
            # 读取配置
            config_path = os.path.join(os.path.dirname(__file__), "config.toml")
            with open(config_path, "rb") as f:
                config = tomllib.load(f)

            # 获取Gemini配置
            plugin_config = config.get("GeminiImage", {})
            self.enable = plugin_config.get("enable", False)

            # 读取多个API密钥
            api_keys = plugin_config.get("gemini_api_keys", [])
            # 兼容旧版配置，如果存在gemini_api_key且api_keys为空，则使用旧版配置
            if not api_keys and "gemini_api_key" in plugin_config:
                old_key = plugin_config.get("gemini_api_key", "")
                if old_key:
                    api_keys = [old_key]
                    logger.info("使用旧版配置中的gemini_api_key")

            # 过滤掉空字符串
            self.api_keys = [key for key in api_keys if key.strip()]
            if not self.api_keys:
                logger.warning("未配置有效的Gemini API密钥")
                self.api_keys = [""]  # 添加一个空字符串，避免索引错误

            # 初始化API密钥轮询索引
            self.current_key_index = 0

            # 初始化会话ID与API密钥的映射关系
            self.session_key_mapping = {}

            # 初始化API密钥错误计数
            self.key_error_counts = {key: 0 for key in self.api_keys}

            # 初始化API密钥最后使用时间
            self.key_last_used = {key: 0 for key in self.api_keys}

            # 获取当前可用的API密钥（用于向后兼容）
            self.api_key = self.api_keys[0] if self.api_keys else ""

            self.model = plugin_config.get("model", "gemini-2.0-flash-exp-image-generation")

            # 获取命令配置
            self.commands = plugin_config.get("commands", ["#生成图片", "#画图", "#图片生成"])
            self.edit_commands = plugin_config.get("edit_commands", ["#编辑图片", "#修改图片"])
            self.exit_commands = plugin_config.get("exit_commands", ["#结束对话", "#退出对话", "#关闭对话", "#结束"])  # 从配置读取结束对话命令

            # 获取新增命令配置
            self.merge_commands = plugin_config.get("merge_commands", ["#融图", "#合成图片"])
            self.start_merge_commands = plugin_config.get("start_merge_commands", ["#开始融合", "#生成融图"])
            self.image_reverse_commands = plugin_config.get("image_reverse_commands", ["#反推提示", "#反推"])
            self.prompt_enhance_commands = plugin_config.get("prompt_enhance_commands", ["#提示词", "#生成提示词"])
            self.image_analysis_commands = plugin_config.get("image_analysis_commands", ["#分析图片", "#图片分析", "g分析"])

            # 记录命令配置
            logger.info(f"GeminiImage插件融图命令配置: {self.merge_commands}")
            logger.info(f"GeminiImage插件开始融合命令配置: {self.start_merge_commands}")
            logger.info(f"GeminiImage插件反向提示词命令配置: {self.image_reverse_commands}")
            logger.info(f"GeminiImage插件提示词增强命令配置: {self.prompt_enhance_commands}")
            logger.info(f"GeminiImage插件图片分析命令配置: {self.image_analysis_commands}")

            # 获取积分配置
            self.enable_points = plugin_config.get("enable_points", True)
            self.show_points_message = plugin_config.get("show_points_message", True)
            self.generate_cost = plugin_config.get("generate_image_cost", 10)
            self.edit_cost = plugin_config.get("edit_image_cost", 15)
            self.merge_cost = plugin_config.get("merge_image_cost", 20)
            self.reverse_cost = plugin_config.get("reverse_image_cost", 5)
            self.analysis_cost = plugin_config.get("analysis_image_cost", 5)

            # 获取图片保存配置
            self.save_path = plugin_config.get("save_path", "temp")
            self.save_dir = os.path.join(os.path.dirname(__file__), self.save_path)
            os.makedirs(self.save_dir, exist_ok=True)

            # 获取管理员列表
            self.admins = plugin_config.get("admins", [])

            # 获取代理配置
            self.enable_proxy = plugin_config.get("enable_proxy", False)
            self.proxy_url = plugin_config.get("proxy_url", "")

            # 获取API基础URL配置
            self.base_url = plugin_config.get("base_url", "https://generativelanguage.googleapis.com")
            # 移除末尾的斜杠，确保不会出现双斜杠
            if self.base_url and self.base_url.endswith("/"):
                self.base_url = self.base_url.rstrip("/")

            # 检查是否是标准Google AI URL
            if self.base_url and "generativelanguage.googleapis.com" not in self.base_url:
                logger.warning(f"Base URL '{self.base_url}' doesn't look like standard Google AI URL. Ensure it's correct.")

            # 获取提示词增强相关配置
            self.enhance_prompt = plugin_config.get("enhance_prompt", True)
            self.prompt_model = plugin_config.get("prompt_model", "gemini-2.0-flash")
            self.reverse_model = plugin_config.get("reverse_model", "gemini-2.0-flash")
            self.analysis_model = plugin_config.get("analysis_model", "gemini-2.0-flash")

            # 获取对话前缀配置
            self.conversation_prefixes = plugin_config.get("conversation_prefixes", ["@绘图", "@图片", "@Gemini"])
            self.require_prefix_for_conversation = plugin_config.get("require_prefix_for_conversation", True)

            # 获取重试机制相关配置
            self.max_retries = plugin_config.get("max_retries", 3)
            self.initial_retry_delay = plugin_config.get("initial_retry_delay", 1)
            self.max_retry_delay = plugin_config.get("max_retry_delay", 10)

            # 获取融图相关配置
            self.max_merge_images = plugin_config.get("max_merge_images", 5)
            self.merge_image_wait_timeout = plugin_config.get("merge_image_wait_timeout", 180)

            # 获取反向提示词相关配置
            self.reverse_image_wait_timeout = plugin_config.get("reverse_image_wait_timeout", 180)

            # 初始化数据库
            self.db = XYBotDB()

            # 初始化会话状态，用于保存上下文
            self.conversations = defaultdict(list)  # 用户ID -> 对话历史列表
            self.conversation_expiry = 600  # 会话过期时间(秒)
            self.conversation_timestamps = {}  # 用户ID -> 最后活动时间

            # 存储最后一次生成的图片路径
            self.last_images = {}  # 会话标识 -> 最后一次生成的图片路径

            # 全局图片缓存，用于存储最近接收到的图片
            # 修改为使用(聊天ID, 用户ID)作为键，以区分群聊中不同用户
            self.image_cache = {}  # (聊天ID, 用户ID) -> {content: bytes, timestamp: float}
            self.image_cache_timeout = 300  # 图片缓存过期时间(秒)

            # 融图相关状态变量
            self.waiting_for_merge_images = {}  # 用户ID -> {"提示词": 提示词, "图片列表": [图片数据], "开始时间": 时间戳}

            # 反推提示词相关状态变量
            self.waiting_for_reverse_image = {}  # 用户ID -> 是否等待反推图片
            self.waiting_for_reverse_image_time = {}  # 用户ID -> 开始等待反推图片的时间戳

            # 图片分析相关状态变量
            self.waiting_for_analyze_image = {}  # 用户ID -> 是否等待分析图片
            self.waiting_for_analyze_image_time = {}  # 用户ID -> 开始等待分析图片的时间戳
            self.waiting_for_analyze_image_query = {}  # 用户ID -> 分析图片时的具体问题
            self.analyze_image_wait_timeout = plugin_config.get("analyze_image_wait_timeout", 180)

            # 保存配置对象，供其他方法使用
            self.config = plugin_config

            # 验证关键配置
            if not any(self.api_keys) or all(not key for key in self.api_keys):
                logger.warning("GeminiImage插件未配置有效的API密钥")

            # 定时任务通过@schedule装饰器自动注册，无需手动注册
            logger.info("GeminiImage插件定时清理任务将通过装饰器自动注册")

            logger.info("GeminiImage插件初始化成功")
            if self.enable_proxy:
                logger.info(f"GeminiImage插件已启用代理: {self.proxy_url}")

        except Exception as e:
            logger.error(f"GeminiImage插件初始化失败: {str(e)}")
            logger.error(traceback.format_exc())
            self.enable = False

    @on_text_message(priority=30)
    async def handle_generate_image(self, bot: WechatAPIClient, message: dict) -> bool:
        """处理生成图片的命令"""
        if not self.enable:
            return True  # 插件未启用，继续执行后续插件

        # 检查是否是融图命令
        # 尝试获取不同格式的消息内容
        text = message.get("content", message.get("Content", "")).strip()
        chat_id = message.get("chat_id", message.get("FromWxid", ""))
        user_id = message.get("user_id", message.get("SenderWxid", ""))
        conversation_key = f"{chat_id}_{user_id}"

        logger.info(f"GeminiImage收到消息: {text[:20]}{'...' if len(text) > 20 else ''}")

        # 首先检查消息是否带有所需前缀，无论是否有活跃会话
        has_prefix, processed_content = self._check_message_prefix(text)

        # 如果消息带有前缀，记录详细日志并立即处理为连续对话
        if has_prefix:
            logger.info(f"检测到前缀消息: '{text}'，处理后的消息: '{processed_content}'")

            # 检查是否有活跃会话
            if conversation_key not in self.conversations:
                logger.info(f"没有找到活跃会话，但检测到前缀，为用户 {user_id} 创建新会话")
                # 创建新会话
                self.conversations[conversation_key] = []
                self.conversation_timestamps[conversation_key] = time.time()

            # 更新content为处理后的内容（已移除前缀）
            content = processed_content

            # 在群聊中，检查是否包含唤醒词或@机器人
            if message.get("IsGroup", False):
                # 在群聊中必须包含唤醒词或@机器人才能继续对话
                if not self.has_wake_word(content) and not self.is_at_message(message):
                    # 没有唤醒词，不处理
                    logger.info(f"群聊中没有唤醒词或@机器人，不处理消息")
                    return True

                # 清理消息内容，移除唤醒词和@标记
                clean_content = self.get_clean_content(content)
                if not clean_content.strip():
                    # 清理后内容为空，不处理
                    logger.info(f"清理后内容为空，不处理消息")
                    return True

                # 使用清理后的内容继续对话
                content = clean_content
                logger.info(f"群聊中清理后的内容: '{content}'")

            # 修改原始消息内容，确保后续处理使用处理后的内容
            message["Content"] = content

            # 立即处理连续对话
            try:
                logger.info(f"立即处理连续对话: 用户={user_id}, 内容='{content}'")

                # 检查积分
                if self.enable_points and user_id not in self.admins:
                    points = self.db.get_points(user_id)
                    if points < self.generate_cost:
                        await bot.send_at_message(chat_id, f"\n您的积分不足，生成图片需要{self.generate_cost}积分，您当前有{points}积分", [user_id])
                        return False  # 积分不足，阻止后续插件执行

                # 发送处理中消息
                await bot.send_at_message(chat_id, "\n正在处理您的请求，请稍候...", [user_id])

                # 获取上下文历史
                conversation_history = self.conversations[conversation_key]
                logger.info(f"对话历史长度: {len(conversation_history)}")

                # 添加用户提示到会话
                user_message = {"role": "user", "parts": [{"text": content}]}

                # 检查是否有上一次生成的图片，如果有则自动作为输入
                last_image_path = self.last_images.get(conversation_key)
                logger.info(f"上一次图片路径: {last_image_path}")

                # 如果没有找到图片路径，尝试从缓存获取
                if not last_image_path or not os.path.exists(last_image_path):
                    logger.info("未找到上一次图片路径，尝试从缓存获取")
                    image_data = await self._get_recent_image(chat_id, user_id)
                    if image_data:
                        # 如果找到缓存的图片，保存到本地再处理
                        image_path = os.path.join(self.save_dir, f"temp_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                        with open(image_path, "wb") as f:
                            f.write(image_data)
                        self.last_images[conversation_key] = image_path
                        last_image_path = image_path
                        logger.info(f"从缓存找到图片，保存到：{image_path}")

                        # 再次保存到缓存，确保使用了所有可能的键格式
                        self._save_image_to_cache(chat_id, user_id, image_data)
                    else:
                        # 尝试使用更宽松的条件查找图片路径
                        logger.info("未找到缓存图片，尝试使用更宽松的条件查找图片路径")
                        for key, value in self.last_images.items():
                            if (chat_id in key or user_id in key) and os.path.exists(value):
                                last_image_path = value
                                logger.info(f"使用宽松条件找到图片路径: {last_image_path}, 键: {key}")
                                break

                if last_image_path and os.path.exists(last_image_path):
                    # 处理带图片的连续对话
                    logger.info(f"找到上一次图片，将使用该图片进行编辑")
                    # 读取上一次生成的图片
                    with open(last_image_path, "rb") as f:
                        image_data = f.read()

                    # 调用编辑图片API
                    logger.info(f"调用编辑图片API")
                    # 在连续对话模式下，设置is_continuous_dialogue为True
                    # 添加中文提示，确保返回中文结果
                    content_with_lang = f"请用中文回答：{content}"
                    edited_images, text_responses = await self._edit_image(content_with_lang, image_data, conversation_history, is_continuous_dialogue=True)

                    # 处理编辑图片结果
                    # 确保 edited_images 和 text_responses 不为 None
                    if edited_images is None:
                        edited_images = []
                    if text_responses is None:
                        text_responses = []

                    if len(edited_images) > 0 and edited_images[0]:
                        logger.info(f"成功获取编辑后的图片结果")
                        # 保存编辑后的图片
                        new_image_path = os.path.join(self.save_dir, f"edited_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                        with open(new_image_path, "wb") as f:
                            f.write(edited_images[0])

                        # 更新最后生成的图片路径
                        self.last_images[conversation_key] = new_image_path

                        # 扣除积分
                        if self.enable_points and user_id not in self.admins:
                            points = self.db.get_points(user_id)
                            self.db.add_points(user_id, -self.edit_cost)  # 使用编辑积分
                            points_msg = f"已扣除{self.edit_cost}积分，当前剩余{points - self.edit_cost}积分"
                        else:
                            points_msg = ""

                        # 发送文本回复（如果有）
                        first_valid_text = next((t for t in text_responses if t), None)
                        if first_valid_text:
                            # 清理文本，去除多余的空格和换行
                            cleaned_text = first_valid_text.strip()
                            # 将多个连续空格替换为单个空格
                            import re
                            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                            # 移除文本开头和结尾的引号
                            if cleaned_text.startswith('"') and cleaned_text.endswith('"'):
                                cleaned_text = cleaned_text[1:-1]
                            # 构建消息文本，避免在没有积分消息时添加多余的换行
                            if points_msg:
                                message_text = f"{cleaned_text}\n\n{points_msg}"
                            else:
                                message_text = cleaned_text
                            await bot.send_text_message(chat_id, message_text)
                        else:
                            await bot.send_text_message(chat_id, f"图片编辑成功！{points_msg if points_msg else ''}")
                        # 添加短暂延迟，确保文本发送完成
                        await asyncio.sleep(0.5)

                        # 发送图片
                        logger.info(f"发送编辑后的图片")
                        with open(new_image_path, "rb") as f:
                            await bot.send_image_message(chat_id, f.read())
                        # 添加延迟，确保图片发送完成
                        await asyncio.sleep(1.5)

                        # 更新会话历史
                        # 添加包含图片的用户消息
                        user_message = {
                            "role": "user",
                            "parts": [
                                {"text": content},
                                {"image_url": last_image_path}
                            ]
                        }
                        conversation_history.append(user_message)

                        assistant_message = {
                            "role": "model",
                            "parts": [
                                {"text": first_valid_text if first_valid_text else "我已编辑了图片"},
                                {"image_url": new_image_path}
                            ]
                        }
                        conversation_history.append(assistant_message)

                        # 限制会话历史长度
                        if len(conversation_history) > 10:
                            conversation_history = conversation_history[-10:]

                        # 更新会话时间戳
                        self.conversation_timestamps[conversation_key] = time.time()
                    else:
                        # 检查是否有文本响应，可能是内容被拒绝
                        first_valid_text = next((t for t in text_responses if t), None)
                        if first_valid_text:
                            # 内容审核拒绝的情况，翻译并转发拒绝消息给用户
                            # 检查是否是JSON格式的错误信息
                            try:
                                error_data = json.loads(first_valid_text)
                                # 构建友好的错误消息
                                error_message = "图片编辑请求被拒绝。"

                                # 尝试从错误数据中提取有用信息
                                if "candidates" in error_data and error_data["candidates"]:
                                    candidate = error_data["candidates"][0]
                                    if "finishReason" in candidate:
                                        error_message += f"原因: {candidate['finishReason']}. "

                                    if "safetyRatings" in candidate:
                                        blocked_categories = []
                                        for rating in candidate["safetyRatings"]:
                                            if rating.get("blocked", False):
                                                category = rating.get("category", "未知类别")
                                                probability = rating.get("probability", "未知")
                                                blocked_categories.append(f"{category}({probability})")

                                        if blocked_categories:
                                            error_message += f"被拒绝的类别: {', '.join(blocked_categories)}。"

                                error_message += "请修改您的请求。"
                                translated_response = error_message
                            except json.JSONDecodeError:
                                # 不是JSON格式，使用常规翻译
                                translated_response = self._translate_gemini_message(first_valid_text)

                            await bot.send_at_message(chat_id, f"\n{translated_response}", [user_id])
                            logger.warning(f"API拒绝编辑图片，提示: {first_valid_text}")
                        else:
                            logger.error(f"编辑图片失败，未获取到有效的图片数据")
                            await bot.send_at_message(chat_id, "\n图片编辑失败，请稍后再试或修改描述", [user_id])
                else:
                    # 处理纯文本连续对话，生成新图片
                    logger.info(f"没有找到上一次图片或文件不存在，将生成新图片")
                    # 生成新图片
                    parts_list, image_count = await self._generate_image(content, conversation_history, is_continuous_dialogue=True)

                    # 处理生成图片结果
                    if image_count > 0:
                        logger.info(f"成功获取生成的图片结果")

                        # 扣除积分
                        if self.enable_points and user_id not in self.admins:
                            points = self.db.get_points(user_id)
                            self.db.add_points(user_id, -self.generate_cost)
                            points_msg = f"已扣除{self.generate_cost}积分，当前剩余{points - self.generate_cost}积分"
                            # 先发送积分信息
                            await bot.send_text_message(chat_id, points_msg)
                            # 添加短暂延迟
                            await asyncio.sleep(0.5)

                        # 保存图片路径列表，用于更新会话历史
                        image_paths = []
                        last_image_path = None

                        # 判断是否是多图文请求
                        is_multi_image = self._is_multi_image_request(content)

                        if is_multi_image:
                            # 多图文请求的处理方式
                            logger.info("多图文请求，按照原版程序的方式处理")

                            # 提取所有文本部分（故事内容）
                            story_contents = [part["content"] for part in parts_list if part["type"] == "text"]

                            # 提取所有图片部分
                            image_parts = [part["content"] for part in parts_list if part["type"] == "image"]

                            # 确保我们有故事内容
                            if not story_contents:
                                logger.warning("没有找到故事内容，使用默认文本")
                                story_contents = ["这是一个精彩的故事场景"]

                            # 保存图片到本地并准备发送
                            saved_images = []
                            for i, image_data in enumerate(image_parts):
                                # 保存图片到本地
                                image_path = os.path.join(self.save_dir, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}_{i}.png")
                                with open(image_path, "wb") as f:
                                    f.write(image_data)
                                saved_images.append(image_path)
                                # 保存图片路径
                                image_paths.append(image_path)
                                last_image_path = image_path

                            # 按照一一对应的方式发送图片和文本
                            logger.info(f"准备发送 {len(saved_images)} 张图片和 {len(story_contents)} 段文本")

                            # 确定要发送的对数
                            pairs_count = min(len(saved_images), len(story_contents))

                            # 一一对应发送图片和文本，确保每次发送完成后再发送下一条
                            for i in range(pairs_count):
                                # 先发送文本
                                if i < len(story_contents) and story_contents[i].strip():
                                    await bot.send_text_message(chat_id, story_contents[i])
                                    # 添加短暂延迟，确保文本发送完成
                                    await asyncio.sleep(0.5)

                                # 再发送图片
                                if i < len(saved_images):
                                    with open(saved_images[i], "rb") as f:
                                        await bot.send_image_message(chat_id, f.read())
                                    # 添加延迟，确保图片发送完成
                                    await asyncio.sleep(1.5)

                            # 如果还有剩余的文本，发送剩余文本
                            for i in range(pairs_count, len(story_contents)):
                                if story_contents[i].strip():
                                    await bot.send_text_message(chat_id, story_contents[i])
                                    # 添加短暂延迟
                                    await asyncio.sleep(0.5)

                            # 如果还有剩余的图片，发送剩余图片
                            for i in range(pairs_count, len(saved_images)):
                                with open(saved_images[i], "rb") as f:
                                    await bot.send_image_message(chat_id, f.read())
                                # 添加延迟
                                await asyncio.sleep(1.5)
                        else:
                            # 常规请求的处理方式
                            # 按照原始顺序发送文本和图片
                            current_text = ""

                            for part in parts_list:
                                if part["type"] == "text":
                                    # 累积文本，直到遇到图片才发送
                                    current_text += part["content"]
                                elif part["type"] == "image":
                                    # 如果有累积的文本，先发送文本
                                    if current_text.strip():
                                        await bot.send_text_message(chat_id, current_text)
                                        current_text = ""
                                        # 添加短暂延迟，确保文本发送完成
                                        await asyncio.sleep(0.5)

                                    # 保存图片到本地
                                    image_path = os.path.join(self.save_dir, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                                    with open(image_path, "wb") as f:
                                        f.write(part["content"])

                                    # 发送图片
                                    with open(image_path, "rb") as f:
                                        await bot.send_image_message(chat_id, f.read())
                                    # 添加延迟，确保图片发送完成
                                    await asyncio.sleep(1.5)

                                    # 保存图片路径
                                    image_paths.append(image_path)
                                    last_image_path = image_path

                            # 发送剩余的文本（如果有）
                            if current_text.strip():
                                await bot.send_text_message(chat_id, current_text)

                        # 保存最后生成的图片路径（用于后续编辑）
                        if last_image_path:
                            self.last_images[conversation_key] = last_image_path

                        logger.info(f"发送生成的图片完成")

                        # 更新会话历史
                        conversation_history.append(user_message)

                        # 创建助手消息部分
                        assistant_parts = []

                        # 按照原始顺序添加文本和图片
                        image_index = 0
                        for part in parts_list:
                            if part["type"] == "text":
                                assistant_parts.append({"text": part["content"]})
                            elif part["type"] == "image" and image_index < len(image_paths):
                                assistant_parts.append({"image_url": image_paths[image_index]})
                                image_index += 1

                        # 如果没有文本，添加默认文本
                        if not any("text" in p for p in assistant_parts):
                            assistant_parts.insert(0, {"text": "我已基于您的提示生成了图片"})

                        assistant_message = {
                            "role": "model",
                            "parts": assistant_parts
                        }
                        conversation_history.append(assistant_message)

                        # 限制会话历史长度
                        if len(conversation_history) > 10:
                            conversation_history = conversation_history[-10:]

                        # 更新会话时间戳
                        self.conversation_timestamps[conversation_key] = time.time()
                    else:
                        # 检查是否有文本响应，可能是内容被拒绝
                        # 尝试从 parts_list 中提取文本响应
                        text_parts = [part["content"] for part in parts_list if part["type"] == "text"]

                        if text_parts:
                            # 内容审核拒绝的情况，翻译并转发拒绝消息给用户
                            # 检查是否是JSON格式的错误信息
                            try:
                                error_data = json.loads(text_parts[0])
                                # 构建友好的错误消息
                                error_message = "图片生成请求被拒绝。"

                                # 尝试从错误数据中提取有用信息
                                if "candidates" in error_data and error_data["candidates"]:
                                    candidate = error_data["candidates"][0]
                                    if "finishReason" in candidate:
                                        error_message += f"原因: {candidate['finishReason']}. "

                                    if "safetyRatings" in candidate:
                                        blocked_categories = []
                                        for rating in candidate["safetyRatings"]:
                                            if rating.get("blocked", False):
                                                category = rating.get("category", "未知类别")
                                                probability = rating.get("probability", "未知")
                                                blocked_categories.append(f"{category}({probability})")

                                        if blocked_categories:
                                            error_message += f"被拒绝的类别: {', '.join(blocked_categories)}。"

                                error_message += "请修改您的请求。"
                                translated_response = error_message
                            except json.JSONDecodeError:
                                # 不是JSON格式，使用常规翻译
                                translated_response = self._translate_gemini_message(text_parts[0])

                            await bot.send_at_message(chat_id, f"\n{translated_response}", [user_id])
                            logger.warning(f"API拒绝生成图片，提示: {text_parts[0]}")
                        else:
                            logger.error(f"生成图片失败，未获取到有效的图片数据")
                            await bot.send_at_message(chat_id, "\n图片生成失败，请稍后再试或修改提示词", [user_id])

                return False  # 已处理命令，阻止后续插件执行
            except Exception as e:
                logger.error(f"连续对话处理失败: {str(e)}")
                logger.error(traceback.format_exc())
                await bot.send_at_message(chat_id, f"\n处理失败: {str(e)}", [user_id])
                return False  # 已处理命令，阻止后续插件执行

            # 更新text变量，用于后续处理（如果需要）
            text = processed_content

        # 处理融图命令 - 使用正则表达式来检查命令
        # 创建一个正则表达式模式，匹配任何以融图命令开头的文本
        import re
        merge_cmd_pattern = '|'.join(re.escape(cmd) for cmd in self.merge_commands)
        merge_cmd_regex = re.compile(f'^({merge_cmd_pattern})(\\s|$)')

        match = merge_cmd_regex.match(text)

        # 如果匹配成功，处理融图命令
        if match:
            logger.info("匹配成功，开始处理融图命令")
            # 设置为False，阻断后续插件执行

            try:
                # 检查是否有足够的积分
                if self.enable_points and self.merge_cost > 0:
                    points = await self.db.get_user_points(user_id)
                    logger.info(f"用户 {user_id} 当前积分: {points}, 需要积分: {self.merge_cost}")
                    if points < self.merge_cost:
                        await bot.send_text_message(chat_id, f"您的积分不足，需要 {self.merge_cost} 积分才能使用融图功能，当前积分: {points}")
                        return True

                # 提取提示词
                prompt = text
                matched_cmd = match.group(1)  # 获取匹配到的命令
                prompt = text[len(matched_cmd):].strip()

                # 初始化等待融图状态
                self.waiting_for_merge_images[user_id] = {
                    "提示词": prompt,
                    "图片列表": [],
                    "开始时间": time.time()
                }

                # 发送提示消息
                await bot.send_text_message(chat_id, f"请上传要融合的图片（最多 {self.max_merge_images} 张），然后发送 {self.start_merge_commands[0]} 开始融合")
                return False  # 阻断后续插件执行
            except Exception as e:
                logger.error(f"处理融图命令异常: {str(e)}")
                logger.error(traceback.format_exc())
                await bot.send_text_message(chat_id, f"处理融图命令失败: {str(e)}")
                return False  # 阻断后续插件执行

        # 处理开始融合命令 - 使用正则表达式来检查命令
        start_merge_cmd_pattern = '|'.join(re.escape(cmd) for cmd in self.start_merge_commands)
        start_merge_cmd_regex = re.compile(f'^({start_merge_cmd_pattern})(\\s|$)')
        start_match = start_merge_cmd_regex.match(text)

        if start_match:
            logger.info("匹配成功，开始处理开始融合命令")

            try:
                # 检查所有可能的用户ID，确保能够找到等待融图状态
                possible_user_ids = [user_id, chat_id, message.get("SenderWxid", ""), message.get("FromWxid", "")]

                found_user_id = None
                for possible_id in possible_user_ids:
                    if possible_id in self.waiting_for_merge_images:
                        found_user_id = possible_id
                        break

                if found_user_id:
                    user_id = found_user_id
                    merge_data = self.waiting_for_merge_images[user_id]
                    image_list = merge_data["图片列表"]
                    prompt = merge_data["提示词"]

                    if not image_list:
                        await bot.send_text_message(chat_id, "请先上传要融合的图片")
                        return False  # 阻断后续插件执行

                    # 扣除积分
                    if self.enable_points and self.merge_cost > 0:
                        points_before = await self.db.get_user_points(user_id)
                        await self.db.update_user_points(user_id, -self.merge_cost)
                        points_after = await self.db.get_user_points(user_id)
                        logger.info(f"用户 {user_id} 融图扣除积分 {self.merge_cost}，积分变化: {points_before} -> {points_after}")

                        # 如果启用了积分消息显示，发送积分消息
                        if self.show_points_message:
                            points_msg = f"已扣除{self.merge_cost}积分，当前剩余{points_after}积分"
                            await bot.send_text_message(chat_id, points_msg)
                            # 添加短暂延迟
                            await asyncio.sleep(0.5)

                    # 处理融图请求
                    success = await self._handle_merge_images(bot, message, prompt, image_list)

                    # 清除等待状态
                    del self.waiting_for_merge_images[user_id]
                    return False  # 阻断后续插件执行
                else:
                    await bot.send_text_message(chat_id, "请先发送融图命令并上传图片")
                    return False  # 阻断后续插件执行
            except Exception as e:
                logger.error(f"处理开始融合命令异常: {str(e)}")
                logger.error(traceback.format_exc())
                await bot.send_text_message(chat_id, f"处理开始融合命令失败: {str(e)}")
                return False  # 阻断后续插件执行

        # 处理反向提示词命令 - 使用正则表达式来检查命令
        reverse_cmd_pattern = '|'.join(re.escape(cmd) for cmd in self.image_reverse_commands)
        reverse_cmd_regex = re.compile(f'^({reverse_cmd_pattern})(\\s|$)')
        reverse_match = reverse_cmd_regex.match(text)

        if reverse_match:
            # 检查是否有足够的积分
            if self.enable_points and self.reverse_cost > 0:
                points = await self.db.get_user_points(user_id)
                if points < self.reverse_cost:
                    await bot.send_text_message(chat_id, f"您的积分不足，需要 {self.reverse_cost} 积分才能使用反向提示词功能，当前积分: {points}")
                    return False  # 阻断后续插件执行

            # 检查是否有最近的图片
            image_data = await self._get_recent_image(chat_id, user_id)
            if image_data:
                # 扣除积分
                if self.enable_points and self.reverse_cost > 0:
                    points_before = await self.db.get_user_points(user_id)
                    await self.db.update_user_points(user_id, -self.reverse_cost)
                    points_after = await self.db.get_user_points(user_id)

                    # 如果启用了积分消息显示，发送积分消息
                    if self.show_points_message:
                        points_msg = f"已扣除{self.reverse_cost}积分，当前剩余{points_after}积分"
                        await bot.send_text_message(chat_id, points_msg)
                        # 添加短暂延迟
                        await asyncio.sleep(0.5)

                # 处理反向提示词请求
                await self._handle_reverse_image(bot, message, image_data)
                return False  # 阻断后续插件执行
            else:
                # 设置等待状态，等待用户上传图片
                self.waiting_for_reverse_image[user_id] = True
                self.waiting_for_reverse_image_time[user_id] = time.time()
                await bot.send_text_message(chat_id, "请上传要生成提示词的图片")
                return False  # 阻断后续插件执行

        # 处理图片分析命令 - 使用正则表达式来检查命令
        analysis_cmd_pattern = '|'.join(re.escape(cmd) for cmd in self.image_analysis_commands)
        analysis_cmd_regex = re.compile(f'^({analysis_cmd_pattern})(\\s|$)')
        analysis_match = analysis_cmd_regex.match(text)

        if analysis_match:
            # 提取用户的分析问题（如果有）
            cmd_length = len(analysis_match.group(1))
            user_query = text[cmd_length:].strip()

            # 检查是否有足够的积分
            if self.enable_points and self.analysis_cost > 0:
                points = await self.db.get_user_points(user_id)
                if points < self.analysis_cost:
                    await bot.send_text_message(chat_id, f"您的积分不足，需要 {self.analysis_cost} 积分才能使用图片分析功能，当前积分: {points}")
                    return False  # 阻断后续插件执行

            # 检查是否有最近的图片
            image_data = await self._get_recent_image(chat_id, user_id)
            if image_data:
                # 扣除积分
                if self.enable_points and self.analysis_cost > 0:
                    points_before = await self.db.get_user_points(user_id)
                    await self.db.update_user_points(user_id, -self.analysis_cost)
                    points_after = await self.db.get_user_points(user_id)

                    # 如果启用了积分消息显示，发送积分消息
                    if self.show_points_message:
                        points_msg = f"已扣除{self.analysis_cost}积分，当前剩余{points_after}积分"
                        await bot.send_text_message(chat_id, points_msg)
                        # 添加短暂延迟
                        await asyncio.sleep(0.5)

                # 保存用户的分析问题
                if user_query:
                    self.waiting_for_analyze_image_query[user_id] = user_query
                    logger.info(f"保存用户分析问题: {user_query}")

                # 处理图片分析请求
                await self._handle_analyze_image(bot, message, image_data)
                return False  # 阻断后续插件执行
            else:
                # 设置等待状态，等待用户上传图片
                self.waiting_for_analyze_image[user_id] = True
                self.waiting_for_analyze_image_time[user_id] = time.time()

                # 保存用户的分析问题
                if user_query:
                    self.waiting_for_analyze_image_query[user_id] = user_query
                    logger.info(f"保存用户分析问题: {user_query}")
                    await bot.send_text_message(chat_id, f"请上传要分析的图片，我将特别关注：{user_query}")
                else:
                    await bot.send_text_message(chat_id, "请上传要分析的图片")

                return False  # 阻断后续插件执行

        # 处理提示词生成命令 - 使用正则表达式来检查命令
        prompt_cmd_pattern = '|'.join(re.escape(cmd) for cmd in self.prompt_enhance_commands)
        prompt_cmd_regex = re.compile(f'^({prompt_cmd_pattern})(\\s|$)')
        prompt_match = prompt_cmd_regex.match(text)

        if prompt_match:
            # 提取提示词
            prompt = text
            for cmd in self.prompt_enhance_commands:
                if text.startswith(cmd):
                    prompt = text[len(cmd):].strip()
                    break

            if not prompt:
                await bot.send_text_message(chat_id, "请提供要增强的提示词")
                return False  # 阻断后续插件执行

            # 发送提示消息
            await bot.send_text_message(chat_id, "正在生成详细提示词，请稍候...")

            # 生成详细提示词
            detailed_prompt = await self._enhance_prompt_direct(prompt, detailed_output=True)
            if detailed_prompt:
                await bot.send_text_message(chat_id, detailed_prompt)
            else:
                await bot.send_text_message(chat_id, "生成提示词失败，请稍后再试")

            return False  # 阻断后续插件执行

        # 使用之前获取的text、chat_id和user_id，保持一致性
        content = text
        from_wxid = chat_id
        sender_wxid = user_id

        # 清理过期的会话
        self._cleanup_expired_conversations()

        # 会话标识 - 已经在前面定义了，这里保持一致
        # conversation_key = f"{from_wxid}_{sender_wxid}"

        # 检查是否是结束对话命令
        if content in self.exit_commands:
            if conversation_key in self.conversations:
                # 清除会话数据
                del self.conversations[conversation_key]
                if conversation_key in self.conversation_timestamps:
                    del self.conversation_timestamps[conversation_key]
                if conversation_key in self.last_images:
                    del self.last_images[conversation_key]

                await bot.send_at_message(from_wxid, "\n已结束Gemini图像生成对话，下次需要时请使用命令重新开始", [sender_wxid])
                return False  # 阻止后续插件执行
            else:
                # 没有活跃会话
                await bot.send_at_message(from_wxid, "\n您当前没有活跃的Gemini图像生成对话", [sender_wxid])
                return False  # 阻止后续插件执行

        # 检查是否是生成图片命令
        for cmd in self.commands:
            if content.startswith(cmd):
                # 提取提示词
                prompt = content[len(cmd):].strip()
                if not prompt:
                    await bot.send_at_message(from_wxid, "\n请提供描述内容，格式：#生成图片 [描述]", [sender_wxid])
                    return False  # 命令格式错误，阻止后续插件执行

                # 检查API密钥是否配置
                if not self.api_key:
                    await bot.send_at_message(from_wxid, "\n请先在配置文件中设置Gemini API密钥", [sender_wxid])
                    return False

                # 检查积分
                if self.enable_points and sender_wxid not in self.admins:
                    points = self.db.get_points(sender_wxid)
                    if points < self.generate_cost:
                        await bot.send_at_message(from_wxid, f"\n您的积分不足，生成图片需要{self.generate_cost}积分，您当前有{points}积分", [sender_wxid])
                        return False  # 积分不足，阻止后续插件执行

                # 生成图片
                try:
                    # 发送处理中消息
                    await bot.send_at_message(from_wxid, "\n正在生成图片，请稍候...", [sender_wxid])

                    # 获取上下文历史
                    conversation_history = self.conversations[conversation_key]

                    # 添加用户提示到会话
                    user_message = {"role": "user", "parts": [{"text": prompt}]}

                    # 调用Gemini API生成图片
                    parts_list, image_count = await self._generate_image(prompt, conversation_history, is_continuous_dialogue=False)

                    if image_count > 0:
                        # 扣除积分
                        if self.enable_points and sender_wxid not in self.admins:
                            self.db.add_points(sender_wxid, -self.generate_cost)
                            points_msg = f"已扣除{self.generate_cost}积分，当前剩余{points - self.generate_cost}积分"
                            # 先发送积分信息（如果启用了积分消息显示）
                            if self.show_points_message:
                                await bot.send_text_message(from_wxid, points_msg)
                                # 添加短暂延迟
                                await asyncio.sleep(0.5)

                        # 保存图片路径列表，用于更新会话历史
                        image_paths = []
                        last_image_path = None

                        # 判断是否是多图文请求
                        is_multi_image = self._is_multi_image_request(prompt)

                        if is_multi_image:
                            # 多图文请求的处理方式
                            logger.info("多图文请求，按照原版程序的方式处理")

                            # 提取所有文本部分（故事内容）
                            story_contents = [part["content"] for part in parts_list if part["type"] == "text"]

                            # 提取所有图片部分
                            image_parts = [part["content"] for part in parts_list if part["type"] == "image"]

                            # 确保我们有故事内容
                            if not story_contents:
                                logger.warning("没有找到故事内容，使用默认文本")
                                story_contents = ["这是一个精彩的故事场景"]

                            # 保存图片到本地并准备发送
                            saved_images = []
                            for i, image_data in enumerate(image_parts):
                                # 保存图片到本地
                                image_path = os.path.join(self.save_dir, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}_{i}.png")
                                with open(image_path, "wb") as f:
                                    f.write(image_data)
                                saved_images.append(image_path)
                                # 保存图片路径
                                image_paths.append(image_path)
                                last_image_path = image_path

                            # 如果没有图片或图片数量少于故事内容，可能需要为每个故事内容单独生成图片
                            if len(saved_images) < len(story_contents):
                                logger.info(f"图片数量({len(saved_images)})少于故事内容({len(story_contents)})，尝试为每个故事内容单独生成图片")

                                # 提取中文提示词
                                chinese_prompts = self._extract_chinese_prompt(prompt)
                                if not chinese_prompts:
                                    logger.warning("没有找到中文提示词，使用故事内容作为提示词")
                                    chinese_prompts = story_contents

                                # 确保中文提示词和故事内容数量一致
                                while len(chinese_prompts) < len(story_contents):
                                    chinese_prompts.append(story_contents[len(chinese_prompts)])

                                # 为每个缺少图片的故事内容单独生成图片
                                for i in range(len(saved_images), len(story_contents)):
                                    if i < len(chinese_prompts):
                                        logger.info(f"为第 {i+1} 个故事内容单独生成图片，提示词: {chinese_prompts[i][:50]}...")

                                        # 单独调用API生成图片
                                        try:
                                            # 构建请求URL
                                            single_url = f"{self.base_url}/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent"
                                            single_headers = {
                                                "Content-Type": "application/json",
                                            }
                                            single_params = {
                                                "key": self.api_key
                                            }

                                            # 构建请求数据
                                            single_data = {
                                                "contents": [
                                                    {
                                                        "parts": [
                                                            {
                                                                "text": chinese_prompts[i]
                                                            }
                                                        ]
                                                    }
                                                ],
                                                "generation_config": {
                                                    "response_modalities": ["Image"],
                                                    "temperature": 0.4,
                                                    "topP": 0.95,
                                                    "topK": 64
                                                }
                                            }

                                            # 创建代理配置
                                            single_proxy = None
                                            if self.enable_proxy and self.proxy_url:
                                                single_proxy = self.proxy_url

                                            # 发送请求
                                            async with aiohttp.ClientSession() as single_session:
                                                async with single_session.post(
                                                    single_url,
                                                    headers=single_headers,
                                                    params=single_params,
                                                    json=single_data,
                                                    proxy=single_proxy,
                                                    timeout=aiohttp.ClientTimeout(total=60)
                                                ) as single_response:
                                                    single_response_text = await single_response.text()

                                                    if single_response.status == 200:
                                                        single_result = json.loads(single_response_text)
                                                        single_candidates = single_result.get("candidates", [])

                                                        if single_candidates and len(single_candidates) > 0:
                                                            single_content = single_candidates[0].get("content", {})
                                                            single_parts = single_content.get("parts", [])

                                                            for single_part in single_parts:
                                                                if "inlineData" in single_part:
                                                                    single_inline_data = single_part.get("inlineData", {})
                                                                    if single_inline_data and "data" in single_inline_data:
                                                                        # 解码图片数据
                                                                        single_image_data = base64.b64decode(single_inline_data["data"])
                                                                        logger.info(f"单独生成图片成功，大小: {len(single_image_data)} 字节")

                                                                        # 保存图片到本地
                                                                        image_path = os.path.join(self.save_dir, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}_{i}.png")
                                                                        with open(image_path, "wb") as f:
                                                                            f.write(single_image_data)
                                                                        saved_images.append(image_path)
                                                                        image_paths.append(image_path)
                                                                        last_image_path = image_path
                                                                        logger.info(f"为第 {i+1} 个故事内容单独生成图片成功")
                                                                        break
                                        except Exception as e:
                                            logger.error(f"单独生成图片失败: {str(e)}")
                                            logger.error(traceback.format_exc())

                            # 按照一一对应的方式发送图片和文本
                            logger.info(f"准备发送 {len(saved_images)} 张图片和 {len(story_contents)} 段文本")

                            # 确定要发送的对数
                            pairs_count = min(len(saved_images), len(story_contents))

                            # 一一对应发送图片和文本，确保每次发送完成后再发送下一条
                            for i in range(pairs_count):
                                # 先发送文本
                                if i < len(story_contents) and story_contents[i].strip():
                                    await bot.send_text_message(from_wxid, story_contents[i])
                                    # 添加短暂延迟，确保文本发送完成
                                    await asyncio.sleep(0.5)

                                # 再发送图片
                                if i < len(saved_images):
                                    with open(saved_images[i], "rb") as f:
                                        await bot.send_image_message(from_wxid, f.read())
                                    # 添加延迟，确保图片发送完成
                                    await asyncio.sleep(1.5)

                            # 如果还有剩余的文本，发送剩余文本
                            for i in range(pairs_count, len(story_contents)):
                                if story_contents[i].strip():
                                    await bot.send_text_message(from_wxid, story_contents[i])
                                    # 添加短暂延迟
                                    await asyncio.sleep(0.5)

                            # 如果还有剩余的图片，发送剩余图片
                            for i in range(pairs_count, len(saved_images)):
                                with open(saved_images[i], "rb") as f:
                                    await bot.send_image_message(from_wxid, f.read())
                                # 添加延迟
                                await asyncio.sleep(1.5)
                        else:
                            # 常规请求的处理方式
                            # 按照原始顺序发送文本和图片
                            current_text = ""

                            for part in parts_list:
                                if part["type"] == "text":
                                    # 累积文本，直到遇到图片才发送
                                    current_text += part["content"]
                                elif part["type"] == "image":
                                    # 如果有累积的文本，先发送文本
                                    if current_text.strip():
                                        await bot.send_text_message(from_wxid, current_text)
                                        current_text = ""
                                        # 添加短暂延迟，确保文本发送完成
                                        await asyncio.sleep(0.5)

                                    # 保存图片到本地
                                    image_path = os.path.join(self.save_dir, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                                    with open(image_path, "wb") as f:
                                        f.write(part["content"])

                                    # 发送图片
                                    with open(image_path, "rb") as f:
                                        await bot.send_image_message(from_wxid, f.read())
                                    # 添加延迟，确保图片发送完成
                                    await asyncio.sleep(1.5)

                                    # 保存图片路径
                                    image_paths.append(image_path)
                                    last_image_path = image_path

                            # 发送剩余的文本（如果有）
                            if current_text.strip():
                                await bot.send_text_message(from_wxid, current_text)
                                # 添加短暂延迟
                                await asyncio.sleep(0.5)

                        # 保存最后生成的图片路径（用于后续编辑）
                        if last_image_path:
                            self.last_images[conversation_key] = last_image_path

                        # 不再发送对话提示
                        # if not conversation_history:  # 如果是新会话
                        #     await bot.send_text_message(from_wxid, f"已开始图像对话，可以直接发消息继续修改图片。需要结束时请发送\"{self.exit_commands[0]}\"")

                        # 更新会话历史
                        conversation_history.append(user_message)

                        # 创建助手消息部分
                        assistant_parts = []

                        # 按照原始顺序添加文本和图片
                        image_index = 0
                        for part in parts_list:
                            if part["type"] == "text":
                                assistant_parts.append({"text": part["content"]})
                            elif part["type"] == "image" and image_index < len(image_paths):
                                assistant_parts.append({"image_url": image_paths[image_index]})
                                image_index += 1

                        # 如果没有文本，添加默认文本
                        if not any("text" in p for p in assistant_parts):
                            assistant_parts.insert(0, {"text": "我已基于您的提示生成了图片"})

                        assistant_message = {
                            "role": "model",
                            "parts": assistant_parts
                        }
                        conversation_history.append(assistant_message)

                        # 限制会话历史长度
                        if len(conversation_history) > 10:  # 保留最近5轮对话
                            conversation_history = conversation_history[-10:]

                        # 更新会话时间戳
                        self.conversation_timestamps[conversation_key] = time.time()
                    else:
                        # 检查是否有文本响应，可能是内容被拒绝
                        # 尝试从 parts_list 中提取文本响应
                        text_parts = [part["content"] for part in parts_list if part["type"] == "text"]

                        if text_parts:
                            # 内容审核拒绝的情况，翻译并转发拒绝消息给用户
                            translated_response = self._translate_gemini_message(text_parts[0])
                            await bot.send_at_message(from_wxid, f"\n{translated_response}", [sender_wxid])
                            logger.warning(f"API拒绝生成图片，提示: {text_parts[0]}")
                        else:
                            logger.error(f"生成图片失败，未获取到有效的图片数据")
                            await bot.send_at_message(from_wxid, "\n图片生成失败，请稍后再试或修改提示词", [sender_wxid])
                except Exception as e:
                    logger.error(f"生成图片失败: {str(e)}")
                    logger.error(traceback.format_exc())
                    await bot.send_at_message(from_wxid, f"\n生成图片失败: {str(e)}", [sender_wxid])
                return False  # 已处理命令，阻止后续插件执行

        # 检查是否是编辑图片命令（针对已保存的图片）
        for cmd in self.edit_commands:
            if content.startswith(cmd):
                # 提取提示词
                prompt = content[len(cmd):].strip()
                if not prompt:
                    await bot.send_at_message(from_wxid, "\n请提供编辑描述，格式：#编辑图片 [描述]", [sender_wxid])
                    return False  # 命令格式错误，阻止后续插件执行

                # 检查API密钥是否配置
                if not self.api_key:
                    await bot.send_at_message(from_wxid, "\n请先在配置文件中设置Gemini API密钥", [sender_wxid])
                    return False

                # 先尝试从缓存获取最近的图片
                image_data = await self._get_recent_image(from_wxid, sender_wxid)
                if image_data:
                    # 如果找到缓存的图片，保存到本地再处理
                    image_path = os.path.join(self.save_dir, f"temp_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                    self.last_images[conversation_key] = image_path
                    logger.info(f"找到最近缓存的图片，保存到：{image_path}")

                    # 再次保存到缓存，确保使用了所有可能的键格式
                    self._save_image_to_cache(from_wxid, sender_wxid, image_data)

                # 检查是否有上一次上传/生成的图片
                last_image_path = self.last_images.get(conversation_key)
                if not last_image_path or not os.path.exists(last_image_path):
                    # 再次尝试查找图片，使用更宽松的条件
                    logger.info("未找到图片路径，尝试使用更宽松的条件再次查找")
                    for key, value in self.last_images.items():
                        # 只有当会话活跃时才使用宽松条件查找图片
                        if key in self.conversations and (from_wxid in key or sender_wxid in key) and os.path.exists(value):
                            last_image_path = value
                            logger.info(f"使用宽松条件找到图片路径: {last_image_path}, 键: {key}")
                            break

                    if not last_image_path or not os.path.exists(last_image_path):
                        await bot.send_at_message(from_wxid, "\n未找到可编辑的图片，请先上传一张图片", [sender_wxid])
                        return False

                # 检查积分
                if self.enable_points and sender_wxid not in self.admins:
                    points = self.db.get_points(sender_wxid)
                    if points < self.edit_cost:
                        await bot.send_at_message(from_wxid, f"\n您的积分不足，编辑图片需要{self.edit_cost}积分，您当前有{points}积分", [sender_wxid])
                        return False  # 积分不足，阻止后续插件执行

                # 编辑图片
                try:
                    # 发送处理中消息
                    await bot.send_at_message(from_wxid, "\n正在编辑图片，请稍候...", [sender_wxid])

                    # 读取上一次的图片
                    with open(last_image_path, "rb") as f:
                        image_data = f.read()

                    # 获取会话上下文
                    conversation_history = self.conversations[conversation_key]

                    # 调用Gemini API编辑图片
                    edited_images, text_responses = await self._edit_image(prompt, image_data, conversation_history)

                    # 确保 edited_images 和 text_responses 不为 None
                    if edited_images is None:
                        edited_images = []
                    if text_responses is None:
                        text_responses = []

                    if len(edited_images) > 0 and edited_images[0]:
                        # 保存编辑后的图片
                        edited_image_path = os.path.join(self.save_dir, f"edited_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                        logger.info(f"保存编辑后的图片到: {edited_image_path}, 数据大小: {len(edited_images[0])} 字节")
                        # 检查图片数据是否有效
                        if edited_images[0][:8].hex().startswith('89504e47') or edited_images[0][:3].hex().startswith('ffd8ff'):
                            logger.info(f"图片数据是有效的PNG或JPEG格式")
                        else:
                            logger.warning(f"图片数据不是标准的PNG或JPEG格式")
                        with open(edited_image_path, "wb") as f:
                            f.write(edited_images[0])

                        # 更新最后生成的图片路径
                        self.last_images[conversation_key] = edited_image_path

                        # 扣除积分
                        if self.enable_points and sender_wxid not in self.admins:
                            self.db.add_points(sender_wxid, -self.edit_cost)
                            points_msg = f"已扣除{self.edit_cost}积分，当前剩余{points - self.edit_cost}积分"
                        else:
                            points_msg = ""

                        # 如果不显示积分消息，清空积分消息
                        if not self.show_points_message:
                            points_msg = ""

                        # 发送文本回复（如果有）
                        first_valid_text = next((t for t in text_responses if t), None)
                        if first_valid_text:
                            # 清理文本，去除多余的空格和换行
                            cleaned_text = first_valid_text.strip()
                            # 将多个连续空格替换为单个空格
                            import re
                            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                            # 移除文本开头和结尾的引号
                            if cleaned_text.startswith('"') and cleaned_text.endswith('"'):
                                cleaned_text = cleaned_text[1:-1]
                            # 构建消息文本，避免在没有积分消息时添加多余的换行
                            if points_msg:
                                message_text = f"{cleaned_text}\n\n{points_msg}"
                            else:
                                message_text = cleaned_text
                            await bot.send_text_message(from_wxid, message_text)
                        else:
                            await bot.send_text_message(from_wxid, f"图片编辑成功！{points_msg if points_msg else ''}")
                        # 添加短暂延迟，确保文本发送完成
                        await asyncio.sleep(0.5)

                        # 发送图片
                        logger.info(f"准备发送编辑后的图片: {edited_image_path}")
                        try:
                            with open(edited_image_path, "rb") as f:
                                image_data = f.read()
                                # 检查图片数据是否有效
                                if image_data[:8].hex().startswith('89504e47') or image_data[:3].hex().startswith('ffd8ff'):
                                    logger.info(f"读取的图片数据是有效的PNG或JPEG格式")
                                else:
                                    logger.warning(f"读取的图片数据不是标准的PNG或JPEG格式")
                                await bot.send_image_message(from_wxid, image_data)
                                # 添加延迟，确保图片发送完成
                                await asyncio.sleep(1.5)
                        except Exception as e:
                            logger.error(f"发送图片失败: {str(e)}")
                            logger.error(traceback.format_exc())

                        # 不再发送对话提示
                        # if not conversation_history:  # 如果是新会话
                        #     await bot.send_text_message(from_wxid, f"已开始图像对话，可以直接发消息继续修改图片。需要结束时请发送\"{self.exit_commands[0]}\"")

                        # 更新会话历史
                        user_message = {
                            "role": "user",
                            "parts": [
                                {"text": prompt},
                                {"image_url": last_image_path}
                            ]
                        }
                        conversation_history.append(user_message)

                        assistant_message = {
                            "role": "model",
                            "parts": [
                                {"text": first_valid_text if first_valid_text else "我已编辑完成图片"},
                                {"image_url": edited_image_path}
                            ]
                        }
                        conversation_history.append(assistant_message)

                        # 限制会话历史长度
                        if len(conversation_history) > 10:  # 保留最近5轮对话
                            conversation_history = conversation_history[-10:]

                        # 更新会话时间戳
                        self.conversation_timestamps[conversation_key] = time.time()
                    else:
                        # 检查是否有文本响应，可能是内容被拒绝
                        first_valid_text = next((t for t in text_responses if t), None)
                        if first_valid_text:
                            # 内容审核拒绝的情况，翻译并转发拒绝消息给用户
                            # 检查是否是JSON格式的错误信息
                            try:
                                error_data = json.loads(first_valid_text)
                                # 构建友好的错误消息
                                error_message = "图片编辑请求被拒绝。"

                                # 尝试从错误数据中提取有用信息
                                if "candidates" in error_data and error_data["candidates"]:
                                    candidate = error_data["candidates"][0]
                                    if "finishReason" in candidate:
                                        error_message += f"原因: {candidate['finishReason']}. "

                                    if "safetyRatings" in candidate:
                                        blocked_categories = []
                                        for rating in candidate["safetyRatings"]:
                                            if rating.get("blocked", False):
                                                category = rating.get("category", "未知类别")
                                                probability = rating.get("probability", "未知")
                                                blocked_categories.append(f"{category}({probability})")

                                        if blocked_categories:
                                            error_message += f"被拒绝的类别: {', '.join(blocked_categories)}。"

                                error_message += "请修改您的请求。"
                                translated_response = error_message
                            except json.JSONDecodeError:
                                # 不是JSON格式，使用常规翻译
                                translated_response = self._translate_gemini_message(first_valid_text)

                            await bot.send_at_message(from_wxid, f"\n{translated_response}", [sender_wxid])
                            logger.warning(f"API拒绝编辑图片，提示: {first_valid_text}")
                        else:
                            logger.error(f"编辑图片失败，未获取到有效的图片数据")
                            await bot.send_at_message(from_wxid, "\n图片编辑失败，请稍后再试或修改描述", [sender_wxid])
                except Exception as e:
                    logger.error(f"编辑图片失败: {str(e)}")
                    logger.error(traceback.format_exc())
                    await bot.send_at_message(from_wxid, f"\n编辑图片失败: {str(e)}", [sender_wxid])
                return False  # 已处理命令，阻止后续插件执行

        # 这部分代码已经在前面处理过了，不需要重复处理
        # 如果前面没有检测到前缀，这里也不会检测到
        # 如果前面检测到了前缀，这里就不需要再处理一次
        # 因此，这段代码可以删除

        # 如果没有检测到前缀，但有活跃会话，检查是否需要前缀
        if conversation_key in self.conversations and content and not any(content.startswith(cmd) for cmd in self.commands + self.edit_commands):
            # 如果需要前缀但没有找到前缀，则不处理这条消息
            if self.require_prefix_for_conversation:
                logger.info(f"消息 '{content}' 没有包含所需前缀，不处理为连续对话")
                return True

            # 在群聊中，检查是否包含唤醒词或@机器人
            if message.get("IsGroup", False):
                # 在群聊中必须包含唤醒词或@机器人才能继续对话
                if not self.has_wake_word(content) and not self.is_at_message(message):
                    # 没有唤醒词，不处理
                    return True

                # 清理消息内容，移除唤醒词和@标记
                clean_content = self.get_clean_content(content)
                if not clean_content.strip():
                    # 清理后内容为空，不处理
                    return True

                # 使用清理后的内容继续对话
                content = clean_content

            # 修改原始消息内容，确保后续处理使用处理后的内容
            message["Content"] = content

            # 有活跃会话，且包含必要的唤醒词，视为继续对话
            try:
                logger.info(f"继续对话: 用户={sender_wxid}, 内容='{content}'")

                # 检查积分
                if self.enable_points and sender_wxid not in self.admins:
                    points = self.db.get_points(sender_wxid)
                    if points < self.generate_cost:
                        await bot.send_at_message(from_wxid, f"\n您的积分不足，生成图片需要{self.generate_cost}积分，您当前有{points}积分", [sender_wxid])
                        return False  # 积分不足，阻止后续插件执行

                # 发送处理中消息
                await bot.send_at_message(from_wxid, "\n正在处理您的请求，请稍候...", [sender_wxid])

                # 获取上下文历史
                conversation_history = self.conversations[conversation_key]
                logger.info(f"对话历史长度: {len(conversation_history)}")

                # 添加用户提示到会话
                user_message = {"role": "user", "parts": [{"text": content}]}

                # 检查是否有上一次生成的图片，如果有则自动作为输入
                last_image_path = self.last_images.get(conversation_key)
                logger.info(f"上一次图片路径: {last_image_path}")

                # 如果没有找到图片路径，尝试从缓存获取
                if not last_image_path or not os.path.exists(last_image_path):
                    logger.info("未找到上一次图片路径，尝试从缓存获取")
                    image_data = await self._get_recent_image(from_wxid, sender_wxid)
                    if image_data:
                        # 如果找到缓存的图片，保存到本地再处理
                        image_path = os.path.join(self.save_dir, f"temp_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                        with open(image_path, "wb") as f:
                            f.write(image_data)
                        self.last_images[conversation_key] = image_path
                        last_image_path = image_path
                        logger.info(f"从缓存找到图片，保存到：{image_path}")

                        # 再次保存到缓存，确保使用了所有可能的键格式
                        self._save_image_to_cache(from_wxid, sender_wxid, image_data)
                    else:
                        # 尝试使用更宽松的条件查找图片路径
                        logger.info("未找到缓存图片，尝试使用更宽松的条件查找图片路径")
                        for key, value in self.last_images.items():
                            # 只有当会话活跃时才使用宽松条件查找图片
                            if key in self.conversations and (from_wxid in key or sender_wxid in key) and os.path.exists(value):
                                last_image_path = value
                                logger.info(f"使用宽松条件找到图片路径: {last_image_path}, 键: {key}")
                                break

                if last_image_path and os.path.exists(last_image_path):
                    logger.info(f"找到上一次图片，将使用该图片进行编辑")
                    # 读取上一次生成的图片
                    with open(last_image_path, "rb") as f:
                        image_data = f.read()

                    # 调用编辑图片API
                    logger.info(f"调用编辑图片API")
                    # 在连续对话模式下，设置is_continuous_dialogue为True
                    # 添加中文提示，确保返回中文结果
                    content_with_lang = f"请用中文回答：{content}"
                    edited_images, text_responses = await self._edit_image(content_with_lang, image_data, conversation_history, is_continuous_dialogue=True)

                    # 确保 edited_images 和 text_responses 不为 None
                    if edited_images is None:
                        edited_images = []
                    if text_responses is None:
                        text_responses = []

                    if len(edited_images) > 0 and edited_images[0]:
                        logger.info(f"成功获取编辑后的图片结果")
                        # 保存编辑后的图片
                        new_image_path = os.path.join(self.save_dir, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                        with open(new_image_path, "wb") as f:
                            f.write(edited_images[0])

                        # 更新最后生成的图片路径
                        self.last_images[conversation_key] = new_image_path

                        # 扣除积分
                        if self.enable_points and sender_wxid not in self.admins:
                            self.db.add_points(sender_wxid, -self.edit_cost)  # 使用编辑积分
                            points_msg = f"已扣除{self.edit_cost}积分，当前剩余{points - self.edit_cost}积分"
                        else:
                            points_msg = ""

                        # 发送文本回复（如果有）
                        first_valid_text = next((t for t in text_responses if t), None)
                        if first_valid_text:
                            # 清理文本，去除多余的空格和换行
                            cleaned_text = first_valid_text.strip()
                            # 将多个连续空格替换为单个空格
                            import re
                            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                            # 移除文本开头和结尾的引号
                            if cleaned_text.startswith('"') and cleaned_text.endswith('"'):
                                cleaned_text = cleaned_text[1:-1]
                            # 构建消息文本，避免在没有积分消息时添加多余的换行
                            if points_msg:
                                message_text = f"{cleaned_text}\n\n{points_msg}"
                            else:
                                message_text = cleaned_text
                            await bot.send_text_message(from_wxid, message_text)
                        else:
                            await bot.send_text_message(from_wxid, f"图片编辑成功！{points_msg if points_msg else ''}")
                        # 添加短暂延迟，确保文本发送完成
                        await asyncio.sleep(0.5)

                        # 发送图片
                        logger.info(f"发送编辑后的图片")
                        with open(new_image_path, "rb") as f:
                            await bot.send_image_message(from_wxid, f.read())
                        # 添加延迟，确保图片发送完成
                        await asyncio.sleep(1.5)

                        # 更新会话历史
                        # 添加包含图片的用户消息
                        user_message = {
                            "role": "user",
                            "parts": [
                                {"text": content},
                                {"image_url": last_image_path}
                            ]
                        }
                        conversation_history.append(user_message)

                        assistant_message = {
                            "role": "model",
                            "parts": [
                                {"text": first_valid_text if first_valid_text else "我已编辑了图片"},
                                {"image_url": new_image_path}
                            ]
                        }
                        conversation_history.append(assistant_message)

                        # 限制会话历史长度
                        if len(conversation_history) > 10:
                            conversation_history = conversation_history[-10:]

                        # 更新会话时间戳
                        self.conversation_timestamps[conversation_key] = time.time()

                        return False  # 已处理命令，阻止后续插件执行
                    else:
                        # 检查是否有文本响应，可能是内容被拒绝
                        first_valid_text = next((t for t in text_responses if t), None)
                        if first_valid_text:
                            # 内容审核拒绝的情况，翻译并转发拒绝消息给用户
                            # 检查是否是JSON格式的错误信息
                            try:
                                error_data = json.loads(first_valid_text)
                                # 构建友好的错误消息
                                error_message = "图片编辑请求被拒绝。"

                                # 尝试从错误数据中提取有用信息
                                if "candidates" in error_data and error_data["candidates"]:
                                    candidate = error_data["candidates"][0]
                                    if "finishReason" in candidate:
                                        error_message += f"原因: {candidate['finishReason']}. "

                                    if "safetyRatings" in candidate:
                                        blocked_categories = []
                                        for rating in candidate["safetyRatings"]:
                                            if rating.get("blocked", False):
                                                category = rating.get("category", "未知类别")
                                                probability = rating.get("probability", "未知")
                                                blocked_categories.append(f"{category}({probability})")

                                        if blocked_categories:
                                            error_message += f"被拒绝的类别: {', '.join(blocked_categories)}。"

                                error_message += "请修改您的请求。"
                                translated_response = error_message
                            except json.JSONDecodeError:
                                # 不是JSON格式，使用常规翻译
                                translated_response = self._translate_gemini_message(first_valid_text)

                            await bot.send_at_message(from_wxid, f"\n{translated_response}", [sender_wxid])
                            logger.warning(f"API拒绝编辑图片，提示: {first_valid_text}")
                        else:
                            logger.error(f"编辑图片失败，未获取到有效的图片数据")
                            await bot.send_at_message(from_wxid, "\n图片编辑失败，请稍后再试或修改描述", [sender_wxid])
                else:
                    logger.info(f"没有找到上一次图片或文件不存在，将生成新图片")
                    # 没有上一次图片，当作生成新图片处理
                    parts_list, image_count = await self._generate_image(content, conversation_history)

                    if image_count > 0:
                        logger.info(f"成功获取生成的图片结果")

                        # 扣除积分
                        if self.enable_points and sender_wxid not in self.admins:
                            self.db.add_points(sender_wxid, -self.generate_cost)
                            points_msg = f"已扣除{self.generate_cost}积分，当前剩余{points - self.generate_cost}积分"
                            # 先发送积分信息
                            await bot.send_text_message(from_wxid, points_msg)
                            # 添加短暂延迟
                            await asyncio.sleep(0.5)

                        # 保存图片路径列表，用于更新会话历史
                        image_paths = []
                        last_image_path = None

                        # 判断是否是多图文请求
                        is_multi_image = self._is_multi_image_request(content)

                        if is_multi_image:
                            # 多图文请求的处理方式
                            logger.info("多图文请求，按照原版程序的方式处理")

                            # 提取所有文本部分（故事内容）
                            story_contents = [part["content"] for part in parts_list if part["type"] == "text"]

                            # 提取所有图片部分
                            image_parts = [part["content"] for part in parts_list if part["type"] == "image"]

                            # 确保我们有故事内容
                            if not story_contents:
                                logger.warning("没有找到故事内容，使用默认文本")
                                story_contents = ["这是一个精彩的故事场景"]

                            # 保存图片到本地并准备发送
                            saved_images = []
                            for i, image_data in enumerate(image_parts):
                                # 保存图片到本地
                                image_path = os.path.join(self.save_dir, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}_{i}.png")
                                with open(image_path, "wb") as f:
                                    f.write(image_data)
                                saved_images.append(image_path)
                                # 保存图片路径
                                image_paths.append(image_path)
                                last_image_path = image_path

                            # 如果没有图片或图片数量少于故事内容，可能需要为每个故事内容单独生成图片
                            if len(saved_images) < len(story_contents):
                                logger.info(f"图片数量({len(saved_images)})少于故事内容({len(story_contents)})，尝试为每个故事内容单独生成图片")

                                # 提取中文提示词
                                chinese_prompts = self._extract_chinese_prompt(prompt)
                                if not chinese_prompts:
                                    logger.warning("没有找到中文提示词，使用故事内容作为提示词")
                                    chinese_prompts = story_contents

                                # 确保中文提示词和故事内容数量一致
                                while len(chinese_prompts) < len(story_contents):
                                    chinese_prompts.append(story_contents[len(chinese_prompts)])

                                # 为每个缺少图片的故事内容单独生成图片
                                for i in range(len(saved_images), len(story_contents)):
                                    if i < len(chinese_prompts):
                                        logger.info(f"为第 {i+1} 个故事内容单独生成图片，提示词: {chinese_prompts[i][:50]}...")

                                        # 单独调用API生成图片
                                        try:
                                            # 构建请求URL
                                            single_url = f"{self.base_url}/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent"
                                            # 检查URL格式是否正确
                                            if not single_url.startswith("http"):
                                                logger.warning(f"URL格式可能不正确: {single_url}")
                                                # 尝试修复URL格式
                                                single_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent"
                                            single_headers = {
                                                "Content-Type": "application/json",
                                            }
                                            single_params = {
                                                "key": self.api_key
                                            }

                                            # 构建请求数据
                                            single_data = {
                                                "contents": [
                                                    {
                                                        "role": "user",
                                                        "parts": [
                                                            {
                                                                "text": chinese_prompts[i]
                                                            }
                                                        ]
                                                    }
                                                ],
                                                "generation_config": {
                                                    "response_modalities": ["Image"],
                                                    "temperature": 0.4,
                                                    "topP": 0.95,
                                                    "topK": 64
                                                }
                                            }

                                            # 创建代理配置
                                            single_proxy = None
                                            if self.enable_proxy and self.proxy_url:
                                                single_proxy = self.proxy_url

                                            # 发送请求
                                            async with aiohttp.ClientSession() as single_session:
                                                async with single_session.post(
                                                    single_url,
                                                    headers=single_headers,
                                                    params=single_params,
                                                    json=single_data,
                                                    proxy=single_proxy,
                                                    timeout=aiohttp.ClientTimeout(total=60)
                                                ) as single_response:
                                                    single_response_text = await single_response.text()

                                                    if single_response.status == 200:
                                                        single_result = json.loads(single_response_text)
                                                        single_candidates = single_result.get("candidates", [])

                                                        if single_candidates and len(single_candidates) > 0:
                                                            single_content = single_candidates[0].get("content", {})
                                                            single_parts = single_content.get("parts", [])

                                                            for single_part in single_parts:
                                                                if "inlineData" in single_part:
                                                                    single_inline_data = single_part.get("inlineData", {})
                                                                    if single_inline_data and "data" in single_inline_data:
                                                                        # 返回Base64解码后的图片数据
                                                                        single_image_data = base64.b64decode(single_inline_data["data"])

                                                                        # 保存图片到本地
                                                                        image_path = os.path.join(self.save_dir, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}_{i}.png")
                                                                        with open(image_path, "wb") as f:
                                                                            f.write(single_image_data)
                                                                        saved_images.append(image_path)
                                                                        image_paths.append(image_path)
                                                                        last_image_path = image_path
                                                                        logger.info(f"为第 {i+1} 个故事内容单独生成图片成功")
                                                                        break
                                        except Exception as e:
                                            logger.error(f"单独生成图片失败: {str(e)}")
                                            logger.error(traceback.format_exc())

                            # 按照一一对应的方式发送图片和文本
                            logger.info(f"准备发送 {len(saved_images)} 张图片和 {len(story_contents)} 段文本")

                            # 确定要发送的对数
                            pairs_count = min(len(saved_images), len(story_contents))

                            # 一一对应发送图片和文本
                            for i in range(pairs_count):
                                # 先发送文本
                                if i < len(story_contents) and story_contents[i].strip():
                                    await bot.send_text_message(from_wxid, story_contents[i])

                                # 再发送图片
                                if i < len(saved_images):
                                    with open(saved_images[i], "rb") as f:
                                        await bot.send_image_message(from_wxid, f.read())

                            # 如果还有剩余的文本，发送剩余文本
                            for i in range(pairs_count, len(story_contents)):
                                if story_contents[i].strip():
                                    await bot.send_text_message(from_wxid, story_contents[i])

                            # 如果还有剩余的图片，发送剩余图片
                            for i in range(pairs_count, len(saved_images)):
                                with open(saved_images[i], "rb") as f:
                                    await bot.send_image_message(from_wxid, f.read())
                        else:
                            # 常规请求的处理方式
                            # 按照原始顺序发送文本和图片
                            current_text = ""

                            for part in parts_list:
                                if part["type"] == "text":
                                    # 累积文本，直到遇到图片才发送
                                    current_text += part["content"]
                                elif part["type"] == "image":
                                    # 如果有累积的文本，先发送文本
                                    if current_text.strip():
                                        await bot.send_text_message(from_wxid, current_text)
                                        current_text = ""

                                    # 保存图片到本地
                                    image_path = os.path.join(self.save_dir, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                                    with open(image_path, "wb") as f:
                                        f.write(part["content"])

                                    # 发送图片
                                    with open(image_path, "rb") as f:
                                        await bot.send_image_message(from_wxid, f.read())

                                    # 保存图片路径
                                    image_paths.append(image_path)
                                    last_image_path = image_path

                            # 发送剩余的文本（如果有）
                            if current_text.strip():
                                await bot.send_text_message(from_wxid, current_text)

                        # 保存最后生成的图片路径（用于后续编辑）
                        if last_image_path:
                            self.last_images[conversation_key] = last_image_path

                        logger.info(f"发送生成的图片完成")

                        # 更新会话历史
                        conversation_history.append(user_message)

                        # 创建助手消息部分
                        assistant_parts = []

                        # 按照原始顺序添加文本和图片
                        image_index = 0
                        for part in parts_list:
                            if part["type"] == "text":
                                assistant_parts.append({"text": part["content"]})
                            elif part["type"] == "image" and image_index < len(image_paths):
                                assistant_parts.append({"image_url": image_paths[image_index]})
                                image_index += 1

                        # 如果没有文本，添加默认文本
                        if not any("text" in p for p in assistant_parts):
                            assistant_parts.insert(0, {"text": "我已基于您的提示生成了图片"})

                        assistant_message = {
                            "role": "model",
                            "parts": assistant_parts
                        }
                        conversation_history.append(assistant_message)

                        # 限制会话历史长度
                        if len(conversation_history) > 10:
                            conversation_history = conversation_history[-10:]

                        # 更新会话时间戳
                        self.conversation_timestamps[conversation_key] = time.time()
                    else:
                        # 检查是否有文本响应，可能是内容被拒绝
                        # 尝试从 parts_list 中提取文本响应
                        text_parts = [part["content"] for part in parts_list if part["type"] == "text"]

                        if text_parts:
                            # 内容审核拒绝的情况，翻译并转发拒绝消息给用户
                            translated_response = self._translate_gemini_message(text_parts[0])
                            await bot.send_at_message(from_wxid, f"\n{translated_response}", [sender_wxid])
                            logger.warning(f"API拒绝生成图片，提示: {text_parts[0]}")
                        else:
                            logger.error(f"生成图片失败，未获取到有效的图片数据")
                            await bot.send_at_message(from_wxid, "\n图片生成失败，请稍后再试或修改提示词", [sender_wxid])
                return False
            except Exception as e:
                logger.error(f"对话继续生成图片失败: {str(e)}")
                logger.error(traceback.format_exc())
                await bot.send_at_message(from_wxid, f"\n生成失败: {str(e)}", [sender_wxid])
                return False  # 已处理命令，阻止后续插件执行

        # 不是本插件的命令，继续执行后续插件
        return True

    @on_file_message(priority=30)
    async def handle_edit_image(self, bot: WechatAPIClient, message: dict) -> bool:
        """处理编辑图片的命令"""
        if not self.enable:
            return True  # 插件未启用，继续执行后续插件

        from_wxid = message.get("FromWxid", "")
        sender_wxid = message.get("SenderWxid", "")
        file_info = message.get("FileInfo", {})

        # 清理过期的会话
        self._cleanup_expired_conversations()

        # 会话标识
        conversation_key = f"{from_wxid}_{sender_wxid}"

        # 检查消息是否含有文件信息
        if not file_info or "FileID" not in file_info:
            return True  # 不是有效的文件消息，继续执行后续插件

        # 检查是否是图片编辑命令
        if "FileSummary" in file_info:
            summary = file_info.get("FileSummary", "").strip()

            for cmd in self.edit_commands:
                if summary.startswith(cmd):
                    # 提取提示词
                    prompt = summary[len(cmd):].strip()
                    if not prompt:
                        await bot.send_at_message(from_wxid, "\n请提供编辑描述，格式：#编辑图片 [描述]", [sender_wxid])
                        return False  # 命令格式错误，阻止后续插件执行

                    # 检查API密钥是否配置
                    if not self.api_key:
                        await bot.send_at_message(from_wxid, "\n请先在配置文件中设置Gemini API密钥", [sender_wxid])
                        return False

                    # 检查文件类型是否为图片
                    file_name = file_info.get("FileName", "").lower()
                    valid_extensions = [".jpg", ".jpeg", ".png", ".webp"]
                    is_image = any(file_name.endswith(ext) for ext in valid_extensions)

                    if not is_image:
                        await bot.send_at_message(from_wxid, "\n请上传图片文件（支持JPG、PNG、WEBP格式）", [sender_wxid])
                        return False

                    # 检查积分
                    if self.enable_points and sender_wxid not in self.admins:
                        points = self.db.get_points(sender_wxid)
                        if points < self.edit_cost:
                            await bot.send_at_message(from_wxid, f"\n您的积分不足，编辑图片需要{self.edit_cost}积分，您当前有{points}积分", [sender_wxid])
                            return False  # 积分不足，阻止后续插件执行

                    # 编辑图片
                    try:
                        # 发送处理中消息
                        await bot.send_at_message(from_wxid, "\n正在编辑图片，请稍候...", [sender_wxid])

                        # 下载用户上传的图片
                        file_id = file_info.get("FileID")
                        file_content = await bot.download_file(file_id)

                        # 保存原始图片
                        orig_image_path = os.path.join(self.save_dir, f"orig_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                        with open(orig_image_path, "wb") as f:
                            f.write(file_content)

                        # 保存到图片缓存
                        self._save_image_to_cache(from_wxid, sender_wxid, file_content)
                        logger.info(f"保存上传的文件到图片缓存，大小: {len(file_content)} 字节")

                        # 获取会话上下文
                        conversation_history = self.conversations[conversation_key]

                        # 调用Gemini API编辑图片
                        edited_images, text_responses = await self._edit_image(prompt, file_content, conversation_history)

                        # 确保 edited_images 和 text_responses 不为 None
                        if edited_images is None:
                            edited_images = []
                        if text_responses is None:
                            text_responses = []

                        if len(edited_images) > 0 and edited_images[0]:
                            # 保存编辑后的图片
                            edited_image_path = os.path.join(self.save_dir, f"edited_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                            with open(edited_image_path, "wb") as f:
                                f.write(edited_images[0])

                            # 更新最后生成的图片路径
                            self.last_images[conversation_key] = edited_image_path

                            # 扣除积分
                            if self.enable_points and sender_wxid not in self.admins:
                                self.db.add_points(sender_wxid, -self.edit_cost)
                                points_msg = f"已扣除{self.edit_cost}积分，当前剩余{points - self.edit_cost}积分"
                            else:
                                points_msg = ""

                            # 发送文本回复（如果有）
                            first_valid_text = next((t for t in text_responses if t), None)
                            if first_valid_text:
                                await bot.send_text_message(from_wxid, f"{first_valid_text}\n\n{points_msg if points_msg else ''}")
                            else:
                                await bot.send_text_message(from_wxid, f"图片编辑成功！{points_msg if points_msg else ''}")
                            # 添加短暂延迟，确保文本发送完成
                            await asyncio.sleep(0.5)

                            # 发送图片
                            with open(edited_image_path, "rb") as f:
                                await bot.send_image_message(from_wxid, f.read())
                            # 添加延迟，确保图片发送完成
                            await asyncio.sleep(1.5)

                            # 不再发送对话提示
                            # if not conversation_history:  # 如果是新会话
                            #     await bot.send_text_message(from_wxid, f"已开始图像对话，可以直接发消息继续修改图片。需要结束时请发送\"{self.exit_commands[0]}\"")

                            # 更新会话历史
                            user_message = {
                                "role": "user",
                                "parts": [
                                    {"text": prompt},
                                    {"image_url": orig_image_path}
                                ]
                            }
                            conversation_history.append(user_message)

                            assistant_message = {
                                "role": "model",
                                "parts": [
                                    {"text": first_valid_text if first_valid_text else "我已编辑完成图片"},
                                    {"image_url": edited_image_path}
                                ]
                            }
                            conversation_history.append(assistant_message)

                            # 限制会话历史长度
                            if len(conversation_history) > 10:  # 保留最近5轮对话
                                conversation_history = conversation_history[-10:]

                            # 更新会话时间戳
                            self.conversation_timestamps[conversation_key] = time.time()
                        else:
                            # 检查是否有文本响应，可能是内容被拒绝
                            first_valid_text = next((t for t in text_responses if t), None)
                            if first_valid_text:
                                # 内容审核拒绝的情况，翻译并转发拒绝消息给用户
                                translated_response = self._translate_gemini_message(first_valid_text)
                                await bot.send_at_message(from_wxid, f"\n{translated_response}", [sender_wxid])
                                logger.warning(f"API拒绝编辑图片，提示: {first_valid_text}")
                            else:
                                logger.error(f"编辑图片失败，未获取到有效的图片数据")
                                await bot.send_at_message(from_wxid, "\n图片编辑失败，请稍后再试或修改描述", [sender_wxid])
                    except Exception as e:
                        logger.error(f"编辑图片失败: {str(e)}")
                        logger.error(traceback.format_exc())
                        await bot.send_at_message(from_wxid, f"\n编辑图片失败: {str(e)}", [sender_wxid])
                    return False  # 已处理命令，阻止后续插件执行

        # 不是本插件的命令，继续执行后续插件
        return True

    @on_image_message(priority=30)
    async def handle_image_edit(self, bot: WechatAPIClient, message: dict) -> bool:
        """处理图片消息，缓存图片数据以备后续编辑使用"""
        if not self.enable:
            return True  # 插件未启用，继续执行后续插件

        # 获取会话信息
        chat_id = message.get("chat_id", message.get("FromWxid", ""))
        user_id = message.get("user_id", message.get("SenderWxid", ""))
        conversation_key = f"{chat_id}_{user_id}"

        from_wxid = message.get("FromWxid", "")
        sender_wxid = message.get("SenderWxid", "")

        # 记录详细的消息信息
        logger.info(f"GeminiImage收到图片消息: MsgId={message.get('MsgId', '')}, FromWxid={from_wxid}, SenderWxid={sender_wxid}")
        logger.info(f"等待融图状态: {self.waiting_for_merge_images}")

        # 确保使用正确的用户ID
        if not user_id and sender_wxid:
            user_id = sender_wxid
            logger.info(f"使用SenderWxid作为用户ID: {user_id}")
        elif not user_id and from_wxid:
            user_id = from_wxid
            logger.info(f"使用FromWxid作为用户ID: {user_id}")

        # 检查是否在等待融图图片
        if user_id in self.waiting_for_merge_images:
            merge_data = self.waiting_for_merge_images[user_id]
            # 检查是否超时
            if time.time() - merge_data["开始时间"] > self.merge_image_wait_timeout:
                # 超时，清除等待状态
                del self.waiting_for_merge_images[user_id]
                await bot.send_text_message(chat_id, "融图等待超时，请重新开始")
                logger.info(f"用户 {user_id} 融图等待超时，已清除等待状态")
            else:
                # 未超时，添加图片到列表
                image_list = merge_data["图片列表"]

                # 检查是否已达到最大图片数量
                if len(image_list) >= self.max_merge_images:
                    await bot.send_text_message(chat_id, f"已达到最大图片数量 {self.max_merge_images} 张，请发送 {self.start_merge_commands[0]} 开始融合")
                    logger.info(f"用户 {user_id} 已达到最大融图图片数量 {self.max_merge_images} 张")
                    return False  # 阻断后续插件执行

                logger.info(f"用户 {user_id} 正在等待融图图片，当前已有 {len(image_list)} 张图片")

        # 检查是否在等待反向提示词图片
        if user_id in self.waiting_for_reverse_image and self.waiting_for_reverse_image[user_id]:
            # 检查是否超时
            if time.time() - self.waiting_for_reverse_image_time[user_id] > self.reverse_image_wait_timeout:
                # 超时，清除等待状态
                del self.waiting_for_reverse_image[user_id]
                del self.waiting_for_reverse_image_time[user_id]
                await bot.send_text_message(chat_id, "反向提示词等待超时，请重新开始")
            else:
                # 未超时，处理反向提示词请求
                # 扣除积分
                if self.enable_points and self.reverse_cost > 0:
                    await self.db.update_user_points(user_id, -self.reverse_cost)

        # 检查是否在等待图片分析
        if user_id in self.waiting_for_analyze_image and self.waiting_for_analyze_image[user_id]:
            # 检查是否超时
            if time.time() - self.waiting_for_analyze_image_time[user_id] > self.analyze_image_wait_timeout:
                # 超时，清除等待状态
                del self.waiting_for_analyze_image[user_id]
                del self.waiting_for_analyze_image_time[user_id]
                await bot.send_text_message(chat_id, "图片分析等待超时，请重新开始")
            else:
                # 未超时，处理图片分析请求
                # 扣除积分
                if self.enable_points and self.analysis_cost > 0:
                    points_before = await self.db.get_user_points(user_id)
                    await self.db.update_user_points(user_id, -self.analysis_cost)
                    points_after = await self.db.get_user_points(user_id)
                    logger.info(f"用户 {user_id} 图片分析扣除积分 {self.analysis_cost}，积分变化: {points_before} -> {points_after}")

        # 在群聊中，使用发送者ID作为图片所有者
        # 在私聊中，FromWxid和SenderWxid相同
        is_group = message.get("IsGroup", False)
        image_owner = sender_wxid if is_group else from_wxid

        try:
            # 清理过期缓存
            self._cleanup_image_cache()

            # 提取图片数据 - 首先尝试直接从ImgBuf获取
            if "ImgBuf" in message and message["ImgBuf"] and len(message["ImgBuf"]) > 100:
                image_data = message["ImgBuf"]
                logger.info(f"从ImgBuf提取到图片数据，大小: {len(image_data)} 字节")

                # 保存图片到缓存
                self._save_image_to_cache(from_wxid, image_owner, image_data)

                # 处理融图图片
                if user_id in self.waiting_for_merge_images:
                    merge_data = self.waiting_for_merge_images[user_id]
                    image_list = merge_data["图片列表"]

                    # 检查是否已达到最大图片数量
                    if len(image_list) >= self.max_merge_images:
                        await bot.send_text_message(chat_id, f"已达到最大图片数量 {self.max_merge_images} 张，请发送 {self.start_merge_commands[0]} 开始融合")
                        logger.info(f"用户 {user_id} 已达到最大融图图片数量 {self.max_merge_images} 张")
                    else:
                        # 添加图片到列表
                        image_list.append(image_data)
                        logger.info(f"已添加第 {len(image_list)} 张融图图片，大小: {len(image_data)} 字节")

                        # 发送提示消息
                        await bot.send_text_message(chat_id, f"已添加第 {len(image_list)} 张图片，还可以继续添加 {self.max_merge_images - len(image_list)} 张图片，或发送 {self.start_merge_commands[0]} 开始融合")

                        # 如果已达到最大图片数量，自动开始融合
                        if len(image_list) >= self.max_merge_images:
                            prompt = merge_data["提示词"]
                            logger.info(f"已达到最大融图图片数量 {self.max_merge_images}，自动开始融合，提示词: {prompt}")

                            # 扣除积分
                            if self.enable_points and self.merge_cost > 0:
                                points_before = await self.db.get_user_points(user_id)
                                await self.db.update_user_points(user_id, -self.merge_cost)
                                points_after = await self.db.get_user_points(user_id)
                                logger.info(f"用户 {user_id} 融图扣除积分 {self.merge_cost}，积分变化: {points_before} -> {points_after}")

                            # 处理融图请求
                            success = await self._handle_merge_images(bot, message, prompt, image_list)

                            # 清除等待状态
                            del self.waiting_for_merge_images[user_id]
                            logger.info(f"融图处理{'成功' if success else '失败'}，已清除用户 {user_id} 的等待状态")

                # 处理反向提示词图片
                if user_id in self.waiting_for_reverse_image and self.waiting_for_reverse_image[user_id]:
                    # 清除等待状态
                    del self.waiting_for_reverse_image[user_id]
                    del self.waiting_for_reverse_image_time[user_id]

                    # 处理反向提示词请求
                    await self._handle_reverse_image(bot, message, image_data)
                    return False  # 阻断后续插件执行

                # 处理图片分析请求
                if user_id in self.waiting_for_analyze_image and self.waiting_for_analyze_image[user_id]:
                    # 清除等待状态
                    del self.waiting_for_analyze_image[user_id]
                    del self.waiting_for_analyze_image_time[user_id]

                    # 处理图片分析请求
                    await self._handle_analyze_image(bot, message, image_data)
                    return False  # 阻断后续插件执行

                return False  # 阻断后续插件执行

            # 如果ImgBuf中没有有效数据，尝试从Content中提取Base64图片数据
            content = message.get("Content", "")
            if content and isinstance(content, str):
                # 检查是否是XML格式
                if content.startswith("<?xml") and "<img" in content:
                    logger.info("检测到XML格式图片消息，尝试提取Base64数据")
                    try:
                        # 查找XML后附带的Base64数据
                        xml_end = content.find("</msg>")
                        if xml_end > 0 and len(content) > xml_end + 6:
                            # XML后面可能有Base64数据
                            base64_data = content[xml_end + 6:].strip()
                            if base64_data:
                                try:
                                    image_data = base64.b64decode(base64_data)
                                    logger.info(f"从XML后面提取到Base64数据，长度: {len(image_data)} 字节")

                                    # 保存图片到缓存
                                    self._save_image_to_cache(from_wxid, image_owner, image_data)
                                    return True
                                except Exception as e:
                                    logger.error(f"XML后Base64解码失败: {e}")

                        # 如果上面的方法失败，尝试直接检测任何位置的Base64图片头部标识
                        base64_markers = ["iVBOR", "/9j/", "R0lGOD", "UklGR", "PD94bWw", "Qk0", "SUkqAA"]
                        for marker in base64_markers:
                            if marker in content:
                                idx = content.find(marker)
                                if idx > 0:
                                    try:
                                        # 可能的Base64数据，截取从标记开始到结束的部分
                                        base64_data = content[idx:]
                                        # 去除可能的非Base64字符
                                        base64_data = ''.join(c for c in base64_data if c in
                                                              'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')

                                        # 修正长度确保是4的倍数
                                        padding = len(base64_data) % 4
                                        if padding:
                                            base64_data += '=' * (4 - padding)

                                        # 尝试解码
                                        image_data = base64.b64decode(base64_data)
                                        if len(image_data) > 1000:  # 确保至少有一些数据
                                            logger.info(f"从内容中提取到{marker}格式图片数据，长度: {len(image_data)} 字节")

                                            # 保存图片到缓存 - 使用(聊天ID, 用户ID)作为键
                                            cache_key = (from_wxid, image_owner)
                                            self.image_cache[cache_key] = {
                                                "content": image_data,
                                                "timestamp": time.time()
                                            }
                                            return False  # 阻断后续插件执行
                                    except Exception as e:
                                        logger.error(f"提取{marker}格式图片数据失败: {e}")
                    except Exception as e:
                        logger.error(f"提取XML中图片数据失败: {e}")

                # 如果前面的方法都失败了，再尝试一种方法，直接提取整个content作为可能的Base64数据
                # 这对于某些不标准的消息格式可能有效
                try:
                    # 尝试将整个content作为Base64处理
                    base64_content = content.replace(' ', '+')  # 修复可能的URL安全编码
                    # 修正长度确保是4的倍数
                    padding = len(base64_content) % 4
                    if padding:
                        base64_content += '=' * (4 - padding)

                    image_data = base64.b64decode(base64_content)
                    # 如果解码成功且数据量足够大，可能是图片
                    if len(image_data) > 10000:  # 图片数据通常较大
                        try:
                            # 仅尝试打开，不进行验证，避免某些非标准图片格式失败
                            with Image.open(BytesIO(image_data)) as img:
                                width, height = img.size
                                if width > 10 and height > 10:  # 确保是有效图片
                                    logger.info(f"从内容解码成功，图片尺寸: {width}x{height}")

                                    # 保存图片到缓存
                                    self._save_image_to_cache(from_wxid, image_owner, image_data)

                                    # 处理融图图片
                                    if user_id in self.waiting_for_merge_images:
                                        merge_data = self.waiting_for_merge_images[user_id]
                                        image_list = merge_data["图片列表"]

                                        # 添加图片到列表
                                        image_list.append(image_data)
                                        logger.info(f"已添加第 {len(image_list)} 张融图图片，大小: {len(image_data)} 字节")

                                        # 发送提示消息
                                        await bot.send_text_message(chat_id, f"已添加第 {len(image_list)} 张图片，还可以继续添加 {self.max_merge_images - len(image_list)} 张图片，或发送 {self.start_merge_commands[0]} 开始融合")

                                        # 如果已达到最大图片数量，自动开始融合
                                        if len(image_list) >= self.max_merge_images:
                                            prompt = merge_data["提示词"]
                                            logger.info(f"已达到最大融图图片数量 {self.max_merge_images}，自动开始融合，提示词: {prompt}")

                                            # 扣除积分
                                            if self.enable_points and self.merge_cost > 0:
                                                await self.db.update_user_points(user_id, -self.merge_cost)
                                                logger.info(f"已扣除融图积分 {self.merge_cost}")

                                            # 处理融图请求
                                            await self._handle_merge_images(bot, message, prompt, image_list)

                                            # 清除等待状态
                                            del self.waiting_for_merge_images[user_id]
                                            logger.info("融图处理完成，已清除等待状态")

                                        return False  # 阻断后续插件执行

                                    # 处理反向提示词图片
                                    if user_id in self.waiting_for_reverse_image and self.waiting_for_reverse_image[user_id]:
                                        # 清除等待状态
                                        del self.waiting_for_reverse_image[user_id]
                                        del self.waiting_for_reverse_image_time[user_id]

                                        # 处理反向提示词请求
                                        await self._handle_reverse_image(bot, message, image_data)
                                        return False  # 阻断后续插件执行

                                    # 处理图片分析请求
                                    if user_id in self.waiting_for_analyze_image and self.waiting_for_analyze_image[user_id]:
                                        # 清除等待状态
                                        del self.waiting_for_analyze_image[user_id]
                                        del self.waiting_for_analyze_image_time[user_id]

                                        # 处理图片分析请求
                                        await self._handle_analyze_image(bot, message, image_data)
                                        return False  # 阻断后续插件执行

                                    return False  # 阻断后续插件执行
                        except Exception as img_e:
                            logger.error(f"解码后数据不是有效图片: {img_e}")
                except Exception as e:
                    # 解码失败不是错误，只是这种方法不适用
                    pass

            logger.warning("未能从消息中提取有效的图片数据")

            # 如果没有提取到图片数据，但在等待融图、反向提示词或图片分析，发送提示消息
            if user_id in self.waiting_for_merge_images:
                await bot.send_text_message(chat_id, "无法提取图片数据，请重新上传")
            elif user_id in self.waiting_for_reverse_image and self.waiting_for_reverse_image[user_id]:
                await bot.send_text_message(chat_id, "无法提取图片数据，请重新上传")
            elif user_id in self.waiting_for_analyze_image and self.waiting_for_analyze_image[user_id]:
                await bot.send_text_message(chat_id, "无法提取图片数据，请重新上传")
        except Exception as e:
            logger.error(f"处理图片消息失败: {str(e)}")
            logger.error(traceback.format_exc())

        # 如果是在等待融图、反向提示词或图片分析的状态，阻断后续插件执行
        if user_id in self.waiting_for_merge_images or (user_id in self.waiting_for_reverse_image and self.waiting_for_reverse_image[user_id]) or (user_id in self.waiting_for_analyze_image and self.waiting_for_analyze_image[user_id]):
            return False  # 阻断后续插件执行
        return True  # 继续执行后续插件

    def _cleanup_expired_conversations(self):
        """清理过期的会话"""
        current_time = time.time()
        expired_keys = []

        for key, timestamp in self.conversation_timestamps.items():
            if current_time - timestamp > self.conversation_expiry:
                expired_keys.append(key)

        for key in expired_keys:
            if key in self.conversations:
                del self.conversations[key]
            if key in self.conversation_timestamps:
                del self.conversation_timestamps[key]
            if key in self.last_images:
                del self.last_images[key]

    def _cleanup_image_cache(self):
        """清理过期的图片缓存"""
        current_time = time.time()
        expired_keys = []

        for key, cache_data in self.image_cache.items():
            if current_time - cache_data["timestamp"] > self.image_cache_timeout:
                expired_keys.append(key)

        for key in expired_keys:
            del self.image_cache[key]

    @schedule('interval', minutes=5)
    async def scheduled_cleanup(self, bot=None):
        """定时清理过期的图片缓存和会话"""
        try:
            self._cleanup_image_cache()
            self._cleanup_expired_conversations()
            self._cleanup_temp_files()
            # 清理过期的会话密钥映射
            self.clean_expired_session_keys()
            logger.info("定时清理图片缓存、会话、临时文件和会话密钥映射完成")
        except Exception as e:
            logger.error(f"定时清理任务异常: {str(e)}")
            logger.error(traceback.format_exc())

    def has_wake_word(self, message: str) -> bool:
        """检查消息是否包含唤醒词

        Args:
            message: 消息内容

        Returns:
            bool: 是否包含唤醒词
        """
        # 从配置中获取唤醒词列表
        wake_words = self.config.get("wake_words", [])

        # 检查消息是否包含任何唤醒词
        for word in wake_words:
            if word in message:
                logger.info(f"检测到唤醒词 '{word}' 在消息中")
                return True

        return False

    def is_at_message(self, message: dict) -> bool:
        """检查消息是否@了机器人

        Args:
            message: 消息数据

        Returns:
            bool: 是否@了机器人
        """
        # 检查消息内容是否包含@标记
        content = message.get("content", message.get("Content", ""))

        # 获取机器人名称列表
        robot_names = self.config.get("robot_names", ["bot", "机器人"])

        # 检查是否有@机器人的标记
        for name in robot_names:
            at_pattern = f"@{name}"
            if at_pattern in content:
                logger.info(f"检测到@机器人标记 '{at_pattern}' 在消息中")
                return True

        # 检查消息属性中是否标记了@
        is_at = message.get("IsAt", False)
        if is_at:
            logger.info("消息属性中标记了@机器人")
            return True

        return False

    def get_clean_content(self, content: str) -> str:
        """清理消息内容，移除唤醒词和@标记

        Args:
            content: 原始消息内容

        Returns:
            str: 清理后的消息内容
        """
        # 移除@标记
        # 正则表达式匹配@xxx格式
        content = re.sub(r'@[^\s]+\s*', '', content)

        # 移除唤醒词
        wake_words = self.config.get("wake_words", [])
        for word in wake_words:
            if word in content:
                content = content.replace(word, "", 1)  # 只替换第一次出现的唤醒词
                break  # 只移除一个唤醒词

        # 清理多余的空格
        content = content.strip()

        return content

    def _check_message_prefix(self, message: str) -> Tuple[bool, str]:
        """检查消息是否带有所需前缀

        Args:
            message: 消息内容

        Returns:
            Tuple[bool, str]: (是否有前缀, 处理后的消息内容)
        """
        # 如果不需要前缀，直接返回True和原始消息
        if not self.require_prefix_for_conversation:
            return True, message

        # 使用正则表达式检查消息是否以任何前缀开头，允许前缀后面有空格
        import re
        for prefix in self.conversation_prefixes:
            # 检查消息是否以"前缀+空格"开头
            pattern = f"^{re.escape(prefix)}\\s+"
            match = re.match(pattern, message)
            if match:
                # 移除前缀和空格，返回处理后的消息
                processed_message = message[match.end():].strip()
                logger.info(f"检测到前缀 '{prefix}' 带空格，处理后的消息: '{processed_message}'")
                return True, processed_message

            # 也检查消息是否恰好等于前缀（没有后续内容）
            if message.strip() == prefix:
                logger.info(f"检测到前缀 '{prefix}'，但没有后续内容")
                return True, ""

            # 检查消息是否以前缀开头（原始检查）
            if message.startswith(prefix):
                processed_message = message[len(prefix):].strip()
                logger.info(f"检测到前缀 '{prefix}'，处理后的消息: '{processed_message}'")
                return True, processed_message

        # 没有找到前缀
        logger.debug(f"消息 '{message}' 没有包含所需前缀")
        return False, message

    def _cleanup_temp_files(self):
        """清理临时文件"""
        try:
            # 获取当前时间
            current_time = time.time()
            # 获取临时目录中的所有文件
            for file_name in os.listdir(self.save_dir):
                file_path = os.path.join(self.save_dir, file_name)
                # 检查是否是文件
                if os.path.isfile(file_path):
                    # 获取文件的修改时间
                    file_mod_time = os.path.getmtime(file_path)
                    # 如果文件超过24小时未修改，则删除
                    if current_time - file_mod_time > 24 * 3600:
                        try:
                            os.remove(file_path)
                            logger.info(f"已删除过期临时文件: {file_path}")
                        except Exception as e:
                            logger.error(f"删除临时文件失败: {str(e)}")
        except Exception as e:
            logger.error(f"清理临时文件失败: {str(e)}")
            logger.error(traceback.format_exc())

    def _save_image_to_cache(self, from_wxid: str, sender_wxid: str, image_data: bytes):
        """保存图片到缓存

        Args:
            from_wxid: 消息来源ID
            sender_wxid: 发送者ID
            image_data: 图片数据
        """
        # 使用(from_wxid, sender_wxid)作为键
        cache_key = (from_wxid, sender_wxid)
        self.image_cache[cache_key] = {
            "content": image_data,
            "timestamp": time.time()
        }

        logger.info(f"成功缓存图片数据，大小: {len(image_data)} 字节，键: {cache_key}")
        logger.info(f"当前图片缓存包含 {len(self.image_cache)} 个条目")

        # 检查所有可能的用户ID，确保能够找到等待融图状态
        possible_user_ids = [sender_wxid, from_wxid]
        for user_id in possible_user_ids:
            if user_id in self.waiting_for_merge_images:
                merge_data = self.waiting_for_merge_images[user_id]
                image_list = merge_data["图片列表"]

                # 检查是否已达到最大图片数量
                if len(image_list) < self.max_merge_images:
                    # 添加图片到列表
                    image_list.append(image_data)
                    logger.info(f"在_save_image_to_cache中添加第 {len(image_list)} 张融图图片，大小: {len(image_data)} 字节，用户ID: {user_id}")

                    # 更新等待融合的图片列表
                    self.waiting_for_merge_images[user_id]["图片列表"] = image_list

                    # 找到匹配的用户ID后，不再继续检查
                    break
                else:
                    logger.info(f"用户 {user_id} 已达到最大融图图片数量 {self.max_merge_images} 张")
            else:
                logger.info(f"用户 {user_id} 不在等待融图状态")

    async def _get_recent_image(self, chat_id: str, user_id: str) -> Optional[bytes]:
        """获取最近的图片

        Args:
            chat_id: 聊天ID
            user_id: 用户ID

        Returns:
            Optional[bytes]: 图片数据，如果没有则返回None
        """
        try:
            # 检查会话是否活跃
            conversation_key = f"{chat_id}_{user_id}"
            if conversation_key not in self.conversations:
                # 会话已经结束，不返回任何图片
                logger.info(f"会话 {conversation_key} 已结束或不存在，不返回任何图片")
                return None

            # 清理过期缓存
            self._cleanup_image_cache()

            # 尝试从缓存中获取图片
            for key, cache_data in self.image_cache.items():
                # 检查是否是当前聊天的图片
                if key[0] == chat_id or key[1] == user_id:
                    return cache_data["content"]

            # 如果没有找到，尝试从最后生成的图片中获取
            if conversation_key in self.last_images:
                image_path = self.last_images[conversation_key]
                if os.path.exists(image_path):
                    with open(image_path, "rb") as f:
                        return f.read()

            return None
        except Exception as e:
            logger.error(f"获取最近图片失败: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def _is_multi_image_request(self, text: str) -> bool:
        """检测是否是多图文请求

        Args:
            text: 要检查的文本

        Returns:
            bool: 是否是多图文请求
        """
        if not text or not isinstance(text, str):
            return False

        # 多图文请求的关键词和模式
        multi_image_keywords = [
            # 多个场景/图片相关
            '多个场景', '多个图片', '多张图片', '多幅图片',
            '多个步骤', '多个阶段', '多个过程',
            '多幅图', '多幅画', '多幅描述',

            # 图文并茂相关
            '图文并茂', '图文并茂形式', '图文并茂形式告诉',
            '图文结合', '图文对应', '图文配合',

            # 顺序/步骤相关
            '按顺序', '按步骤', '按阶段', '按过程',
            '一步一步', '一步步', '一步一图', '一步一张',
            '一幅一幅', '一幅一段', '一幅一句',

            # 教程相关
            '教程', '教学', '教程图片', '教学图片',
            '演示', '演示图片', '演示过程',

            # 配图相关
            '每个步骤配一张图', '每个场景配一张图',
            '每个阶段配一张图', '每个过程配一张图',
            '每一幅', '每一张', '每一步', '每一阶段',
            '配上文字', '配上文字说明', '配文字', '配文字说明',
            '配图', '配图片', '配描述', '配说明',

            # 绘本/漫画/连环画相关
            '绘本', '绘本故事', '绘本形式', '绘本风格',
            '漫画', '漫画形式', '漫画风格', '漫画故事',
            '连环画', '连环画形式', '连环画风格', '连环画故事',
            '故事书', '故事书形式', '故事书风格',
            '分页', '分页展示', '分页描述',

            # 其他多图文相关
            '系列图片', '系列图片展示', '系列图片描述',
            '连续图片', '连续图片展示', '连续图片描述'
        ]

        # 检查是否包含多图文关键词
        for keyword in multi_image_keywords:
            if keyword in text:
                return True

        # 检查是否包含数字+步骤/场景的模式，如"1.准备材料 2.切菜"
        if re.search(r'\d+[.\s]*[步骤场景阶段过程幅张图片图画]', text):
            return True

        # 检查是否包含"第一步""第二步"等模式
        if re.search(r'[第首最先][一二三四五六七八九十两三四五六七八九十][步骤场景阶段过程幅张图片图画页]', text):
            return True

        # 检查是否包含"怎么做""如何做"等模式，通常表示教程
        if re.search(r'[怎么如何][做制作烹饪烧煮煎炒焖煬煲煸熙炖炒焖煬煲煸熙炖]', text):
            return True

        # 检查是否包含"每一"+图片/步骤等模式
        if re.search(r'每[一个一张一幅一步一阶段一过程一场景]', text):
            return True

        # 检查是否包含"配"+文字/图片等模式
        if re.search(r'配[上文字图片图描述说明]', text):
            return True

        # 检查是否包含"绘本""漫画""连环画""故事书"等模式
        if re.search(r'[绘漫连故][本画环事][故书画的形式风格样式]?', text):
            return True

        return False

    async def _enhance_multi_image_prompt(self, prompt: str) -> str:
        """增强多图文提示词，生成分镜脚本

        Args:
            prompt: 原始提示词

        Returns:
            str: 增强后的分镜脚本
        """
        try:
            # 使用多图文系统提示词
            url = f"{self.base_url}/v1beta/models/{self.prompt_model}:generateContent"
            # 检查URL格式是否正确
            if not url.startswith("http"):
                logger.warning(f"URL格式可能不正确: {url}")
                # 尝试修复URL格式
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.prompt_model}:generateContent"
            headers = {
                "Content-Type": "application/json",
            }

            params = {
                "key": self.api_key
            }

            # 构建用户提示词，添加明确的指示要求生成图片
            user_prompt = f"{prompt}\n\n请为每个场景生成详细的中文提示词，以便后续生成图片。"

            data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": user_prompt
                            }
                        ]
                    }
                ],
                "systemInstruction": {
                    "role": "system",
                    "parts": [
                        {
                            "text": MULTI_IMAGE_SYSTEM_PROMPT
                        }
                    ]
                },
                "generationConfig": {
                    "temperature": 0.9,
                    "topP": 0.95,
                    "topK": 64,
                    "maxOutputTokens": 8192,
                    "responseMimeType": "text/plain"
                }
            }

            # 创建代理配置
            proxy = None
            if self.enable_proxy and self.proxy_url:
                proxy = self.proxy_url

            # 使用重试机制
            retry_count = 0
            retry_delay = self.initial_retry_delay

            while retry_count <= self.max_retries:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url,
                            headers=headers,
                            params=params,
                            json=data,
                            proxy=proxy,
                            timeout=aiohttp.ClientTimeout(total=60)  # 增加超时时间到60秒
                        ) as response:
                            response_text = await response.text()

                            if response.status == 200:
                                result = json.loads(response_text)
                                candidates = result.get("candidates", [])
                                if candidates and len(candidates) > 0:
                                    content = candidates[0].get("content", {})
                                    parts = content.get("parts", [])

                                    for part in parts:
                                        if "text" in part and part["text"]:
                                            enhanced_prompt = part["text"]
                                            logger.info(f"多图文分镜脚本生成成功，长度: {len(enhanced_prompt)}")
                                            # 添加明确的指示，确保生成图片
                                            enhanced_prompt += "\n\n请生成上述场景的图片，确保在回复中包含图片。"
                                            return enhanced_prompt

                                return prompt  # 如果无法解析响应，返回原始提示词
                            else:
                                logger.error(f"生成多图文分镜脚本API调用失败 (状态码: {response.status}): {response_text}")

                                # 检查是否是可重试的错误
                                if response.status in [429, 500, 502, 503, 504]:
                                    retry_count += 1
                                    if retry_count <= self.max_retries:
                                        logger.info(f"第 {retry_count} 次重试生成多图文分镜脚本，等待 {retry_delay} 秒")
                                        await asyncio.sleep(retry_delay)
                                        # 指数退避策略
                                        retry_delay = min(retry_delay * 2, self.max_retry_delay)
                                        continue

                                return prompt  # 返回原始提示词
                except Exception as e:
                    logger.error(f"生成多图文分镜脚本异常: {str(e)}")

                    retry_count += 1
                    if retry_count <= self.max_retries:
                        logger.info(f"第 {retry_count} 次重试生成多图文分镜脚本，等待 {retry_delay} 秒")
                        await asyncio.sleep(retry_delay)
                        # 指数退避策略
                        retry_delay = min(retry_delay * 2, self.max_retry_delay)
                        continue

                    return prompt  # 返回原始提示词

                # 如果成功，跳出循环
                break

            # 如果所有重试都失败，返回原始提示词加上默认的多图文指示
            return f"{prompt}\n\n请生成多个场景的图片，确保在回复中包含图片。"
        except Exception as e:
            logger.error(f"生成多图文分镜脚本失败: {e}")
            logger.error(traceback.format_exc())
            # 如果异常，返回原始提示词加上默认的多图文指示
            return f"{prompt}\n\n请生成多个场景的图片，确保在回复中包含图片。"

    def _extract_story_content(self, text: str) -> List[str]:
        """从分镜脚本中提取故事内容/说明文字

        Args:
            text: 分镜脚本文本

        Returns:
            List[str]: 提取出的故事内容/说明文字列表
        """
        story_contents = []

        # 使用正则表达式提取故事内容/说明文字
        pattern = r"\*\*故事内容/说明文字：\*\*\s*([^\*]+)"
        matches = re.findall(pattern, text)

        if matches:
            for match in matches:
                story_contents.append(match.strip())

        return story_contents

    def _extract_chinese_prompt(self, text: str) -> List[str]:
        """从分镜脚本中提取中文提示词

        Args:
            text: 分镜脚本文本

        Returns:
            List[str]: 提取出的中文提示词列表
        """
        chinese_prompts = []

        # 记录原始文本长度，用于调试
        logger.info(f"提取中文提示词，原始文本长度: {len(text)} 字节")

        # 使用多种模式匹配中文提示词
        patterns = [
            # 标准格式：**中文提示词：** 内容
            r"\*\*中文提示词：\*\*\s*([^\*]+)(?=\*\*|———|——|—|$)",
            # 变体1：中文提示词： 内容
            r"中文提示词：\s*([^\*]+)(?=\*\*|———|——|—|$)",
            # 变体2：**图片提示词：** 内容
            r"\*\*图片提示词：\*\*\s*([^\*]+)(?=\*\*|———|——|—|$)",
            # 变体3：图片提示词： 内容
            r"图片提示词：\s*([^\*]+)(?=\*\*|———|——|—|$)",
            # 变体4：**提示词：** 内容
            r"\*\*提示词：\*\*\s*([^\*]+)(?=\*\*|———|——|—|$)",
            # 变体5：提示词： 内容
            r"提示词：\s*([^\*]+)(?=\*\*|———|——|—|$)",
            # 变体6：**场景N提示词：** 内容 (N是数字)
            r"\*\*场景\d+提示词：\*\*\s*([^\*]+)(?=\*\*|———|——|—|$)",
            # 变体7：场景N提示词： 内容 (N是数字)
            r"场景\d+提示词：\s*([^\*]+)(?=\*\*|———|——|—|$)"
        ]

        # 尝试所有模式
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, text)
            if matches:
                logger.info(f"使用模式 {i+1} 找到 {len(matches)} 个中文提示词")
                for match in matches:
                    # 清理提示词，移除多余的空白字符和换行符
                    cleaned_match = re.sub(r'\s+', ' ', match).strip()
                    if cleaned_match and cleaned_match not in chinese_prompts:  # 避免重复
                        chinese_prompts.append(cleaned_match)

        # 如果没有找到中文提示词，尝试查找英文提示词并标记
        if not chinese_prompts:
            english_patterns = [
                r"\*\*English Prompt:\*\*\s*([^\*]+)(?=\*\*|———|——|—|$)",
                r"English Prompt:\s*([^\*]+)(?=\*\*|———|——|—|$)"
            ]

            for pattern in english_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    logger.info(f"找到 {len(matches)} 个英文提示词")
                    for match in matches:
                        cleaned_match = re.sub(r'\s+', ' ', match).strip()
                        if cleaned_match:
                            # 标记为英文提示词，后续处理可能需要翻译
                            chinese_prompts.append(f"[英文提示词] {cleaned_match}")

        # 如果仍然没有找到提示词，尝试使用分段方式提取
        if not chinese_prompts and "场景" in text and "：" in text:
            # 尝试按场景分段提取
            scene_blocks = re.split(r'场景\s*\d+\s*[:：]', text)
            if len(scene_blocks) > 1:  # 第一个元素是分割前的内容，跳过
                logger.info(f"按场景分段找到 {len(scene_blocks)-1} 个场景块")
                for i, block in enumerate(scene_blocks[1:], 1):
                    # 取每个场景的前200个字符作为提示词
                    if block.strip():
                        prompt = block.strip()[:200]
                        chinese_prompts.append(f"场景{i}: {prompt}")

        logger.info(f"最终提取到 {len(chinese_prompts)} 个中文提示词")
        if chinese_prompts:
            for i, prompt in enumerate(chinese_prompts):
                logger.info(f"提示词 {i+1}: {prompt[:50]}..." if len(prompt) > 50 else f"提示词 {i+1}: {prompt}")

        return chinese_prompts

    def _translate_gemini_message(self, message: str) -> str:
        """翻译Gemini的错误消息"""
        if not message:
            return ""

        # 常见的拒绝消息模式
        rejection_patterns = {
            r"I cannot generate": "我无法生成",
            r"I'm unable to generate": "我无法生成",
            r"I apologize": "很抱歉",
            r"I'm sorry": "很抱歉",
            r"violates our content policy": "违反了内容政策",
            r"harmful content": "有害内容",
            r"inappropriate content": "不适当的内容",
            r"content policy": "内容政策",
            r"safety guidelines": "安全准则",
            r"cannot fulfill": "无法满足",
            r"cannot create": "无法创建",
            r"cannot provide": "无法提供",
        }

        translated = message
        for pattern, replacement in rejection_patterns.items():
            translated = re.sub(pattern, replacement, translated, flags=re.IGNORECASE)

        return translated

    def _clean_response_text(self, text: str) -> str:
        """清理响应文本，移除对话式语句"""
        if not text:
            return ""

        # 移除常见的对话式开头
        patterns_to_remove = [
            r"^好的，",
            r"^当然，",
            r"^我已经",
            r"^以下是",
            r"^这是",
            r"^请参考",
            r"^根据您的要求",
            r"^根据你的要求",
            r"^根据您的图片",
            r"^根据你的图片",
            r"^我理解您",
            r"^我理解你",
            r"^我会",
        ]

        result = text
        for pattern in patterns_to_remove:
            result = re.sub(pattern, "", result)

        return result.strip()

    async def _enhance_prompt(self, prompt: str, is_edit: bool = False) -> str:
        """增强提示词"""
        if not self.enhance_prompt:
            return prompt

        try:
            # 检查是否是编辑指令
            if is_edit:
                return await self._enhance_edit_prompt(prompt)

            # 使用标准系统提示词增强提示词
            url = f"{self.base_url}/v1beta/models/{self.prompt_model}:generateContent"
            # 检查URL格式是否正确
            if not url.startswith("http"):
                logger.warning(f"URL格式可能不正确: {url}")
                # 尝试修复URL格式
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.prompt_model}:generateContent"
            headers = {
                "Content-Type": "application/json",
            }

            # 获取会话ID
            session_id = f"enhance_{uuid.uuid4().hex[:8]}"  # 为提示词增强生成一个唯一的会话ID

            # 获取API密钥
            api_key = self.get_api_key_for_session(session_id)

            params = {
                "key": api_key
            }

            data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "systemInstruction": {
                    "role": "system",
                    "parts": [
                        {
                            "text": STANDARD_SYSTEM_PROMPT
                        }
                    ]
                }
            }

            # 创建代理配置
            proxy = None
            if self.enable_proxy and self.proxy_url:
                proxy = self.proxy_url

            # 使用重试机制
            retry_count = 0
            retry_delay = self.initial_retry_delay

            while retry_count <= self.max_retries:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url,
                            headers=headers,
                            params=params,
                            json=data,
                            proxy=proxy,
                            timeout=aiohttp.ClientTimeout(total=60)  # 增加超时时间到60秒
                        ) as response:
                            response_text = await response.text()

                            if response.status == 200:
                                result = json.loads(response_text)
                                candidates = result.get("candidates", [])
                                if candidates and len(candidates) > 0:
                                    content = candidates[0].get("content", {})
                                    parts = content.get("parts", [])

                                    for part in parts:
                                        if "text" in part and part["text"]:
                                            return part["text"]

                                return prompt  # 如果无法解析响应，返回原始提示词
                            else:
                                logger.error(f"增强提示词API调用失败 (状态码: {response.status}): {response_text}")

                                # 如果是API密钥错误，尝试切换密钥
                                if response.status == 400 and "API key not valid" in response_text:
                                    logger.warning("API密钥无效，尝试切换密钥")
                                    # 标记当前密钥出错，并获取新密钥
                                    new_api_key = self.mark_api_key_error(api_key, session_id)
                                    if new_api_key and new_api_key != api_key:
                                        # 更新请求参数中的API密钥
                                        api_key = new_api_key
                                        params["key"] = api_key
                                        logger.info("已切换到新的API密钥")
                                        # 不增加重试计数，直接使用新密钥重试
                                        continue

                                # 检查是否是可重试的错误
                                if response.status in [429, 500, 502, 503, 504]:
                                    retry_count += 1
                                    if retry_count <= self.max_retries:
                                        logger.info(f"第 {retry_count} 次重试增强提示词，等待 {retry_delay} 秒")
                                        await asyncio.sleep(retry_delay)
                                        # 指数退避策略
                                        retry_delay = min(retry_delay * 2, self.max_retry_delay)
                                        continue

                                return prompt  # 返回原始提示词
                except Exception as e:
                    logger.error(f"增强提示词异常: {str(e)}")

                    retry_count += 1
                    if retry_count <= self.max_retries:
                        logger.info(f"第 {retry_count} 次重试增强提示词，等待 {retry_delay} 秒")
                        await asyncio.sleep(retry_delay)
                        # 指数退避策略
                        retry_delay = min(retry_delay * 2, self.max_retry_delay)
                        continue

                    return prompt  # 返回原始提示词

                # 如果成功，跳出循环
                break

            return prompt  # 如果所有重试都失败，返回原始提示词
        except Exception as e:
            logger.error(f"增强提示词失败: {str(e)}")
            logger.error(traceback.format_exc())
            return prompt  # 返回原始提示词

    async def _enhance_prompt_direct(self, prompt: str, detailed_output: bool = False) -> str:
        """直接生成详细提示词，用于提示词生成功能"""
        try:
            # 使用详细输出系统提示词
            url = f"{self.base_url}/v1beta/models/{self.prompt_model}:generateContent"
            # 检查URL格式是否正确
            if not url.startswith("http"):
                logger.warning(f"URL格式可能不正确: {url}")
                # 尝试修复URL格式
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.prompt_model}:generateContent"
            headers = {
                "Content-Type": "application/json",
            }

            # 获取会话ID
            session_id = f"enhance_direct_{uuid.uuid4().hex[:8]}"  # 为直接提示词增强生成一个唯一的会话ID

            # 获取API密钥
            api_key = self.get_api_key_for_session(session_id)

            params = {
                "key": api_key
            }

            data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "systemInstruction": {
                    "role": "system",
                    "parts": [
                        {
                            "text": DETAILED_SYSTEM_PROMPT if detailed_output else STANDARD_SYSTEM_PROMPT
                        }
                    ]
                }
            }

            # 创建代理配置
            proxy = None
            if self.enable_proxy and self.proxy_url:
                proxy = self.proxy_url

            # 使用重试机制
            retry_count = 0
            retry_delay = self.initial_retry_delay

            while retry_count <= self.max_retries:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url,
                            headers=headers,
                            params=params,
                            json=data,
                            proxy=proxy,
                            timeout=aiohttp.ClientTimeout(total=60)  # 增加超时时间到60秒
                        ) as response:
                            response_text = await response.text()

                            if response.status == 200:
                                result = json.loads(response_text)
                                candidates = result.get("candidates", [])
                                if candidates and len(candidates) > 0:
                                    content = candidates[0].get("content", {})
                                    parts = content.get("parts", [])

                                    for part in parts:
                                        if "text" in part and part["text"]:
                                            return part["text"]

                                return None  # 如果无法解析响应，返回None
                            else:
                                logger.error(f"生成提示词API调用失败 (状态码: {response.status}): {response_text}")

                                # 如果是API密钥错误，尝试切换密钥
                                if response.status == 400 and "API key not valid" in response_text:
                                    logger.warning("API密钥无效，尝试切换密钥")
                                    # 标记当前密钥出错，并获取新密钥
                                    new_api_key = self.mark_api_key_error(api_key, session_id)
                                    if new_api_key and new_api_key != api_key:
                                        # 更新请求参数中的API密钥
                                        api_key = new_api_key
                                        params["key"] = api_key
                                        logger.info("已切换到新的API密钥")
                                        # 不增加重试计数，直接使用新密钥重试
                                        continue

                                # 检查是否是可重试的错误
                                if response.status in [429, 500, 502, 503, 504]:
                                    retry_count += 1
                                    if retry_count <= self.max_retries:
                                        logger.info(f"第 {retry_count} 次重试生成提示词，等待 {retry_delay} 秒")
                                        await asyncio.sleep(retry_delay)
                                        # 指数退避策略
                                        retry_delay = min(retry_delay * 2, self.max_retry_delay)
                                        continue

                                return None  # 返回None
                except Exception as e:
                    logger.error(f"生成提示词异常: {str(e)}")

                    retry_count += 1
                    if retry_count <= self.max_retries:
                        logger.info(f"第 {retry_count} 次重试生成提示词，等待 {retry_delay} 秒")
                        await asyncio.sleep(retry_delay)
                        # 指数退避策略
                        retry_delay = min(retry_delay * 2, self.max_retry_delay)
                        continue

                    return None  # 返回None

                # 如果成功，跳出循环
                break

            return None  # 如果所有重试都失败，返回None
        except Exception as e:
            logger.error(f"生成提示词失败: {str(e)}")
            logger.error(traceback.format_exc())
            return None  # 返回None

    async def _enhance_edit_prompt(self, prompt: str) -> str:
        """增强编辑提示词，使其更加适合图像编辑

        Args:
            prompt: 原始编辑提示词

        Returns:
            str: 增强后的编辑提示词
        """
        if not self.enhance_prompt:
            return prompt

        # 记录原始提示词，用于调试
        logger.info(f"开始增强编辑提示词: {prompt}")

        # 检查提示词是否为空
        if not prompt or not prompt.strip():
            logger.warning("编辑提示词为空，返回默认提示词")
            return "请编辑图片，保持原始图片的主要内容和风格"

        try:
            # 使用编辑图像系统提示词
            url = f"{self.base_url}/v1beta/models/{self.prompt_model}:generateContent"
            # 检查URL格式是否正确
            if not url.startswith("http"):
                logger.warning(f"URL格式可能不正确: {url}")
                # 尝试修复URL格式
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.prompt_model}:generateContent"
            headers = {
                "Content-Type": "application/json",
            }

            # 获取会话ID
            session_id = f"enhance_edit_{uuid.uuid4().hex[:8]}"  # 为编辑提示词增强生成一个唯一的会话ID

            # 获取API密钥
            api_key = self.get_api_key_for_session(session_id)

            params = {
                "key": api_key
            }

            # 使用系统提示词
            system_prompt = EDIT_IMAGE_SYSTEM_PROMPT

            # 构建用户提示
            enhanced_user_prompt = f"系统提示: {system_prompt}\n\n编辑指令: {prompt}"

            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": enhanced_user_prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.3,  # 降低温度，使结果更保守
                    "maxOutputTokens": 1024,  # 编辑提示词通常较短
                    "topP": 0.9,
                    "topK": 40,
                }
            }

            # 创建代理配置
            proxy = None
            if self.enable_proxy and self.proxy_url:
                proxy = self.proxy_url

            # 使用重试机制
            retry_count = 0
            retry_delay = self.initial_retry_delay

            while retry_count <= self.max_retries:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url,
                            headers=headers,
                            params=params,
                            json=data,
                            proxy=proxy,
                            timeout=aiohttp.ClientTimeout(total=60)  # 增加超时时间到60秒
                        ) as response:
                            response_text = await response.text()

                            if response.status == 200:
                                result = json.loads(response_text)
                                candidates = result.get("candidates", [])
                                if candidates and len(candidates) > 0:
                                    content = candidates[0].get("content", {})
                                    parts = content.get("parts", [])

                                    for part in parts:
                                        if "text" in part and part["text"]:
                                            return part["text"]

                                return prompt  # 如果无法解析响应，返回原始提示词
                            else:
                                logger.error(f"增强编辑提示词API调用失败 (状态码: {response.status}): {response_text}")

                                # 如果是API密钥错误，尝试切换密钥
                                if response.status == 400 and "API key not valid" in response_text:
                                    logger.warning("API密钥无效，尝试切换密钥")
                                    # 标记当前密钥出错，并获取新密钥
                                    new_api_key = self.mark_api_key_error(api_key, session_id)
                                    if new_api_key and new_api_key != api_key:
                                        # 更新请求参数中的API密钥
                                        api_key = new_api_key
                                        params["key"] = api_key
                                        logger.info("已切换到新的API密钥")
                                        # 不增加重试计数，直接使用新密钥重试
                                        continue

                                # 检查是否是可重试的错误
                                if response.status in [429, 500, 502, 503, 504]:
                                    retry_count += 1
                                    if retry_count <= self.max_retries:
                                        logger.info(f"第 {retry_count} 次重试增强编辑提示词，等待 {retry_delay} 秒")
                                        await asyncio.sleep(retry_delay)
                                        # 指数退避策略
                                        retry_delay = min(retry_delay * 2, self.max_retry_delay)
                                        continue

                                return prompt  # 返回原始提示词
                except Exception as e:
                    logger.error(f"增强编辑提示词异常: {str(e)}")

                    retry_count += 1
                    if retry_count <= self.max_retries:
                        logger.info(f"第 {retry_count} 次重试增强编辑提示词，等待 {retry_delay} 秒")
                        await asyncio.sleep(retry_delay)
                        # 指数退避策略
                        retry_delay = min(retry_delay * 2, self.max_retry_delay)
                        continue

                    return prompt  # 返回原始提示词

                # 如果成功，跳出循环
                break

            return prompt  # 如果所有重试都失败，返回原始提示词
        except Exception as e:
            logger.error(f"增强编辑提示词失败: {str(e)}")
            logger.error(traceback.format_exc())
            return prompt  # 返回原始提示词

    async def _analyze_image(self, image_data: bytes, message_info: dict = None) -> Optional[str]:
        """分析图片内容，返回详细分析结果

        Args:
            image_data: 图片数据
            message_info: 消息相关信息，包含user_id等
        """
        try:
            # 将图片数据转换为Base64编码
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            # 使用图片分析系统提示词
            url = f"{self.base_url}/v1beta/models/{self.analysis_model}:generateContent"
            headers = {
                "Content-Type": "application/json",
            }

            # 获取会话ID
            session_id = ""
            if message_info and "user_id" in message_info:
                user_id = message_info.get("user_id", "")
                chat_id = message_info.get("chat_id", "")
                session_id = f"{chat_id}_{user_id}"

            # 获取API密钥
            api_key = self.get_api_key_for_session(session_id)

            params = {
                "key": api_key
            }

            # 获取用户的分析问题（如果有）
            user_query = ""
            if message_info and "user_id" in message_info:
                user_id = message_info.get("user_id")
                # 首先尝试直接使用user_id作为键
                user_query = self.waiting_for_analyze_image_query.get(user_id, "")

                # 如果没有找到，尝试使用其他可能的键
                if not user_query and self.waiting_for_analyze_image_query:
                    logger.info(f"使用user_id={user_id}未找到分析问题，尝试其他可能的键")
                    # 尝试使用消息中的其他ID
                    possible_keys = [
                        message_info.get("chat_id", ""),
                        message_info.get("from_wxid", ""),
                        message_info.get("sender_wxid", "")
                    ]

                    # 记录所有可能的键和waiting_for_analyze_image_query的内容
                    logger.info(f"可能的键: {possible_keys}")
                    logger.info(f"waiting_for_analyze_image_query: {self.waiting_for_analyze_image_query}")

                    # 尝试所有可能的键
                    for key in possible_keys:
                        if key and key in self.waiting_for_analyze_image_query:
                            user_query = self.waiting_for_analyze_image_query[key]
                            logger.info(f"使用键 {key} 找到用户分析问题: {user_query}")
                            break

                    # 如果仍然没有找到，使用字典中的第一个非空值
                    if not user_query:
                        for key, value in self.waiting_for_analyze_image_query.items():
                            if value:
                                user_query = value
                                logger.info(f"使用字典中的第一个非空值，键 {key}: {user_query}")
                                break

            # 构建用户提示文本
            if user_query and len(user_query.strip()) > 0:
                user_text = f"请用中文分析这张图片，特别关注：{user_query}"
                logger.info(f"使用用户指定的分析问题: {user_query}")
            else:
                user_text = "请用中文分析这张图片"
                logger.info("使用默认分析提示")

            data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": image_base64
                                }
                            },
                            {
                                "text": user_text
                            }
                        ]
                    }
                ],
                "systemInstruction": {
                    "role": "system",
                    "parts": [
                        {
                            "text": IMAGE_ANALYSIS_PROMPT
                        }
                    ]
                },
                "generationConfig": {
                    "temperature": 0.4,
                    "topP": 0.95,
                    "topK": 40
                }
            }

            # 创建代理配置
            proxy = None
            if self.enable_proxy and self.proxy_url:
                proxy = self.proxy_url

            # 使用重试机制
            retry_count = 0
            retry_delay = self.initial_retry_delay

            while retry_count <= self.max_retries:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url,
                            headers=headers,
                            params=params,
                            json=data,
                            proxy=proxy,
                            timeout=aiohttp.ClientTimeout(total=60)  # 增加超时时间到60秒
                        ) as response:
                            response_text = await response.text()

                            if response.status == 200:
                                result = json.loads(response_text)
                                candidates = result.get("candidates", [])
                                if candidates and len(candidates) > 0:
                                    content = candidates[0].get("content", {})
                                    parts = content.get("parts", [])

                                    for part in parts:
                                        if "text" in part and part["text"]:
                                            return part["text"]

                                return None  # 如果无法解析响应，返回None
                            else:
                                logger.error(f"图片分析API调用失败 (状态码: {response.status}): {response_text}")

                                # 如果是API密钥错误，尝试切换密钥
                                if response.status == 400 and "API key not valid" in response_text:
                                    logger.warning("API密钥无效，尝试切换密钥")
                                    # 标记当前密钥出错，并获取新密钥
                                    new_api_key = self.mark_api_key_error(api_key, session_id)
                                    if new_api_key and new_api_key != api_key:
                                        # 更新请求参数中的API密钥
                                        api_key = new_api_key
                                        params["key"] = api_key
                                        logger.info("已切换到新的API密钥")
                                        # 不增加重试计数，直接使用新密钥重试
                                        continue

                                # 检查是否是可重试的错误
                                if response.status in [429, 500, 502, 503, 504]:
                                    retry_count += 1
                                    if retry_count <= self.max_retries:
                                        logger.info(f"第 {retry_count} 次重试图片分析，等待 {retry_delay} 秒")
                                        await asyncio.sleep(retry_delay)
                                        # 指数退避策略
                                        retry_delay = min(retry_delay * 2, self.max_retry_delay)
                                        continue

                                return None  # 返回None
                except Exception as e:
                    logger.error(f"图片分析异常: {str(e)}")

                    retry_count += 1
                    if retry_count <= self.max_retries:
                        logger.info(f"第 {retry_count} 次重试图片分析，等待 {retry_delay} 秒")
                        await asyncio.sleep(retry_delay)
                        # 指数退避策略
                        retry_delay = min(retry_delay * 2, self.max_retry_delay)
                        continue

                    return None  # 返回None

                # 如果成功，跳出循环
                break

            return None  # 如果所有重试都失败，返回None
        except Exception as e:
            logger.error(f"图片分析失败: {str(e)}")
            logger.error(traceback.format_exc())
            return None  # 返回None

    async def _reverse_image(self, image_data: bytes) -> Optional[str]:
        """从图片生成详细提示词"""
        try:
            # 将图片数据转换为Base64编码
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            # 使用反向提示词系统提示词
            url = f"{self.base_url}/v1beta/models/{self.reverse_model}:generateContent"
            headers = {
                "Content-Type": "application/json",
            }

            # 获取会话ID
            session_id = f"reverse_{uuid.uuid4().hex[:8]}"  # 为反向提示词生成一个唯一的会话ID

            # 获取API密钥
            api_key = self.get_api_key_for_session(session_id)

            params = {
                "key": api_key
            }

            data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ],
                "systemInstruction": {
                    "role": "system",
                    "parts": [
                        {
                            "text": REVERSE_PROMPT
                        }
                    ]
                }
            }

            # 创建代理配置
            proxy = None
            if self.enable_proxy and self.proxy_url:
                proxy = self.proxy_url

            # 使用重试机制
            retry_count = 0
            retry_delay = self.initial_retry_delay

            while retry_count <= self.max_retries:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url,
                            headers=headers,
                            params=params,
                            json=data,
                            proxy=proxy,
                            timeout=aiohttp.ClientTimeout(total=60)  # 增加超时时间到60秒
                        ) as response:
                            response_text = await response.text()

                            if response.status == 200:
                                result = json.loads(response_text)
                                candidates = result.get("candidates", [])
                                if candidates and len(candidates) > 0:
                                    content = candidates[0].get("content", {})
                                    parts = content.get("parts", [])

                                    for part in parts:
                                        if "text" in part and part["text"]:
                                            return part["text"]

                                return None  # 如果无法解析响应，返回None
                            else:
                                logger.error(f"反向提示词API调用失败 (状态码: {response.status}): {response_text}")

                                # 如果是API密钥错误，尝试切换密钥
                                if response.status == 400 and "API key not valid" in response_text:
                                    logger.warning("API密钥无效，尝试切换密钥")
                                    # 标记当前密钥出错，并获取新密钥
                                    new_api_key = self.mark_api_key_error(api_key, session_id)
                                    if new_api_key and new_api_key != api_key:
                                        # 更新请求参数中的API密钥
                                        api_key = new_api_key
                                        params["key"] = api_key
                                        logger.info("已切换到新的API密钥")
                                        # 不增加重试计数，直接使用新密钥重试
                                        continue

                                # 检查是否是可重试的错误
                                if response.status in [429, 500, 502, 503, 504]:
                                    retry_count += 1
                                    if retry_count <= self.max_retries:
                                        logger.info(f"第 {retry_count} 次重试反向提示词，等待 {retry_delay} 秒")
                                        await asyncio.sleep(retry_delay)
                                        # 指数退避策略
                                        retry_delay = min(retry_delay * 2, self.max_retry_delay)
                                        continue

                                return None  # 返回None
                except Exception as e:
                    logger.error(f"反向提示词异常: {str(e)}")

                    retry_count += 1
                    if retry_count <= self.max_retries:
                        logger.info(f"第 {retry_count} 次重试反向提示词，等待 {retry_delay} 秒")
                        await asyncio.sleep(retry_delay)
                        # 指数退避策略
                        retry_delay = min(retry_delay * 2, self.max_retry_delay)
                        continue

                    return None  # 返回None

                # 如果成功，跳出循环
                break

            return None  # 如果所有重试都失败，返回None
        except Exception as e:
            logger.error(f"反向提示词失败: {str(e)}")
            logger.error(traceback.format_exc())
            return None  # 返回None

    async def _enhance_merge_prompt(self, prompt: str) -> str:
        """为融图指令提供专门的提示词增强

        Args:
            prompt: 用户原始融图指令

        Returns:
            str: 增强后的融图指令
        """
        if not self.enhance_prompt:
            return prompt

        try:
            # 使用融图系统提示词
            url = f"{self.base_url}/v1beta/models/{self.prompt_model}:generateContent"
            headers = {
                "Content-Type": "application/json",
            }

            # 获取会话ID
            session_id = f"enhance_{uuid.uuid4().hex[:8]}"  # 为提示词增强生成一个唯一的会话ID

            # 获取API密钥
            api_key = self.get_api_key_for_session(session_id)

            params = {
                "key": api_key
            }

            data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "systemInstruction": {
                    "role": "system",
                    "parts": [
                        {
                            "text": MERGE_IMAGE_SYSTEM_PROMPT
                        }
                    ]
                }
            }

            # 创建代理配置
            proxy = None
            if self.enable_proxy and self.proxy_url:
                proxy = self.proxy_url

            # 使用重试机制
            retry_count = 0
            retry_delay = self.initial_retry_delay

            while retry_count <= self.max_retries:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url,
                            headers=headers,
                            params=params,
                            json=data,
                            proxy=proxy,
                            timeout=aiohttp.ClientTimeout(total=60)  # 增加超时时间到60秒
                        ) as response:
                            response_text = await response.text()

                            if response.status == 200:
                                result = json.loads(response_text)
                                candidates = result.get("candidates", [])
                                if candidates and len(candidates) > 0:
                                    content = candidates[0].get("content", {})
                                    parts = content.get("parts", [])

                                    for part in parts:
                                        if "text" in part and part["text"]:
                                            return part["text"]

                                return prompt  # 如果无法解析响应，返回原始提示词
                            else:
                                logger.error(f"增强融图提示词API调用失败 (状态码: {response.status}): {response_text}")

                                # 如果是API密钥错误，尝试切换密钥
                                if response.status == 400 and "API key not valid" in response_text:
                                    logger.warning("API密钥无效，尝试切换密钥")
                                    # 标记当前密钥出错，并获取新密钥
                                    new_api_key = self.mark_api_key_error(api_key, session_id)
                                    if new_api_key and new_api_key != api_key:
                                        # 更新请求参数中的API密钥
                                        api_key = new_api_key
                                        params["key"] = api_key
                                        logger.info("已切换到新的API密钥")
                                        # 不增加重试计数，直接使用新密钥重试
                                        continue

                                # 检查是否是可重试的错误
                                if response.status in [429, 500, 502, 503, 504]:
                                    retry_count += 1
                                    if retry_count <= self.max_retries:
                                        logger.info(f"第 {retry_count} 次重试增强融图提示词，等待 {retry_delay} 秒")
                                        await asyncio.sleep(retry_delay)
                                        # 指数退避策略
                                        retry_delay = min(retry_delay * 2, self.max_retry_delay)
                                        continue

                                return prompt  # 返回原始提示词
                except Exception as e:
                    logger.error(f"增强融图提示词异常: {str(e)}")

                    retry_count += 1
                    if retry_count <= self.max_retries:
                        logger.info(f"第 {retry_count} 次重试增强融图提示词，等待 {retry_delay} 秒")
                        await asyncio.sleep(retry_delay)
                        # 指数退避策略
                        retry_delay = min(retry_delay * 2, self.max_retry_delay)
                        continue

                    return prompt  # 返回原始提示词

                # 如果成功，跳出循环
                break

            return prompt  # 如果所有重试都失败，返回原始提示词
        except Exception as e:
            logger.error(f"增强融图提示词失败: {str(e)}")
            logger.error(traceback.format_exc())
            return prompt  # 返回原始提示词

    async def _handle_merge_images(self, bot: WechatAPIClient, message: dict, prompt: str, image_list: List[bytes]):
        """处理融图请求

        Args:
            bot: 微信API客户端
            message: 消息数据
            prompt: 融图提示词
            image_list: 图片数据列表
        """
        try:
            # 获取会话信息
            chat_id = message.get("chat_id", "")
            user_id = message.get("user_id", "")
            from_wxid = message.get("FromWxid", "")
            sender_wxid = message.get("SenderWxid", "")
            conversation_key = f"{chat_id}_{user_id}"

            logger.info(f"开始处理融图请求，提示词: {prompt}, 图片数量: {len(image_list)}")

            # 增强提示词，使用专门的融图提示词增强
            enhanced_prompt = await self._enhance_merge_prompt(prompt)
            prompt = enhanced_prompt

            # 压缩图片以减小请求体大小
            compressed_images = []
            for i, img_data in enumerate(image_list):
                # 压缩图片，使用高质量设置
                compressed_img = await self._compress_image(img_data, max_size=1200, quality=90)
                compressed_images.append(compressed_img)

            # 发送提示消息
            await bot.send_text_message(chat_id, "正在处理融图请求，请稍候...")

            # 调用API生成融合图片
            image_data, response_text = await self._generate_image_with_multiple_images(prompt, compressed_images)

            if image_data:
                # 保存图片到本地
                image_path = os.path.join(self.save_dir, f"gemini_merge_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                with open(image_path, "wb") as f:
                    f.write(image_data)

                # 保存最后生成的图片路径
                self.last_images[conversation_key] = image_path

                # 保存到图片缓存，确保后续可以编辑
                if from_wxid and sender_wxid:
                    self._save_image_to_cache(from_wxid, sender_wxid, image_data)
                    logger.info(f"已将融合图片保存到缓存，大小: {len(image_data)} 字节")

                # 发送文本和图片
                if response_text:
                    # 清理文本格式
                    cleaned_text = self._clean_response_text(response_text)
                    # 尝试使用from_wxid而不是chat_id
                    if from_wxid:
                        await bot.send_text_message(from_wxid, cleaned_text)
                        logger.info(f"使用from_wxid发送融图文本响应: {cleaned_text[:100]}...")
                    else:
                        await bot.send_text_message(chat_id, cleaned_text)
                        logger.info(f"使用chat_id发送融图文本响应: {cleaned_text[:100]}...")

                # 发送图片
                with open(image_path, "rb") as f:
                    # 尝试使用from_wxid而不是chat_id
                    if from_wxid:
                        await bot.send_image_message(from_wxid, f.read())
                        logger.info(f"使用from_wxid发送融合图片，路径: {image_path}")
                    else:
                        await bot.send_image_message(chat_id, f.read())
                        logger.info(f"使用chat_id发送融合图片，路径: {image_path}")

                # 返回成功信息
                return True
            else:
                # 如果没有生成图片，发送错误消息
                error_msg = "融图失败，请稍后再试或修改提示词"
                if response_text:
                    error_msg = response_text
                # 尝试使用from_wxid而不是chat_id
                if from_wxid:
                    await bot.send_text_message(from_wxid, error_msg)
                    logger.info(f"使用from_wxid发送融图失败消息")
                else:
                    await bot.send_text_message(chat_id, error_msg)
                    logger.info(f"使用chat_id发送融图失败消息")
                logger.error(f"融图失败，未生成有效图片数据，响应文本: {response_text}")
                return False
        except Exception as e:
            logger.error(f"处理融图请求异常: {str(e)}")
            logger.error(traceback.format_exc())
            # 尝试使用from_wxid而不是chat_id
            from_wxid = message.get("FromWxid", "")
            chat_id = message.get("chat_id", "")
            if from_wxid:
                await bot.send_text_message(from_wxid, f"融图失败: {str(e)}")
                logger.info(f"使用from_wxid发送融图异常消息")
            else:
                await bot.send_text_message(chat_id, f"融图失败: {str(e)}")
                logger.info(f"使用chat_id发送融图异常消息")
            return False

    async def _handle_reverse_image(self, bot: WechatAPIClient, message: dict, image_data: bytes):
        """处理反向提示词生成请求

        Args:
            bot: 微信API客户端
            message: 消息数据
            image_data: 图片数据
        """
        try:
            # 获取会话信息
            chat_id = message.get("chat_id", "")
            user_id = message.get("user_id", "")
            from_wxid = message.get("FromWxid", "")
            sender_wxid = message.get("SenderWxid", "")
            conversation_key = f"{chat_id}_{user_id}"

            # 发送提示消息
            # 尝试使用from_wxid而不是chat_id
            if from_wxid:
                await bot.send_text_message(from_wxid, "正在分析图片，生成提示词，请稍候...")
                logger.info(f"使用from_wxid发送反向提示词生成提示消息")
            else:
                await bot.send_text_message(chat_id, "正在分析图片，生成提示词，请稍候...")
                logger.info(f"使用chat_id发送反向提示词生成提示消息")

            # 调用反向提示词生成
            result = await self._reverse_image(image_data)

            if result:
                # 将结果拆分成三部分
                parts = self._split_reverse_result(result)

                # 确定发送目标
                target_id = from_wxid if from_wxid else chat_id

                # 分三次发送结果，每次之间添加随机间隔
                for i, part in enumerate(parts):
                    if part.strip():  # 确保部分内容不为空
                        # 发送当前部分
                        await bot.send_text_message(target_id, part)
                        logger.info(f"发送反向提示词生成结果第 {i+1}/{len(parts)} 部分，长度: {len(part)}")

                        # 如果不是最后一部分，添加随机间隔
                        if i < len(parts) - 1:
                            # 随机间隔1-3秒
                            interval = random.uniform(1, 3)
                            logger.info(f"添加随机间隔 {interval:.2f} 秒")
                            await asyncio.sleep(interval)
            else:
                # 发送错误消息
                # 尝试使用from_wxid而不是chat_id
                if from_wxid:
                    await bot.send_text_message(from_wxid, "无法生成提示词，请稍后再试或尝试其他图片")
                    logger.info(f"使用from_wxid发送反向提示词生成失败消息")
                else:
                    await bot.send_text_message(chat_id, "无法生成提示词，请稍后再试或尝试其他图片")
                    logger.info(f"使用chat_id发送反向提示词生成失败消息")
        except Exception as e:
            logger.error(f"处理反向提示词生成请求异常: {str(e)}")
            logger.error(traceback.format_exc())
            # 尝试使用from_wxid而不是chat_id
            from_wxid = message.get("FromWxid", "")
            chat_id = message.get("chat_id", "")
            if from_wxid:
                await bot.send_text_message(from_wxid, f"生成提示词失败: {str(e)}")
                logger.info(f"使用from_wxid发送反向提示词生成异常消息")
            else:
                await bot.send_text_message(chat_id, f"生成提示词失败: {str(e)}")
                logger.info(f"使用chat_id发送反向提示词生成异常消息")

    def _split_reverse_result(self, result: str) -> List[str]:
        """将反向提示词结果拆分成三部分：
        1. 前面的分析部分
        2. 中文提示词部分
        3. 英文提示词部分

        Args:
            result: 完整的反向提示词结果

        Returns:
            List[str]: 拆分后的三部分内容
        """
        # 初始化三个部分
        analysis_part = ""
        chinese_prompt_part = ""
        english_prompt_part = ""

        # 查找中文提示词部分的开始位置
        chinese_patterns = [
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?中文提示词(?:\*\*)?[:：]',
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?详细中文提示词(?:\*\*)?[:：]',
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?优化中文提示词(?:\*\*)?[:：]',
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?建议中文提示词(?:\*\*)?[:：]',
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?推荐中文提示词(?:\*\*)?[:：]',
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?中文关键词(?:\*\*)?[:：]',
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?中文描述(?:\*\*)?[:：]'
        ]

        # 查找英文提示词部分的开始位置
        english_patterns = [
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?英文提示词(?:\*\*)?[:：]',
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?详细英文提示词(?:\*\*)?[:：]',
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?优化英文提示词(?:\*\*)?[:：]',
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?建议英文提示词(?:\*\*)?[:：]',
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?推荐英文提示词(?:\*\*)?[:：]',
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?英文关键词(?:\*\*)?[:：]',
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?英文描述(?:\*\*)?[:：]',
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?English Prompt(?:\*\*)?[:：]',
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?Detailed English Prompt(?:\*\*)?[:：]',
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?Optimized English Prompt(?:\*\*)?[:：]',
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?Suggested English Prompt(?:\*\*)?[:：]',
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?Recommended English Prompt(?:\*\*)?[:：]',
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?English Keywords(?:\*\*)?[:：]',
            r'(?:^|\n)(?:#+ *)?(?:\*\*)?English Description(?:\*\*)?[:：]'
        ]

        # 查找中文提示词部分
        chinese_start = -1
        for pattern in chinese_patterns:
            match = re.search(pattern, result)
            if match:
                chinese_start = match.start()
                break

        # 查找英文提示词部分
        english_start = -1
        for pattern in english_patterns:
            match = re.search(pattern, result)
            if match:
                english_start = match.start()
                break

        # 根据找到的位置拆分内容
        if chinese_start >= 0 and english_start >= 0:
            # 两个部分都找到了
            if chinese_start < english_start:
                # 中文在前，英文在后
                analysis_part = result[:chinese_start].strip()
                chinese_prompt_part = result[chinese_start:english_start].strip()
                english_prompt_part = result[english_start:].strip()
            else:
                # 英文在前，中文在后
                analysis_part = result[:english_start].strip()
                english_prompt_part = result[english_start:chinese_start].strip()
                chinese_prompt_part = result[chinese_start:].strip()
        elif chinese_start >= 0:
            # 只找到中文部分
            analysis_part = result[:chinese_start].strip()
            chinese_prompt_part = result[chinese_start:].strip()

            # 尝试在中文部分中查找英文提示词的标记
            for pattern in english_patterns:
                match = re.search(pattern, chinese_prompt_part)
                if match:
                    english_start = match.start()
                    english_prompt_part = chinese_prompt_part[english_start:].strip()
                    chinese_prompt_part = chinese_prompt_part[:english_start].strip()
                    break
        elif english_start >= 0:
            # 只找到英文部分
            analysis_part = result[:english_start].strip()
            english_prompt_part = result[english_start:].strip()

            # 尝试在英文部分中查找中文提示词的标记
            for pattern in chinese_patterns:
                match = re.search(pattern, english_prompt_part)
                if match:
                    chinese_start = match.start()
                    chinese_prompt_part = english_prompt_part[chinese_start:].strip()
                    english_prompt_part = english_prompt_part[:chinese_start].strip()
                    break
        else:
            # 没有找到明确的中文或英文提示词部分
            # 尝试查找可能的提示词部分
            prompt_patterns = [
                r'(?:^|\n)(?:#+ *)?(?:\*\*)?提示词(?:\*\*)?[:：]',
                r'(?:^|\n)(?:#+ *)?(?:\*\*)?关键词(?:\*\*)?[:：]',
                r'(?:^|\n)(?:#+ *)?(?:\*\*)?Prompt(?:\*\*)?[:：]',
                r'(?:^|\n)(?:#+ *)?(?:\*\*)?Keywords(?:\*\*)?[:：]'
            ]

            prompt_start = -1
            for pattern in prompt_patterns:
                match = re.search(pattern, result)
                if match:
                    prompt_start = match.start()
                    break

            if prompt_start >= 0:
                # 找到了提示词部分
                analysis_part = result[:prompt_start].strip()
                prompt_part = result[prompt_start:].strip()

                # 尝试将提示词部分分为中文和英文
                # 简单的启发式方法：如果包含大量英文字符，则认为是英文部分
                english_char_ratio = len(re.findall(r'[a-zA-Z]', prompt_part)) / len(prompt_part) if prompt_part else 0

                if english_char_ratio > 0.5:
                    # 主要是英文
                    english_prompt_part = prompt_part
                else:
                    # 主要是中文
                    chinese_prompt_part = prompt_part
            else:
                # 没有找到任何提示词部分，将整个文本作为分析部分
                analysis_part = result

        # 如果某部分为空，尝试从其他部分中提取内容
        if not analysis_part and (chinese_prompt_part or english_prompt_part):
            # 如果分析部分为空但提示词部分不为空，可能是整个文本都是提示词
            # 在这种情况下，我们可以将提示词部分的前半部分作为分析部分
            if chinese_prompt_part:
                half_point = len(chinese_prompt_part) // 2
                analysis_part = chinese_prompt_part[:half_point].strip()
                chinese_prompt_part = chinese_prompt_part[half_point:].strip()
            elif english_prompt_part:
                half_point = len(english_prompt_part) // 2
                analysis_part = english_prompt_part[:half_point].strip()
                english_prompt_part = english_prompt_part[half_point:].strip()

        # 如果中文和英文提示词部分都为空，尝试将分析部分拆分
        if not chinese_prompt_part and not english_prompt_part and analysis_part:
            # 将分析部分的后三分之一作为英文提示词部分
            third_point = len(analysis_part) * 2 // 3
            english_prompt_part = analysis_part[third_point:].strip()
            analysis_part = analysis_part[:third_point].strip()

        # 记录拆分结果
        logger.info(f"反向提示词拆分结果: 分析部分 {len(analysis_part)} 字符, 中文提示词部分 {len(chinese_prompt_part)} 字符, 英文提示词部分 {len(english_prompt_part)} 字符")

        # 返回三个部分
        return [analysis_part, chinese_prompt_part, english_prompt_part]

    async def _compress_image(self, image_data: bytes, max_size: int = 1200, quality: int = 90) -> bytes:
        """压缩图片

        Args:
            image_data: 原始图片数据
            max_size: 最大尺寸
            quality: 压缩质量

        Returns:
            bytes: 压缩后的图片数据
        """
        try:
            # 打开图片
            image = Image.open(BytesIO(image_data))

            # 调整图片大小
            width, height = image.size
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                image = image.resize((new_width, new_height), Image.LANCZOS)

            # 转换为RGB模式（如果是RGBA）
            if image.mode == "RGBA":
                image = image.convert("RGB")

            # 保存为JPEG格式
            output = BytesIO()
            image.save(output, format="JPEG", quality=quality)
            compressed_data = output.getvalue()

            return compressed_data
        except Exception as e:
            logger.error(f"压缩图片失败: {str(e)}")
            logger.error(traceback.format_exc())
            return image_data  # 如果压缩失败，返回原始图片数据

    async def _generate_image_with_multiple_images(self, prompt: str, image_list: List[bytes]) -> Tuple[Optional[bytes], Optional[str]]:
        """使用多张图片生成新图片

        Args:
            prompt: 提示词
            image_list: 图片数据列表

        Returns:
            Tuple[Optional[bytes], Optional[str]]: 生成的图片数据和文本响应
        """
        url = f"{self.base_url}/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent"
        headers = {
            "Content-Type": "application/json",
        }

        params = {
            "key": self.api_key
        }

        # 构建请求体
        parts = [{"text": prompt}]

        # 添加所有图片
        for img_data in image_list:
            img_base64 = base64.b64encode(img_data).decode("utf-8")
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": img_base64
                }
            })

        # 构建完整请求
        data = {
            "contents": [{
                "role": "user",
                "parts": parts
            }],
            "generationConfig": {
                "responseModalities": ["Text", "Image"]
            }
        }

        # 创建代理配置
        proxy = None
        if self.enable_proxy and self.proxy_url:
            proxy = self.proxy_url

        # 使用重试机制
        retry_count = 0
        retry_delay = self.initial_retry_delay

        while retry_count <= self.max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        headers=headers,
                        params=params,
                        json=data,
                        proxy=proxy,
                        timeout=aiohttp.ClientTimeout(total=60)  # 增加超时时间到60秒
                    ) as response:
                        response_text = await response.text()

                        if response.status == 200:
                            result = json.loads(response_text)

                            # 提取响应
                            candidates = result.get("candidates", [])
                            if candidates and len(candidates) > 0:
                                content = candidates[0].get("content", {})
                                parts = content.get("parts", [])

                                # 处理文本和图片响应
                                text_response = None
                                image_data = None

                                for part in parts:
                                    # 处理文本部分
                                    if "text" in part and part["text"]:
                                        text_response = part["text"]

                                    # 处理图片部分
                                    elif "inlineData" in part:
                                        inline_data = part.get("inlineData", {})
                                        if inline_data and "data" in inline_data:
                                            # 解码图片数据
                                            image_data = base64.b64decode(inline_data["data"])

                                if not image_data:
                                    # 如果没有生成图像，尝试使用英文提示词重试
                                    logger.info("未获取到图像，尝试使用英文提示词重试...")
                                    english_prompt = f"Please merge these images. {prompt}. Make sure to include the generated image in your response."

                                    # 更新请求体中的提示词
                                    data["contents"][0]["parts"][0]["text"] = english_prompt

                                    # 重新发送请求
                                    async with session.post(
                                        url,
                                        headers=headers,
                                        params=params,
                                        json=data,
                                        proxy=proxy,
                                        timeout=aiohttp.ClientTimeout(total=60)  # 增加超时时间到60秒
                                    ) as retry_response:
                                        retry_response_text = await retry_response.text()

                                        if retry_response.status == 200:
                                            retry_result = json.loads(retry_response_text)
                                            retry_candidates = retry_result.get("candidates", [])
                                            if retry_candidates and len(retry_candidates) > 0:
                                                retry_content = retry_candidates[0].get("content", {})
                                                retry_parts = retry_content.get("parts", [])

                                                for retry_part in retry_parts:
                                                    # 处理文本部分
                                                    if "text" in retry_part and retry_part["text"]:
                                                        text_response = retry_part["text"]

                                                    # 处理图片部分
                                                    elif "inlineData" in retry_part:
                                                        retry_inline_data = retry_part.get("inlineData", {})
                                                        if retry_inline_data and "data" in retry_inline_data:
                                                            # 解码图片数据
                                                            image_data = base64.b64decode(retry_inline_data["data"])

                                return image_data, text_response
                            else:
                                # 记录响应摘要，避免输出大量base64数据
                                response_summary = self._get_response_summary(response_text)
                                logger.error(f"API响应不包含候选结果: {response_summary}")

                                # 检查是否是可重试的错误
                                retry_count += 1
                                if retry_count <= self.max_retries:
                                    logger.info(f"第 {retry_count} 次重试生成融合图片，等待 {retry_delay} 秒")
                                    await asyncio.sleep(retry_delay)
                                    # 指数退避策略
                                    retry_delay = min(retry_delay * 2, self.max_retry_delay)
                                    continue

                                return None, "API响应不包含候选结果，请稍后再试"
                        else:
                            # 记录响应摘要，避免输出大量base64数据
                            response_summary = self._get_response_summary(response_text)
                            logger.error(f"融合图片API调用失败 (状态码: {response.status}): {response_summary}")

                            # 检查是否是可重试的错误
                            if response.status in [429, 500, 502, 503, 504]:
                                retry_count += 1
                                if retry_count <= self.max_retries:
                                    logger.info(f"第 {retry_count} 次重试生成融合图片，等待 {retry_delay} 秒")
                                    await asyncio.sleep(retry_delay)
                                    # 指数退避策略
                                    retry_delay = min(retry_delay * 2, self.max_retry_delay)
                                    continue

                            return None, f"融合图片API调用失败 (状态码: {response.status})"
            except Exception as e:
                logger.error(f"生成融合图片异常: {str(e)}")

                retry_count += 1
                if retry_count <= self.max_retries:
                    logger.info(f"第 {retry_count} 次重试生成融合图片，等待 {retry_delay} 秒")
                    await asyncio.sleep(retry_delay)
                    # 指数退避策略
                    retry_delay = min(retry_delay * 2, self.max_retry_delay)
                    continue

                return None, f"生成融合图片失败: {str(e)}"

            # 如果成功，跳出循环
            break

        return None, "生成融合图片失败，请稍后再试"

    async def _generate_image(self, prompt: str, conversation_history: List[Dict] = None, is_continuous_dialogue: bool = False) -> Tuple[List[bytes], List[str]]:
        """调用Gemini API生成图片，返回图片数据列表和文本响应列表

        Args:
            prompt: 提示词
            conversation_history: 对话历史
            is_continuous_dialogue: 是否是连续对话模式

        Returns:
            Tuple[List[bytes], List[str]]: 图片数据列表和文本响应列表
        """
        url = f"{self.base_url}/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent"
        # 检查URL格式是否正确
        if not url.startswith("http"):
            logger.warning(f"URL格式可能不正确: {url}")
            # 尝试修复URL格式
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent"
        headers = {
            "Content-Type": "application/json",
        }

        params = {
            "key": self.api_key
        }

        # 检查是否是多图文请求
        is_multi_image = self._is_multi_image_request(prompt)

        # 增强提示词，如果启用了提示词增强且不是连续对话模式
        if self.enhance_prompt and not is_continuous_dialogue:
            # 只在新对话中增强提示词，不在连续对话中增强
            if is_multi_image:
                # 如果是多图文请求，使用多图文提示词增强
                enhanced_prompt = await self._enhance_multi_image_prompt(prompt)
            else:
                # 如果是普通请求，使用标准提示词增强
                enhanced_prompt = await self._enhance_prompt(prompt)

            prompt = enhanced_prompt
        else:
            # 在连续对话中，直接使用原始提示词
            logger.info(f"连续对话模式，不增强提示词，直接使用原始提示词: {prompt}")

        # 构建请求数据
        if conversation_history and len(conversation_history) > 0:
            # 有会话历史，构建上下文
            # 需要处理会话历史中的图片格式
            processed_history = []
            for msg in conversation_history:
                # 转换角色名称，确保使用 "user" 或 "model"
                role = msg["role"]
                if role == "assistant":
                    role = "model"

                processed_msg = {"role": role, "parts": []}
                for part in msg["parts"]:
                    if "text" in part:
                        processed_msg["parts"].append({"text": part["text"]})
                    elif "image_url" in part:
                        # 需要读取图片并转换为inlineData格式
                        try:
                            with open(part["image_url"], "rb") as f:
                                image_data = f.read()
                                image_base64 = base64.b64encode(image_data).decode("utf-8")
                                processed_msg["parts"].append({
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": image_base64
                                    }
                                })
                        except Exception as e:
                            logger.error(f"处理历史图片失败: {e}")
                            # 跳过这个图片
                processed_history.append(processed_msg)

            data = {
                "contents": processed_history + [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generation_config": {
                    "response_modalities": ["Text", "Image"]
                }
            }
        else:
            # 无会话历史，直接使用提示
            data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generation_config": {
                    "response_modalities": ["Text", "Image"]
                }
            }

        # 创建代理配置
        proxy = None
        if self.enable_proxy and self.proxy_url:
            proxy = self.proxy_url

        try:
            # 创建客户端会话，设置代理（如果启用）
            async with aiohttp.ClientSession() as session:
                try:
                    # 使用代理发送请求
                    async with session.post(
                        url,
                        headers=headers,
                        params=params,
                        json=data,
                        proxy=proxy,
                        timeout=aiohttp.ClientTimeout(total=60)  # 增加超时时间到60秒
                    ) as response:
                        response_text = await response.text()


                        if response.status == 200:
                            try:
                                result = json.loads(response_text)

                                # 记录响应状态
                                logger.info(f"Gemini API响应成功")

                                # 提取响应
                                candidates = result.get("candidates", [])
                                if candidates and len(candidates) > 0:
                                    content = candidates[0].get("content", {})
                                    parts = content.get("parts", [])

                                    # 处理文本和图片响应，保持原始顺序
                                    parts_list = []
                                    image_count = 0

                                    # 检查是否是多图文请求
                                    if is_multi_image:
                                        logger.info(f"检测到多图文请求，开始处理分镜脚本")
                                        logger.info(f"原始提示词: {prompt[:200]}..." if len(prompt) > 200 else f"原始提示词: {prompt}")

                                        # 从分镜脚本中提取故事内容和中文提示词
                                        story_contents = self._extract_story_content(prompt)
                                        chinese_prompts = self._extract_chinese_prompt(prompt)

                                        logger.info(f"从分镜脚本中提取到 {len(story_contents)} 个故事内容和 {len(chinese_prompts)} 个中文提示词")

                                        # 记录每个故事内容的前50个字符，便于调试
                                        for i, content in enumerate(story_contents):
                                            logger.info(f"故事内容 {i+1}: {content[:50]}..." if len(content) > 50 else f"故事内容 {i+1}: {content}")

                                        # 如果成功提取到中文提示词，使用这些提示词生成图片
                                        if chinese_prompts:
                                            # 首先从 API 响应中提取所有图片
                                            all_images = []
                                            for part in parts:
                                                if "inlineData" in part:
                                                    inline_data = part.get("inlineData", {})
                                                    if inline_data and "data" in inline_data:
                                                        # 解码图片数据
                                                        image_data = base64.b64decode(inline_data["data"])
                                                        all_images.append(image_data)
                                                        logger.info(f"从 API 响应中提取到第 {len(all_images)} 张图片，大小: {len(image_data)} 字节")

                                            logger.info(f"从 API 响应中总共提取到 {len(all_images)} 张图片")

                                            # 先添加整体的文本描述
                                            if len(parts) > 0 and "text" in parts[0] and parts[0]["text"]:
                                                parts_list.append({"type": "text", "content": parts[0]["text"]})

                                            # 为每个中文提示词/故事内容添加图片
                                            for i in range(max(len(chinese_prompts), len(story_contents))):
                                                # 如果有对应的故事内容，添加到parts_list
                                                if i < len(story_contents):
                                                    parts_list.append({"type": "text", "content": story_contents[i]})

                                                # 如果有对应的图片，使用它
                                                if i < len(all_images):
                                                    parts_list.append({"type": "image", "content": all_images[i]})
                                                    image_count += 1
                                                    logger.info(f"为第 {i+1} 个故事内容使用 API 响应中的图片")
                                                elif i < len(chinese_prompts):
                                                    # 如果没有对应的图片，单独生成一张
                                                    logger.info(f"为第 {i+1} 个故事内容单独生成图片，提示词: {chinese_prompts[i][:50]}...")

                                                    # 单独调用API生成图片
                                                    try:
                                                        # 构建请求URL
                                                        single_url = f"{self.base_url}/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent"
                                                        # 检查URL格式是否正确
                                                        if not single_url.startswith("http"):
                                                            logger.warning(f"URL格式可能不正确: {single_url}")
                                                            # 尝试修复URL格式
                                                            single_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent"
                                                        single_headers = {
                                                            "Content-Type": "application/json",
                                                        }
                                                        single_params = {
                                                            "key": self.api_key
                                                        }

                                                        # 构建请求数据
                                                        # 提取第一个场景的人物描述，用于保持一致性
                                                        character_description = ""
                                                        if i > 0 and len(chinese_prompts) > 0:
                                                            # 尝试从第一个场景的提示词中提取人物描述
                                                            first_prompt = chinese_prompts[0]
                                                            logger.info(f"分析第一个场景的提示词以提取人物描述: {first_prompt[:100]}...")

                                                            # 查找主要人物/对象描述部分
                                                            character_markers = [
                                                                "**主要人物/对象描述**",
                                                                "主要人物/对象描述",
                                                                "**主体对象：**",
                                                                "主体对象：",
                                                                "**人物描述：**",
                                                                "人物描述：",
                                                                "**人物特征：**",
                                                                "人物特征：",
                                                                "**角色描述：**",
                                                                "角色描述："
                                                            ]

                                                            for marker in character_markers:
                                                                if marker in first_prompt:
                                                                    logger.info(f"在第一个场景中找到标记: {marker}")
                                                                    parts = first_prompt.split(marker, 1)
                                                                    if len(parts) > 1:
                                                                        # 提取人物描述部分
                                                                        desc_part = parts[1].strip()
                                                                        # 查找下一个标记
                                                                        next_markers = [
                                                                            "**场景：", "**故事内容", "**1. 图片内容概述",
                                                                            "**场景环境：**", "场景环境：", "**背景：**", "背景：",
                                                                            "**气氛：**", "气氛：", "**风格：**", "风格："
                                                                        ]
                                                                        end_pos = len(desc_part)
                                                                        for next_marker in next_markers:
                                                                            pos = desc_part.find(next_marker)
                                                                            if pos != -1 and pos < end_pos:
                                                                                end_pos = pos

                                                                        character_description = desc_part[:end_pos].strip()
                                                                        logger.info(f"从第一个场景提取到人物描述: {character_description[:100]}...")
                                                                        break

                                                            # 如果没有找到明确的人物描述，尝试提取主体对象部分
                                                            if not character_description:
                                                                # 查找主体对象部分
                                                                object_markers = [
                                                                    "**2. 主体对象：**",
                                                                    "**主体对象：**",
                                                                    "主体对象：",
                                                                    "**主体：**",
                                                                    "主体："
                                                                ]

                                                                for marker in object_markers:
                                                                    if marker in first_prompt:
                                                                        logger.info(f"在第一个场景中找到主体对象标记: {marker}")
                                                                        parts = first_prompt.split(marker, 1)
                                                                        if len(parts) > 1:
                                                                            # 提取主体对象部分
                                                                            obj_part = parts[1].strip()
                                                                            # 查找下一个标记
                                                                            next_markers = [
                                                                                "**3. 场景环境：**", "**场景环境：**", "场景环境：",
                                                                                "**背景：**", "背景：", "**气氛：**", "气氛："
                                                                            ]
                                                                            end_pos = len(obj_part)
                                                                            for next_marker in next_markers:
                                                                                pos = obj_part.find(next_marker)
                                                                                if pos != -1 and pos < end_pos:
                                                                                    end_pos = pos

                                                                            character_description = obj_part[:end_pos].strip()
                                                                            logger.info(f"从第一个场景提取到主体对象描述: {character_description[:100]}...")
                                                                            break

                                                            # 如果还是没有找到人物描述，尝试使用整个第一个场景的提示词
                                                            if not character_description and len(first_prompt) > 0:
                                                                # 使用第一个场景的前100个字符作为人物描述
                                                                character_description = f"保持与第一个场景相同的风格和一致性"
                                                                logger.info(f"未找到明确的人物描述，使用通用一致性提示: {character_description}")

                                                        # 如果找到了人物描述，将其添加到当前场景的提示词中
                                                        enhanced_prompt = chinese_prompts[i]
                                                        if character_description:
                                                            # 检查当前提示词中是否已经包含了人物描述
                                                            if character_description not in enhanced_prompt:
                                                                # 在提示词开头添加人物描述
                                                                enhanced_prompt = f"保持与第一个场景相同的人物特征和风格：{character_description}\n\n{enhanced_prompt}"
                                                                logger.info(f"为第 {i+1} 个场景添加了人物描述，确保一致性")

                                                        # 为每个场景使用不同的温度参数，增加多样性
                                                        # 场景索引越大，温度越高，生成的图片越多样
                                                        scene_temperature = min(0.7, 0.4 + i * 0.05)

                                                        # 添加明确的指示，要求生成与前面场景不同的图片
                                                        scene_instruction = f"为第{i+1}个场景生成一张与前面场景不同的图片。"
                                                        if i > 0:
                                                            scene_instruction += "请确保这张图片与前面的图片有明显区别，但保持人物特征一致。"

                                                        # 在提示词中添加场景编号，帮助模型区分不同场景
                                                        final_prompt = f"{scene_instruction}\n\n场景{i+1}：{enhanced_prompt}"

                                                        logger.info(f"为第 {i+1} 个场景使用温度参数: {scene_temperature}")

                                                        single_data = {
                                                            "contents": [
                                                                {
                                                                    "parts": [
                                                                        {
                                                                            "text": final_prompt
                                                                        }
                                                                    ]
                                                                }
                                                            ],
                                                            "generation_config": {
                                                                "response_modalities": ["Text", "Image"],
                                                                "temperature": scene_temperature,
                                                                "topP": 0.95,
                                                                "topK": 64,
                                                                "seed": int(time.time() * 1000) % 1000000 + i * 1000  # 为每个场景使用不同的随机种子
                                                            }
                                                        }

                                                        # 创建代理配置
                                                        single_proxy = None
                                                        if self.enable_proxy and self.proxy_url:
                                                            single_proxy = self.proxy_url

                                                        # 发送请求
                                                        async with aiohttp.ClientSession() as single_session:
                                                            async with single_session.post(
                                                                single_url,
                                                                headers=single_headers,
                                                                params=single_params,
                                                                json=single_data,
                                                                proxy=single_proxy,
                                                                timeout=aiohttp.ClientTimeout(total=60)
                                                            ) as single_response:
                                                                single_response_text = await single_response.text()

                                                                if single_response.status == 200:
                                                                    single_result = json.loads(single_response_text)

                                                                    single_candidates = single_result.get("candidates", [])

                                                                    if single_candidates and len(single_candidates) > 0:
                                                                        single_content = single_candidates[0].get("content", {})
                                                                        single_parts = single_content.get("parts", [])

                                                                        # 查找图片数据
                                                                        single_image_data = None
                                                                        for single_part in single_parts:
                                                                            if "inlineData" in single_part:
                                                                                single_inline_data = single_part.get("inlineData", {})
                                                                                if single_inline_data and "data" in single_inline_data:
                                                                                    # 解码图片数据
                                                                                    single_image_data = base64.b64decode(single_inline_data["data"])
                                                                                    break

                                                                        if single_image_data:
                                                                            # 生成了图片，添加到结果列表中
                                                                            parts_list.append({"type": "image", "content": single_image_data})
                                                                            image_count += 1

                                                                            # 记录详细的成功信息
                                                                            logger.info(f"为第 {i+1} 个故事内容单独生成图片成功，大小: {len(single_image_data)} 字节")

                                                                            # 保存图片到临时文件进行调试
                                                                            try:
                                                                                debug_image_path = os.path.join(self.save_dir, f"debug_scene_{i+1}_{int(time.time())}.png")
                                                                                with open(debug_image_path, "wb") as f:
                                                                                    f.write(single_image_data)
                                                                                logger.info(f"已保存第 {i+1} 个场景的调试图片到: {debug_image_path}")
                                                                            except Exception as e:
                                                                                logger.error(f"保存调试图片失败: {e}")
                                                                        else:
                                                                            # 生成图片失败，记录详细的错误信息
                                                                            logger.warning(f"未能为第 {i+1} 个故事内容单独生成图片，API 响应中没有图片数据")
                                                                            logger.warning(f"尝试为第 {i+1} 个场景生成图片的提示词: {final_prompt[:200]}...")
                                                                    else:
                                                                        logger.warning(f"未能为第 {i+1} 个故事内容单独生成图片，API 响应中没有候选结果")
                                                                else:
                                                                    logger.error(f"单独生成图片 API 调用失败 (状态码: {single_response.status}): {single_response_text[:200]}...")
                                                    except Exception as e:
                                                        logger.error(f"单独生成图片异常: {str(e)}")
                                                        logger.error(traceback.format_exc())
                                        else:
                                            # 如果没有提取到中文提示词，使用常规处理方式
                                            for part in parts:
                                                # 处理文本部分
                                                if "text" in part and part["text"]:
                                                    parts_list.append({"type": "text", "content": part["text"]})

                                                # 处理图片部分
                                                if "inlineData" in part:
                                                    inline_data = part.get("inlineData", {})
                                                    if inline_data and "data" in inline_data:
                                                        # 解码图片数据
                                                        image_data = base64.b64decode(inline_data["data"])
                                                        parts_list.append({"type": "image", "content": image_data})
                                                        image_count += 1
                                    else:
                                        # 常规处理方式
                                        for part in parts:
                                            # 处理文本部分
                                            if "text" in part and part["text"]:
                                                parts_list.append({"type": "text", "content": part["text"]})

                                            # 处理图片部分
                                            if "inlineData" in part:
                                                inline_data = part.get("inlineData", {})
                                                if inline_data and "data" in inline_data:
                                                    # 解码图片数据
                                                    image_data = base64.b64decode(inline_data["data"])
                                                    parts_list.append({"type": "image", "content": image_data})
                                                    image_count += 1

                                    if image_count == 0:
                                        # 记录响应摘要，避免输出大量base64数据
                                        response_summary = self._get_response_summary(response_text)
                                        logger.error(f"API响应中没有找到图片数据: {response_summary}")
                                        return parts_list, 0

                                    return parts_list, image_count

                                # 记录响应摘要，避免输出大量base64数据
                                response_summary = self._get_response_summary(response_text)
                                logger.error(f"未找到生成的图片数据: {response_summary}")
                                return [], 0
                            except json.JSONDecodeError as je:
                                logger.error(f"解析JSON响应失败: {je}")
                                logger.error(f"响应内容: {response_text[:1000]}...")  # 记录部分响应内容
                                return [], 0
                        else:
                            logger.error(f"Gemini API调用失败 (状态码: {response.status}): {response_text}")
                            return [], 0
                except aiohttp.ClientError as ce:
                    logger.error(f"API请求客户端错误: {ce}")
                    return [], 0
        except Exception as e:
            logger.error(f"API调用异常: {str(e)}")
            logger.error(traceback.format_exc())
            return [], 0

    async def _edit_image(self, prompt: str, image_data_input: Union[bytes, List[bytes]], conversation_history: List[Dict] = None, is_continuous_dialogue: bool = False) -> Tuple[List[Optional[bytes]], List[Optional[str]]]:
        """调用Gemini API编辑图片，返回处理后的图片数据和文本响应

        Args:
            prompt: 编辑图片的文本提示
            image_data_input: 要编辑的图片数据，可以是单个bytes对象或bytes列表
            conversation_history: 会话历史记录
            is_continuous_dialogue: 是否是连续对话模式

        返回值:
            Tuple[List[Optional[bytes]], List[Optional[str]]]: 编辑后的图片数据列表和文本响应列表，
            按照API返回的顺序排列，以支持图文混排内容的处理。
        """
        # 增强编辑提示词，如果启用了提示词增强且不是连续对话模式
        if self.enhance_prompt and not is_continuous_dialogue:
            # 只在新对话中增强提示词，不在连续对话中增强
            enhanced_prompt = await self._enhance_edit_prompt(prompt)
            logger.info(f"原始编辑提示词: {prompt}")
            logger.info(f"增强后的编辑提示词: {enhanced_prompt}")
            prompt = enhanced_prompt
        else:
            # 在连续对话中，直接使用原始提示词
            logger.info(f"连续对话模式，不增强提示词，直接使用原始提示词: {prompt}")

        # 直接使用提示词，不添加额外前缀
        edit_prompt = prompt

        url = f"{self.base_url}/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent"
        # 检查URL格式是否正确
        if not url.startswith("http"):
            logger.warning(f"URL格式可能不正确: {url}")
            # 尝试修复URL格式
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent"
        headers = {
            "Content-Type": "application/json",
        }

        # 获取会话ID
        session_id = f"edit_{uuid.uuid4().hex[:8]}"  # 为编辑图片生成一个唯一的会话ID

        # 获取API密钥
        api_key = self.get_api_key_for_session(session_id)

        params = {
            "key": api_key
        }

        # 确保image_data_input是列表形式
        if isinstance(image_data_input, bytes):
            image_datas = [image_data_input]
        else:
            image_datas = image_data_input

        # 验证图片数据
        if not image_datas or len(image_datas) == 0:
            logger.error("没有提供图片数据")
            return [], []

        # 将图片数据转换为Base64编码
        image_base64 = base64.b64encode(image_datas[0]).decode("utf-8")  # 使用第一张图片

        # 构建请求数据
        if conversation_history and len(conversation_history) > 0:
            # 有会话历史，构建上下文
            # 需要处理会话历史中的图片格式
            processed_history = []
            for msg in conversation_history:
                # 转换角色名称，确保使用 "user" 或 "model"
                role = msg["role"]
                if role == "assistant":
                    role = "model"

                processed_msg = {"role": role, "parts": []}
                for part in msg["parts"]:
                    if "text" in part:
                        processed_msg["parts"].append({"text": part["text"]})
                    elif "image_url" in part:
                        # 需要读取图片并转换为inlineData格式
                        try:
                            with open(part["image_url"], "rb") as f:
                                img_data = f.read()
                                img_base64 = base64.b64encode(img_data).decode("utf-8")
                                processed_msg["parts"].append({
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": img_base64
                                    }
                                })
                        except Exception as e:
                            logger.error(f"处理历史图片失败: {e}")
                            # 跳过这个图片
                processed_history.append(processed_msg)

            data = {
                "contents": processed_history + [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": edit_prompt
                            },
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ],
                "generation_config": {
                    "response_modalities": ["Text", "Image"],
                    "max_output_tokens": 8192,  # 增加输出令牌数量限制
                    "temperature": 0.4  # 降低温度，减少随机性
                }
            }
        else:
            # 无会话历史，直接使用提示和图片
            data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": edit_prompt
                            },
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ],
                "generation_config": {
                    "response_modalities": ["Text", "Image"]
                }
            }

        logger.info(f"构建编辑图片请求数据: 提示词长度={len(edit_prompt)}, 图片大小={len(image_base64)}字节")

        # 记录请求数据的关键部分
        logger.info(f"API请求URL: {url}")
        logger.info(f"API请求参数: {params}")
        # 记录请求数据的结构，但不记录实际的base64数据
        request_data_log = copy.deepcopy(data)
        if "contents" in request_data_log:
            for content in request_data_log["contents"]:
                if "parts" in content:
                    for part in content["parts"]:
                        if "inlineData" in part and "data" in part["inlineData"]:
                            part["inlineData"]["data"] = f"[BASE64_DATA_{len(part['inlineData']['data'])}bytes]"  # 替换为长度信息
        logger.info(f"API请求数据结构: {json.dumps(request_data_log, ensure_ascii=False)[:1000]}...")

        # 创建代理配置
        proxy = None
        if self.enable_proxy and self.proxy_url:
            proxy = self.proxy_url
            logger.info(f"使用代理: {self.proxy_url}")

        # 初始化重试参数
        max_retries = 3
        retry_count = 0
        retry_delay = 1.0  # 初始重试延迟（秒）
        retry_status_codes = [429, 500, 502, 503, 504]  # 需要重试的状态码

        while retry_count <= max_retries:
            try:
                # 创建客户端会话，设置代理（如果启用）
                async with aiohttp.ClientSession() as session:
                    # 使用代理发送请求
                    logger.info(f"开始调用Gemini API编辑图片 (尝试 {retry_count+1}/{max_retries+1})")
                    async with session.post(
                        url,
                        headers=headers,
                        params=params,
                        json=data,
                        proxy=proxy,
                        timeout=aiohttp.ClientTimeout(total=300)  # 增加超时时间到300秒
                    ) as response:
                        response_text = await response.text()
                        logger.info(f"Gemini API响应状态码: {response.status}")

                        if response.status == 200:
                            try:
                                result = json.loads(response_text)

                                # 记录响应内容摘要，避免输出大量base64数据
                                response_summary = self._get_response_summary(response_text)
                                logger.info(f"Gemini API响应内容摘要: {response_summary}")

                                # 检查是否有内容安全问题
                                candidates = result.get("candidates", [])
                                if candidates and len(candidates) > 0:
                                    finish_reason = candidates[0].get("finishReason", "")
                                    if finish_reason == "IMAGE_SAFETY":
                                        logger.warning("Gemini API返回IMAGE_SAFETY，图片内容可能违反安全政策")
                                        # 提取安全评级信息，构建更友好的错误消息
                                        safety_message = "图片内容可能违反安全政策，无法处理该请求。"
                                        try:
                                            if "safetyRatings" in candidates[0]:
                                                safety_ratings = candidates[0]["safetyRatings"]
                                                blocked_categories = []
                                                for rating in safety_ratings:
                                                    if rating.get("blocked", False):
                                                        category = rating.get("category", "未知类别")
                                                        probability = rating.get("probability", "未知")
                                                        blocked_categories.append(f"{category}({probability})")

                                                if blocked_categories:
                                                    safety_message = f"图片内容可能违反安全政策，被拒绝的类别: {', '.join(blocked_categories)}。请修改您的请求。"
                                        except Exception as e:
                                            logger.error(f"解析安全评级信息失败: {e}")

                                        return [], [safety_message]  # 返回空图片列表和友好的错误消息

                                    content = candidates[0].get("content", {})
                                    parts = content.get("parts", [])

                                    # 添加更详细的日志
                                    logger.info(f"API响应包含 {len(parts)} 个部分")

                                    # 检查是否有其他候选结果
                                    if len(candidates) > 1:
                                        logger.info(f"API响应包含多个候选结果: {len(candidates)}")

                                    # 检查是否有完成原因
                                    finish_reason = candidates[0].get("finishReason", "")
                                    if finish_reason:
                                        logger.info(f"API响应的完成原因: {finish_reason}")

                                    # 处理文本和图片响应，以列表形式返回所有部分
                                    text_responses = []
                                    image_datas = []

                                    for i, part in enumerate(parts):
                                        # 处理文本部分
                                        if "text" in part and part["text"]:
                                            text_responses.append(part["text"])
                                            image_datas.append(None)  # 对应位置添加None表示没有图片
                                            logger.info(f"第 {i+1} 部分是文本，内容长度: {len(part['text'])}")

                                        # 处理图片部分
                                        elif "inlineData" in part:
                                            inline_data = part.get("inlineData", {})
                                            if inline_data and "data" in inline_data:
                                                # Base64解码图片数据
                                                img_data = base64.b64decode(inline_data["data"])
                                                # 添加更多日志
                                                logger.info(f"图片数据前20字节: {img_data[:20].hex()}")
                                                # 检查是否是有效的PNG或JPEG文件
                                                if img_data[:8].hex().startswith('89504e47') or img_data[:3].hex().startswith('ffd8ff'):
                                                    logger.info(f"图片数据是有效的PNG或JPEG格式")
                                                else:
                                                    logger.warning(f"图片数据不是标准的PNG或JPEG格式")
                                                # 保存原始图片数据以便调试
                                                debug_path = os.path.join(self.save_dir, f"debug_image_{int(time.time())}_{uuid.uuid4().hex[:8]}.bin")
                                                with open(debug_path, "wb") as f:
                                                    f.write(img_data)
                                                logger.info(f"已保存原始图片数据到: {debug_path}")
                                                image_datas.append(img_data)
                                                text_responses.append(None)  # 对应位置添加None表示没有文本
                                                logger.info(f"第 {i+1} 部分是图片，数据大小: {len(img_data)} 字节")
                                            else:
                                                logger.warning(f"第 {i+1} 部分是图片，但数据为空")
                                        else:
                                            logger.warning(f"第 {i+1} 部分格式未知: {part.keys()}")

                                    valid_images_count = len([img for img in image_datas if img])
                                    valid_texts_count = len([txt for txt in text_responses if txt])
                                    logger.info(f"处理后得到 {valid_images_count} 个有效图片和 {valid_texts_count} 段有效文本")

                                    # 检查是否有可能的截断情况
                                    if len(parts) >= 13 and (valid_images_count + valid_texts_count) >= 13:
                                        logger.warning(f"响应包含 {len(parts)} 个部分，接近API限制，可能存在内容被截断的情况")

                                    if not image_datas or all(img is None for img in image_datas):
                                        logger.error(f"API响应中没有找到图片数据: {result}")
                                        # 检查是否有文本响应，仅返回文本数据
                                        if text_responses and any(text is not None for text in text_responses):
                                            # 获取第一个有效的文本响应
                                            valid_text = next((t for t in text_responses if t), None)
                                            return [], [valid_text]  # 返回空列表表示没有图片, 和包含第一个有效文本的列表
                                        return [], []

                                    # 获取第一个有效的图片和文本
                                    first_valid_image = next((img for img in image_datas if img), None)
                                    first_valid_text = next((text for text in text_responses if text), None)

                                    return [first_valid_image], [first_valid_text]

                                logger.error(f"未找到编辑后的图片数据: {result}")
                                return [], []
                            except json.JSONDecodeError as je:
                                logger.error(f"解析JSON响应失败: {je}")
                                logger.error(f"响应内容: {response_text[:1000]}...")  # 记录部分响应内容
                                # 继续重试
                        elif response.status in retry_status_codes:
                            # 对于需要重试的状态码，记录并继续循环
                            logger.warning(f"Gemini API返回错误 (状态码: {response.status})，将进行重试")
                            # 继续重试
                        else:
                            # 对于其他错误，记录并返回
                            logger.error(f"Gemini API调用失败 (状态码: {response.status}): {response_text}")
                            return [], []
            except aiohttp.ClientError as ce:
                logger.error(f"API请求客户端错误: {ce}")
                # 继续重试
            except Exception as e:
                logger.error(f"API调用异常: {str(e)}")
                logger.error(traceback.format_exc())
                # 继续重试

            # 增加重试计数并等待
            retry_count += 1
            if retry_count <= max_retries:
                wait_time = retry_delay * (2 ** (retry_count - 1))  # 指数退避
                logger.info(f"等待 {wait_time:.2f} 秒后进行第 {retry_count+1} 次重试")
                await asyncio.sleep(wait_time)

        # 所有重试都失败
        logger.error(f"编辑图片失败，已重试 {max_retries} 次")
        return [], []

    def _get_response_summary(self, response_text: str) -> str:
        """获取API响应的摘要，移除base64编码的部分

        Args:
            response_text: API响应的完整文本

        Returns:
            str: 响应摘要，移除了base64编码的部分
        """
        try:
            # 尝试解析JSON
            data = json.loads(response_text)

            # 创建一个新的对象来存储摘要
            summary = {}

            # 复制除了base64数据之外的所有字段
            if "candidates" in data:
                summary["candidates"] = []
                for candidate in data["candidates"]:
                    candidate_summary = {}

                    # 复制除了content之外的所有字段
                    for key, value in candidate.items():
                        if key != "content":
                            candidate_summary[key] = value

                    # 处理content字段
                    if "content" in candidate:
                        content = candidate["content"]
                        content_summary = {}

                        # 复制除了parts之外的所有字段
                        for key, value in content.items():
                            if key != "parts":
                                content_summary[key] = value

                        # 处理parts字段
                        if "parts" in content:
                            parts = content["parts"]
                            parts_summary = []

                            for part in parts:
                                part_summary = {}

                                # 复制文本内容
                                if "text" in part:
                                    # 限制文本长度，避免过长的文本
                                    text = part["text"]
                                    if len(text) > 200:
                                        part_summary["text"] = text[:200] + "... [TEXT TRUNCATED]"
                                    else:
                                        part_summary["text"] = text

                                # 对于inlineData，只保留mimeType和数据长度信息，完全隐藏base64数据
                                if "inlineData" in part:
                                    inline_data = part["inlineData"]
                                    part_summary["inlineData"] = {
                                        "mimeType": inline_data.get("mimeType", "unknown"),
                                        "dataLength": len(inline_data.get("data", "")),
                                        "data": "[BASE64 DATA HIDDEN]"
                                    }

                                parts_summary.append(part_summary)

                            content_summary["parts"] = parts_summary

                        candidate_summary["content"] = content_summary

                    summary["candidates"].append(candidate_summary)

            # 复制其他字段，但隐藏可能的base64数据
            for key, value in data.items():
                if key != "candidates":
                    if isinstance(value, str) and len(value) > 100 and self._is_likely_base64(value):
                        summary[key] = "[BASE64 DATA HIDDEN]"
                    else:
                        summary[key] = value

            # 转换为JSON字符串，限制长度
            result = json.dumps(summary, indent=2)
            if len(result) > 300:
                result = result[:300] + "... [RESPONSE TRUNCATED]"

            return result
        except Exception as e:
            # 如果解析失败，返回前300个字符，避免输出过多内容
            truncated_text = response_text[:300] + "... [RESPONSE TRUNCATED]" if len(response_text) > 300 else response_text
            return f"[无法解析完整响应: {str(e)}] 响应开头: {truncated_text}"

    def _is_likely_base64(self, text: str) -> bool:
        """检查文本是否可能是base64编码的数据"""
        import string

        # 检查是否只包含base64字符
        if not all(c in string.ascii_letters + string.digits + '+/=' for c in text):
            return False

        # 检查长度是否是4的倍数（可能有填充）
        if len(text) % 4 != 0:
            return False

        # 检查是否有足够的变化（随机性）
        char_counts = {}
        for c in text:
            char_counts[c] = char_counts.get(c, 0) + 1

        # 如果字符种类太少，可能不是base64
        if len(char_counts) < 10:
            return False

        return True

    def _translate_gemini_message(self, text: str) -> str:
        """将Gemini API的英文消息翻译成中文"""
        # 常见的内容审核拒绝消息翻译
        if "I'm unable to create this image" in text:
            if "sexually suggestive" in text:
                return "抱歉，我无法创建这张图片。我不能生成带有性暗示或促进有害刻板印象的内容。请提供其他描述。"
            elif "harmful" in text or "dangerous" in text:
                return "抱歉，我无法创建这张图片。我不能生成可能有害或危险的内容。请提供其他描述。"
            elif "violent" in text:
                return "抱歉，我无法创建这张图片。我不能生成暴力或血腥的内容。请提供其他描述。"
            else:
                return "抱歉，我无法创建这张图片。请尝试修改您的描述，提供其他内容。"

        # 其他常见拒绝消息
        if "cannot generate" in text or "can't generate" in text:
            return "抱歉，我无法生成符合您描述的图片。请尝试其他描述。"

        if "against our content policy" in text:
            return "抱歉，您的请求违反了内容政策，无法生成相关图片。请提供其他描述。"

        # 默认情况，原样返回
        return text

    def get_api_key_for_session(self, session_id):
        """根据会话ID获取或分配API密钥

        Args:
            session_id: 会话ID，通常是chat_id和user_id的组合

        Returns:
            str: 适用于该会话的API密钥
        """
        # 如果没有有效的API密钥，返回空字符串
        if not self.api_keys or all(not key for key in self.api_keys):
            return ""

        # 如果会话已经分配了API密钥，且该密钥仍然有效，则继续使用
        if session_id in self.session_key_mapping:
            api_key = self.session_key_mapping[session_id]
            # 检查该密钥是否仍然在可用列表中
            if api_key in self.api_keys:
                # 更新最后使用时间
                self.key_last_used[api_key] = time.time()
                return api_key

        # 为会话分配新的API密钥（轮询方式）
        api_key = self.rotate_api_key()
        self.session_key_mapping[session_id] = api_key
        # 更新最后使用时间
        self.key_last_used[api_key] = time.time()
        logger.info(f"为会话 {session_id} 分配新的API密钥")
        return api_key

    def rotate_api_key(self):
        """轮询选择下一个API密钥

        Returns:
            str: 下一个可用的API密钥
        """
        # 如果没有有效的API密钥，返回空字符串
        if not self.api_keys or all(not key for key in self.api_keys):
            return ""

        # 获取当前索引
        current_index = self.current_key_index
        # 更新索引，准备下一次使用
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        # 返回当前索引对应的API密钥
        return self.api_keys[current_index]

    def mark_api_key_error(self, api_key, session_id=None):
        """标记API密钥出错，并可能切换到下一个密钥

        Args:
            api_key: 出错的API密钥
            session_id: 会话ID，如果提供，则为该会话重新分配密钥

        Returns:
            str: 新的API密钥，如果没有可用的密钥则返回空字符串
        """
        # 如果API密钥不在列表中，忽略
        if api_key not in self.api_keys:
            return ""

        # 增加错误计数
        self.key_error_counts[api_key] += 1
        logger.warning(f"API密钥 {api_key[:5]}... 出错，当前错误计数: {self.key_error_counts[api_key]}")

        # 如果错误计数过高，可以考虑暂时禁用该密钥
        # 这里简单起见，我们不实现禁用逻辑，只是轮询到下一个密钥

        # 如果提供了会话ID，为该会话重新分配密钥
        if session_id:
            # 获取新的API密钥（轮询方式）
            new_api_key = self.rotate_api_key()
            # 如果新密钥与出错密钥相同且有多个密钥，再轮询一次
            if new_api_key == api_key and len(self.api_keys) > 1:
                new_api_key = self.rotate_api_key()
            # 更新会话映射
            self.session_key_mapping[session_id] = new_api_key
            # 更新最后使用时间
            self.key_last_used[new_api_key] = time.time()
            logger.info(f"为会话 {session_id} 重新分配API密钥")
            return new_api_key

        return ""

    def clean_expired_session_keys(self, expiry_seconds=3600):
        """清理过期的会话密钥映射

        Args:
            expiry_seconds: 过期时间（秒），默认1小时
        """
        current_time = time.time()
        expired_sessions = []

        # 查找过期的会话
        for session_id, api_key in self.session_key_mapping.items():
            # 检查该密钥的最后使用时间
            last_used = self.key_last_used.get(api_key, 0)
            if current_time - last_used > expiry_seconds:
                expired_sessions.append(session_id)

        # 删除过期的会话
        for session_id in expired_sessions:
            del self.session_key_mapping[session_id]

        if expired_sessions:
            logger.info(f"清理了 {len(expired_sessions)} 个过期的会话密钥映射")

    def _cleanup_image_cache(self):
        """清理过期的图片缓存"""
        current_time = time.time()
        expired_keys = []

        for key, cache_data in self.image_cache.items():
            if current_time - cache_data["timestamp"] > self.image_cache_timeout:
                expired_keys.append(key)
                logger.info(f"图片缓存过期，将删除键: {key}")

        for key in expired_keys:
            del self.image_cache[key]

        # 记录当前缓存状态
        if expired_keys:
            logger.info(f"清理后图片缓存包含 {len(self.image_cache)} 个条目")

    def _save_image_to_cache(self, chat_id: str, user_id: str, image_data: bytes):
        """保存图片数据到缓存

        Args:
            chat_id: 聊天ID
            user_id: 用户ID
            image_data: 图片数据
        """
        if not image_data:
            logger.warning("尝试保存空图片数据到缓存")
            return

        # 使用多种格式的键保存图片，以确保后续能找到
        # 1. 元组键 (chat_id, user_id)
        tuple_key = (chat_id, user_id)
        self.image_cache[tuple_key] = {
            "content": image_data,
            "timestamp": time.time()
        }

        # 2. 字符串键 "chat_id_user_id"
        str_key = f"{chat_id}_{user_id}"
        self.image_cache[str_key] = {
            "content": image_data,
            "timestamp": time.time()
        }

        # 3. 如果是私聊，也使用单独的chat_id作为键
        if chat_id == user_id:
            self.image_cache[chat_id] = {
                "content": image_data,
                "timestamp": time.time()
            }

        # 4. 保存到最后一次生成的图片路径
        conversation_key = f"{chat_id}_{user_id}"
        image_path = os.path.join(self.save_dir, f"cache_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
        try:
            with open(image_path, "wb") as f:
                f.write(image_data)
            self.last_images[conversation_key] = image_path
            logger.info(f"保存图片到文件: {image_path}")
        except Exception as e:
            logger.error(f"保存图片到文件失败: {e}")

        logger.info(f"成功缓存图片数据，大小: {len(image_data)} 字节，键: {tuple_key}, {str_key}")
        logger.info(f"当前图片缓存包含 {len(self.image_cache)} 个条目")

    async def _get_recent_image(self, chat_id: str, user_id: str) -> Optional[bytes]:
        """获取最近的图片数据，区分群聊中的不同用户"""
        logger.info(f"尝试获取图片缓存，chat_id: {chat_id}, user_id: {user_id}")

        # 记录当前缓存状态
        logger.info(f"当前图片缓存包含 {len(self.image_cache)} 个条目")
        for key in self.image_cache.keys():
            logger.info(f"缓存键: {key}, 类型: {type(key)}")

        # 先尝试从用户专属缓存获取 - 使用元组键
        cache_key = (chat_id, user_id)
        if cache_key in self.image_cache:
            cache_data = self.image_cache[cache_key]
            if time.time() - cache_data["timestamp"] <= self.image_cache_timeout:
                logger.info(f"找到用户 {user_id} 在聊天 {chat_id} 中的图片缓存，使用元组键")
                return cache_data["content"]

        # 尝试使用字符串格式的键 "chat_id_user_id"
        str_cache_key = f"{chat_id}_{user_id}"
        if str_cache_key in self.image_cache:
            cache_data = self.image_cache[str_cache_key]
            if time.time() - cache_data["timestamp"] <= self.image_cache_timeout:
                logger.info(f"找到用户 {user_id} 在聊天 {chat_id} 中的图片缓存，使用字符串键")
                return cache_data["content"]

        # 如果是私聊且没找到，尝试使用旧格式的键
        if chat_id == user_id:
            # 尝试直接使用chat_id作为键
            if chat_id in self.image_cache:
                cache_data = self.image_cache[chat_id]
                if time.time() - cache_data["timestamp"] <= self.image_cache_timeout:
                    logger.info(f"找到旧格式的图片缓存，键: {chat_id}")
                    return cache_data["content"]

            # 尝试使用user_id作为键
            if user_id in self.image_cache:
                cache_data = self.image_cache[user_id]
                if time.time() - cache_data["timestamp"] <= self.image_cache_timeout:
                    logger.info(f"找到旧格式的图片缓存，键: {user_id}")
                    return cache_data["content"]

        # 尝试查找任何包含chat_id或user_id的键
        for key in list(self.image_cache.keys()):
            if isinstance(key, tuple) and len(key) == 2:
                # 检查元组中是否包含chat_id或user_id
                if chat_id in key or user_id in key:
                    cache_data = self.image_cache[key]
                    if time.time() - cache_data["timestamp"] <= self.image_cache_timeout:
                        logger.info(f"找到相关的图片缓存，键: {key}")
                        return cache_data["content"]
            elif isinstance(key, str):
                # 检查字符串键中是否包含chat_id或user_id
                if chat_id in key or user_id in key:
                    cache_data = self.image_cache[key]
                    if time.time() - cache_data["timestamp"] <= self.image_cache_timeout:
                        logger.info(f"找到相关的图片缓存，键: {key}")
                        return cache_data["content"]

        # 如果所有尝试都失败，检查最后一次生成的图片
        conversation_key = f"{chat_id}_{user_id}"
        last_image_path = self.last_images.get(conversation_key)
        if last_image_path and os.path.exists(last_image_path):
            try:
                with open(last_image_path, "rb") as f:
                    image_data = f.read()
                logger.info(f"从最后生成的图片路径获取图片数据: {last_image_path}")
                return image_data
            except Exception as e:
                logger.error(f"读取最后生成的图片失败: {e}")

        logger.warning(f"未找到任何可用的图片缓存，chat_id: {chat_id}, user_id: {user_id}")
        return None

    async def _handle_analyze_image(self, bot: WechatAPIClient, message: dict, image_data: bytes):
        """处理图片分析请求

        Args:
            bot: 微信API客户端
            message: 消息数据
            image_data: 图片数据
        """
        try:
            # 获取会话信息
            chat_id = message.get("chat_id", "")
            user_id = message.get("user_id", "")
            from_wxid = message.get("FromWxid", "")
            sender_wxid = message.get("SenderWxid", "")
            conversation_key = f"{chat_id}_{user_id}"

            # 发送提示消息
            # 尝试使用from_wxid而不是chat_id
            if from_wxid:
                await bot.send_text_message(from_wxid, "正在分析图片，请稍候...")
                logger.info(f"使用from_wxid发送图片分析提示消息")
            else:
                await bot.send_text_message(chat_id, "正在分析图片，请稍候...")
                logger.info(f"使用chat_id发送图片分析提示消息")

            # 创建消息信息字典，传递给_analyze_image方法
            message_info = {
                "user_id": user_id,
                "chat_id": chat_id,
                "from_wxid": from_wxid,
                "sender_wxid": sender_wxid
            }

            # 调用图片分析
            result = await self._analyze_image(image_data, message_info)

            if result:
                # 发送结果
                # 尝试使用from_wxid而不是chat_id
                if from_wxid:
                    await bot.send_text_message(from_wxid, result)
                    logger.info(f"使用from_wxid发送图片分析结果")

                    # 保存图片到会话历史，以便后续对话
                    # 保存图片到本地
                    image_path = os.path.join(self.save_dir, f"analysis_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                    with open(image_path, "wb") as f:
                        f.write(image_data)

                    # 更新会话历史
                    conversation_history = self.conversations.get(conversation_key, [])

                    # 添加用户消息（包含图片）
                    user_message = {
                        "role": "user",
                        "parts": [
                            {"text": "请分析这张图片"},
                            {"image_url": image_path}
                        ]
                    }
                    conversation_history.append(user_message)

                    # 添加助手消息（分析结果）
                    assistant_message = {
                        "role": "model",
                        "parts": [
                            {"text": result}
                        ]
                    }
                    conversation_history.append(assistant_message)

                    # 更新会话历史
                    self.conversations[conversation_key] = conversation_history

                    # 更新会话时间戳
                    self.conversation_timestamps[conversation_key] = time.time()

                    # 保存最后生成的图片路径
                    self.last_images[conversation_key] = image_path

                    logger.info(f"已将图片分析会话添加到历史记录，会话键: {conversation_key}")
                else:
                    await bot.send_text_message(chat_id, result)
                    logger.info(f"使用chat_id发送图片分析结果")
            else:
                # 发送错误消息
                # 尝试使用from_wxid而不是chat_id
                if from_wxid:
                    await bot.send_text_message(from_wxid, "无法分析图片，请稍后再试或尝试其他图片")
                    logger.info(f"使用from_wxid发送图片分析失败消息")
                else:
                    await bot.send_text_message(chat_id, "无法分析图片，请稍后再试或尝试其他图片")
                    logger.info(f"使用chat_id发送图片分析失败消息")
        except Exception as e:
            logger.error(f"处理图片分析请求异常: {str(e)}")
            logger.error(traceback.format_exc())
            # 尝试使用from_wxid而不是chat_id
            from_wxid = message.get("FromWxid", "")
            chat_id = message.get("chat_id", "")
            if from_wxid:
                await bot.send_text_message(from_wxid, f"分析图片失败: {str(e)}")
                logger.info(f"使用from_wxid发送图片分析异常消息")
            else:
                await bot.send_text_message(chat_id, f"分析图片失败: {str(e)}")
                logger.info(f"使用chat_id发送图片分析异常消息")
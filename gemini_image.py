import os
import json
import uuid
import time
import base64
from io import BytesIO
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

from PIL import Image
import requests
from loguru import logger

import plugins
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from plugins import *

@plugins.register(
    name="GeminiImage",
    desire_priority=20,
    hidden=False,
    desc="基于Google Gemini的图像生成插件",
    version="1.0.3",
    author="sofs2005",
)
class GeminiImage(Plugin):
    """基于Google Gemini的图像生成插件
    
    功能：
    1. 生成图片：根据文本描述生成图片
    2. 编辑图片：根据文本描述修改已有图片
    3. 支持会话模式，可以连续对话修改图片
    4. 支持积分系统控制使用
    """
    
    # 默认配置
    DEFAULT_CONFIG = {
        "enable": True,
        "gemini_api_key": "",
        "model": "gemini-2.0-flash-exp-image-generation",
        "commands": ["$生成图片", "$画图", "$图片生成"],
        "edit_commands": ["$编辑图片", "$修改图片"],
        "exit_commands": ["$结束对话", "$退出对话", "$关闭对话", "$结束"],
        "enable_points": False,
        "generate_image_cost": 10,
        "edit_image_cost": 15,
        "save_path": "temp",
        "enable_proxy": False,
        "proxy_url": "",
        "base_url": "https://generativelanguage.googleapis.com",
    }

    def __init__(self):
        """初始化插件配置"""
        try:
            super().__init__()
            
            # 载入配置
            self.config = super().load_config()
            if not self.config:
                self.config = self._load_config_template()
            
            # 使用默认配置初始化
            for key, default_value in self.DEFAULT_CONFIG.items():
                if key not in self.config:
                    self.config[key] = default_value
            
            # 设置配置参数
            self.enable = self.config.get("enable", True)
            self.api_key = self.config.get("gemini_api_key", "")
            self.model = self.config.get("model", "gemini-2.0-flash-exp-image-generation")
            
            # 获取命令配置
            self.commands = self.config.get("commands", ["#生成图片", "#画图", "#图片生成"])
            self.edit_commands = self.config.get("edit_commands", ["#编辑图片", "#修改图片"])
            self.exit_commands = self.config.get("exit_commands", ["#结束对话", "#退出对话", "#关闭对话", "#结束"])
            
            # 获取图片保存配置
            self.save_path = self.config.get("save_path", "temp")
            self.save_dir = os.path.join(os.path.dirname(__file__), self.save_path)
            os.makedirs(self.save_dir, exist_ok=True)
            
            # 获取代理配置
            self.enable_proxy = self.config.get("enable_proxy", False)
            self.proxy_url = self.config.get("proxy_url", "")
            
            # 获取baseurl配置
            self.base_url = self.config.get("base_url", "https://generativelanguage.googleapis.com")
            
            # 初始化会话状态，用于保存上下文
            self.conversations = defaultdict(list)  # 用户ID -> 对话历史列表
            self.conversation_expiry = 600  # 会话过期时间(秒)
            self.conversation_timestamps = {}  # 用户ID -> 最后活动时间
            
            # 存储最后一次生成的图片路径
            self.last_images = {}  # 会话标识 -> 最后一次生成的图片路径
            
            # 全局图片缓存，用于存储最近接收到的图片
            self.image_cache = {}  # 会话标识 -> {content: bytes, timestamp: float}
            self.image_cache_timeout = 300  # 图片缓存过期时间(秒)
            
            # 验证关键配置
            if not self.api_key:
                logger.warning("GeminiImage插件未配置API密钥")
            
            # 绑定事件处理函数
            self.handlers[Event.ON_HANDLE_CONTEXT] = self.on_handle_context
            
            # 设置定期清理标志和最后清理时间
            self._last_cleanup_time = time.time()
            self._start_cleanup_thread()
            
            logger.info("GeminiImage插件初始化成功")
            if self.enable_proxy:
                logger.info(f"GeminiImage插件已启用代理: {self.proxy_url}")
            
        except Exception as e:
            logger.error(f"GeminiImage插件初始化失败: {str(e)}")
            logger.exception(e)
            self.enable = False
    
    def _start_cleanup_thread(self):
        """启动一个后台线程用于定期清理"""
        import threading
        
        # 定义清理函数
        def cleanup_worker():
            while True:
                try:
                    # 获取当前时间
                    current_time = time.time()
                    current_hour = time.localtime(current_time).tm_hour
                    
                    # 晚上2点到4点之间执行清理
                    is_night_time = 2 <= current_hour <= 4
                    time_since_last_cleanup = current_time - self._last_cleanup_time
                    
                    # 如果是夜间或者距离上次清理已经超过24小时，执行清理
                    if is_night_time or time_since_last_cleanup > 24 * 3600:
                        logger.info("执行定期清理临时文件")
                        self._cleanup_temp_files()
                        self._last_cleanup_time = current_time
                
                # 异常处理，确保线程不会因为错误而终止
                except Exception as e:
                    logger.error(f"清理线程发生错误: {str(e)}")
                
                # 每小时检查一次
                time.sleep(3600)
        
        # 创建并启动后台线程
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("临时文件清理线程已启动")

    def on_handle_context(self, e_context: EventContext):
        """处理消息事件"""
        if not self.enable:
            return
        
        context = e_context['context']
        
        # 清理过期的会话和图片缓存
        self._cleanup_expired_conversations()
        self._cleanup_image_cache()
        
        # 基于时间的临时文件清理
        current_time = time.time()
        if not hasattr(self, '_last_cleanup_time'):
            self._last_cleanup_time = current_time
            
        # 检查是否是深夜时段（凌晨2-4点之间）
        current_hour = time.localtime(current_time).tm_hour
        is_night_time = 2 <= current_hour <= 4
        
        # 如果是深夜时段，且距离上次清理已超过6小时，执行清理
        if is_night_time and (current_time - self._last_cleanup_time) > 6 * 3600:
            logger.info("执行夜间定时清理临时文件")
            self._cleanup_temp_files()
            self._last_cleanup_time = current_time
        
        # 会话标识: 用户ID+会话ID
        user_id = context["session_id"]
        conversation_key = user_id
        is_group = context.get("isgroup", False)
        
        # 处理图片消息 - 用于缓存用户发送的图片
        if context.type == ContextType.IMAGE:
            self._handle_image_message(e_context)
            return
            
        # 处理文本消息
        if context.type != ContextType.TEXT:
            return
        
        content = context.content.strip()
        
        # 检查是否是结束对话命令
        if content in self.exit_commands:
            if conversation_key in self.conversations:
                # 清除会话数据
                del self.conversations[conversation_key]
                if conversation_key in self.conversation_timestamps:
                    del self.conversation_timestamps[conversation_key]
                if conversation_key in self.last_images:
                    del self.last_images[conversation_key]
                
                reply = Reply(ReplyType.TEXT, "已结束Gemini图像生成对话，下次需要时请使用命令重新开始")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
            else:
                # 没有活跃会话
                reply = Reply(ReplyType.TEXT, "您当前没有活跃的Gemini图像生成对话")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
            return

        # 检查是否是生成图片命令
        for cmd in self.commands:
            if content.startswith(cmd):
                # 提取提示词
                prompt = content[len(cmd):].strip()
                if not prompt:
                    reply = Reply(ReplyType.TEXT, f"请提供描述内容，格式：{cmd} [描述]")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 检查API密钥是否配置
                if not self.api_key:
                    reply = Reply(ReplyType.TEXT, "请先在配置文件中设置Gemini API密钥")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 尝试生成图片
                try:
                    # 发送处理中消息
                    processing_reply = Reply(ReplyType.TEXT, "正在生成图片，请稍候...")
                    e_context["reply"] = processing_reply
                    
                    # 获取上下文历史
                    conversation_history = self.conversations[conversation_key]
                    
                    # 生成图片
                    image_data, text_response = self._generate_image(prompt, conversation_history)
                    
                    if image_data:
                        # 保存图片到本地
                        reply_text = text_response if text_response else "图片生成成功！"
                        if not conversation_history or len(conversation_history) <= 2:  # 如果是新会话
                            reply_text += f"（已开始图像对话，可以继续发送命令修改图片。需要结束时请发送\"{self.exit_commands[0]}\"）"
                        
                        # 将回复文本添加到文件名中
                        clean_text = reply_text.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_")
                        clean_text = clean_text[:30] + "..." if len(clean_text) > 30 else clean_text
                        
                        image_path = os.path.join(self.save_dir, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}_{clean_text}.png")
                        with open(image_path, "wb") as f:
                            f.write(image_data)
                        
                        # 保存最后生成的图片路径
                        self.last_images[conversation_key] = image_path
                        
                        # 添加用户提示到会话
                        user_message = {"role": "user", "parts": [{"text": prompt}]}
                        conversation_history.append(user_message)
                        
                        # 添加助手回复到会话
                        assistant_message = {
                            "role": "model", 
                            "parts": [
                                {"text": text_response if text_response else "我已生成了图片"},
                                {"image_url": image_path}
                            ]
                        }
                        conversation_history.append(assistant_message)
                        
                        # 限制会话历史长度
                        if len(conversation_history) > 10:  # 保留最近5轮对话
                            conversation_history = conversation_history[-10:]
                        
                        # 更新会话时间戳
                        self.conversation_timestamps[conversation_key] = time.time()
                        
                        # 准备回复文本
                        reply_text = text_response if text_response else "图片生成成功！"
                        if not conversation_history or len(conversation_history) <= 2:  # 如果是新会话
                            reply_text += f"（已开始图像对话，可以继续发送命令修改图片。需要结束时请发送\"{self.exit_commands[0]}\"）"
                        
                        # 先发送文本消息
                        e_context["channel"].send(Reply(ReplyType.TEXT, reply_text), e_context["context"])
                        
                        # 创建文件对象，由框架负责关闭
                        image_file = open(image_path, "rb")
                        e_context["reply"] = Reply(ReplyType.IMAGE, image_file)
                        e_context.action = EventAction.BREAK_PASS
                    else:
                        # 检查是否有文本响应，可能是内容被拒绝
                        if text_response:
                            # 内容审核拒绝的情况，翻译并发送拒绝消息
                            translated_response = self._translate_gemini_message(text_response)
                            reply = Reply(ReplyType.TEXT, translated_response)
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                        else:
                            reply = Reply(ReplyType.TEXT, "图片生成失败，请稍后再试或修改提示词")
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                except Exception as e:
                    logger.error(f"生成图片失败: {str(e)}")
                    logger.exception(e)
                    reply = Reply(ReplyType.TEXT, f"生成图片失败: {str(e)}")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                return

        # 检查是否是编辑图片命令
        for cmd in self.edit_commands:
            if content.startswith(cmd):
                # 提取提示词
                prompt = content[len(cmd):].strip()
                if not prompt:
                    reply = Reply(ReplyType.TEXT, f"请提供编辑描述，格式：{cmd} [描述]")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 检查API密钥是否配置
                if not self.api_key:
                    reply = Reply(ReplyType.TEXT, "请先在配置文件中设置Gemini API密钥")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 先尝试从缓存获取最近的图片
                image_data = self._get_recent_image(conversation_key)
                if image_data:
                    # 如果找到缓存的图片，保存到本地再处理
                    image_path = os.path.join(self.save_dir, f"temp_{int(time.time())}_{uuid.uuid4().hex[:8]}.png")
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                    self.last_images[conversation_key] = image_path
                    logger.info(f"找到最近缓存的图片，保存到：{image_path}")
                    
                    # 尝试编辑图片
                    try:
                        # 发送处理中消息
                        processing_reply = Reply(ReplyType.TEXT, "正在编辑图片，请稍候...")
                        e_context["reply"] = processing_reply
                        
                        # 获取会话上下文
                        conversation_history = self.conversations[conversation_key]
                        
                        # 编辑图片
                        result_image, text_response = self._edit_image(prompt, image_data, conversation_history)
                        
                        if result_image:
                            # 保存编辑后的图片
                            reply_text = text_response if text_response else "图片编辑成功！"
                            if not conversation_history or len(conversation_history) <= 2:  # 如果是新会话
                                reply_text += f"（已开始图像对话，可以继续发送命令修改图片。需要结束时请发送\"{self.exit_commands[0]}\"）"
                            
                            # 将回复文本添加到文件名中
                            clean_text = reply_text.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_")
                            clean_text = clean_text[:30] + "..." if len(clean_text) > 30 else clean_text
                            
                            edited_image_path = os.path.join(self.save_dir, f"edited_{int(time.time())}_{uuid.uuid4().hex[:8]}_{clean_text}.png")
                            with open(edited_image_path, "wb") as f:
                                f.write(result_image)
                            
                            # 更新最后生成的图片路径
                            self.last_images[conversation_key] = edited_image_path
                            
                            # 更新会话历史
                            user_message = {
                                "role": "user", 
                                "parts": [
                                    {"text": prompt},
                                    {"image_url": image_path}
                                ]
                            }
                            conversation_history.append(user_message)
                            
                            assistant_message = {
                                "role": "model", 
                                "parts": [
                                    {"text": text_response if text_response else "我已编辑完成图片"},
                                    {"image_url": edited_image_path}
                                ]
                            }
                            conversation_history.append(assistant_message)
                            
                            # 限制会话历史长度
                            if len(conversation_history) > 10:  # 保留最近5轮对话
                                conversation_history = conversation_history[-10:]
                            
                            # 更新会话时间戳
                            self.conversation_timestamps[conversation_key] = time.time()
                            
                            # 准备回复文本
                            reply_text = text_response if text_response else "图片编辑成功！"
                            if not conversation_history or len(conversation_history) <= 2:  # 如果是新会话
                                reply_text += f"（已开始图像对话，可以继续发送命令修改图片。需要结束时请发送\"{self.exit_commands[0]}\"）"
                            
                            # 先发送文本消息
                            e_context["channel"].send(Reply(ReplyType.TEXT, reply_text), e_context["context"])
                            
                            # 创建文件对象，由框架负责关闭
                            edited_image_file = open(edited_image_path, "rb")
                            e_context["reply"] = Reply(ReplyType.IMAGE, edited_image_file)
                            e_context.action = EventAction.BREAK_PASS
                            return
                        else:
                            # 检查是否有文本响应，可能是内容被拒绝
                            if text_response:
                                # 内容审核拒绝的情况，翻译并发送拒绝消息
                                translated_response = self._translate_gemini_message(text_response)
                                reply = Reply(ReplyType.TEXT, translated_response)
                                e_context["reply"] = reply
                                e_context.action = EventAction.BREAK_PASS
                            else:
                                reply = Reply(ReplyType.TEXT, "图片编辑失败，请稍后再试或修改描述")
                                e_context["reply"] = reply
                                e_context.action = EventAction.BREAK_PASS
                            return
                    except Exception as e:
                        logger.error(f"编辑图片失败: {str(e)}")
                        logger.exception(e)
                        reply = Reply(ReplyType.TEXT, f"编辑图片失败: {str(e)}")
                        e_context["reply"] = reply
                        e_context.action = EventAction.BREAK_PASS
                        return
                
                # 检查是否有上一次上传/生成的图片
                last_image_path = self.last_images.get(conversation_key)
                if not last_image_path or not os.path.exists(last_image_path):
                    reply = Reply(ReplyType.TEXT, "未找到可编辑的图片，请先上传一张图片或使用生成图片命令")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 尝试编辑图片
                try:
                    # 发送处理中消息
                    processing_reply = Reply(ReplyType.TEXT, "正在编辑图片，请稍候...")
                    e_context["reply"] = processing_reply
                    
                    # 读取上一次的图片
                    with open(last_image_path, "rb") as f:
                        image_data = f.read()
                    
                    # 获取会话上下文
                    conversation_history = self.conversations[conversation_key]
                    
                    # 编辑图片
                    result_image, text_response = self._edit_image(prompt, image_data, conversation_history)
                    
                    if result_image:
                        # 保存编辑后的图片
                        reply_text = text_response if text_response else "图片编辑成功！"
                        if not conversation_history or len(conversation_history) <= 2:  # 如果是新会话
                            reply_text += f"（已开始图像对话，可以继续发送命令修改图片。需要结束时请发送\"{self.exit_commands[0]}\"）"
                        
                        # 将回复文本添加到文件名中
                        clean_text = reply_text.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_")
                        clean_text = clean_text[:30] + "..." if len(clean_text) > 30 else clean_text
                        
                        edited_image_path = os.path.join(self.save_dir, f"edited_{int(time.time())}_{uuid.uuid4().hex[:8]}_{clean_text}.png")
                        with open(edited_image_path, "wb") as f:
                            f.write(result_image)
                        
                        # 更新最后生成的图片路径
                        self.last_images[conversation_key] = edited_image_path
                        
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
                                {"text": text_response if text_response else "我已编辑完成图片"},
                                {"image_url": edited_image_path}
                            ]
                        }
                        conversation_history.append(assistant_message)
                        
                        # 限制会话历史长度
                        if len(conversation_history) > 10:  # 保留最近5轮对话
                            conversation_history = conversation_history[-10:]
                        
                        # 更新会话时间戳
                        self.conversation_timestamps[conversation_key] = time.time()
                        
                        # 准备回复文本
                        reply_text = text_response if text_response else "图片编辑成功！"
                        if not conversation_history or len(conversation_history) <= 2:  # 如果是新会话
                            reply_text += f"（已开始图像对话，可以继续发送命令修改图片。需要结束时请发送\"{self.exit_commands[0]}\"）"
                        
                        # 先发送文本消息
                        e_context["channel"].send(Reply(ReplyType.TEXT, reply_text), e_context["context"])
                        
                        # 创建文件对象，由框架负责关闭
                        edited_image_file = open(edited_image_path, "rb")
                        e_context["reply"] = Reply(ReplyType.IMAGE, edited_image_file)
                        e_context.action = EventAction.BREAK_PASS
                    else:
                        # 检查是否有文本响应，可能是内容被拒绝
                        if text_response:
                            # 内容审核拒绝的情况，翻译并发送拒绝消息
                            translated_response = self._translate_gemini_message(text_response)
                            reply = Reply(ReplyType.TEXT, translated_response)
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                        else:
                            reply = Reply(ReplyType.TEXT, "图片编辑失败，请稍后再试或修改描述")
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                except Exception as e:
                    logger.error(f"编辑图片失败: {str(e)}")
                    logger.exception(e)
                    reply = Reply(ReplyType.TEXT, f"编辑图片失败: {str(e)}")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                return

        # 检查是否是对话继续（没有前缀命令，但有活跃会话）
        if conversation_key in self.conversations:
            # 有活跃会话，视为继续对话
            try:
                # 检查是否有上一次生成的图片
                last_image_path = self.last_images.get(conversation_key)
                if not last_image_path or not os.path.exists(last_image_path):
                    # 没有上一次图片，当作生成新图片处理
                    reply = Reply(ReplyType.TEXT, "未找到上一次生成的图片，请使用生成图片命令开始新的会话")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 发送处理中消息
                processing_reply = Reply(ReplyType.TEXT, "正在处理您的请求，请稍候...")
                e_context["reply"] = processing_reply
                
                # 获取上下文历史
                conversation_history = self.conversations[conversation_key]
                
                # 尝试编辑图片
                with open(last_image_path, "rb") as f:
                    image_data = f.read()
                
                # 编辑图片
                result_image, text_response = self._edit_image(content, image_data, conversation_history)
                
                if result_image:
                    # 保存编辑后的图片
                    reply_text = text_response if text_response else "图片修改成功！"
                    
                    # 将回复文本添加到文件名中
                    clean_text = reply_text.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_")
                    clean_text = clean_text[:30] + "..." if len(clean_text) > 30 else clean_text
                    
                    new_image_path = os.path.join(self.save_dir, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}_{clean_text}.png")
                    with open(new_image_path, "wb") as f:
                        f.write(result_image)
                    
                    # 更新最后生成的图片路径
                    self.last_images[conversation_key] = new_image_path
                    
                    # 更新会话历史
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
                            {"text": text_response if text_response else "我已编辑了图片"},
                            {"image_url": new_image_path}
                        ]
                    }
                    conversation_history.append(assistant_message)
                    
                    # 限制会话历史长度
                    if len(conversation_history) > 10:
                        conversation_history = conversation_history[-10:]
                    
                    # 更新会话时间戳
                    self.conversation_timestamps[conversation_key] = time.time()
                    
                    # 准备回复文本
                    reply_text = text_response if text_response else "图片修改成功！"
                    
                    # 先发送文本消息
                    e_context["channel"].send(Reply(ReplyType.TEXT, reply_text), e_context["context"])
                    
                    # 创建文件对象，由框架负责关闭
                    new_image_file = open(new_image_path, "rb")
                    e_context["reply"] = Reply(ReplyType.IMAGE, new_image_file)
                    e_context.action = EventAction.BREAK_PASS
                else:
                    # 检查是否有文本响应，可能是内容被拒绝
                    if text_response:
                        # 内容审核拒绝的情况，翻译并发送拒绝消息
                        translated_response = self._translate_gemini_message(text_response)
                        reply = Reply(ReplyType.TEXT, translated_response)
                        e_context["reply"] = reply
                        e_context.action = EventAction.BREAK_PASS
                    else:
                        reply = Reply(ReplyType.TEXT, "图片修改失败，请稍后再试或修改描述")
                        e_context["reply"] = reply
                        e_context.action = EventAction.BREAK_PASS
            except Exception as e:
                logger.error(f"对话继续生成图片失败: {str(e)}")
                logger.exception(e)
                reply = Reply(ReplyType.TEXT, f"处理失败: {str(e)}")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
            return
    
    def _handle_image_message(self, e_context: EventContext):
        """处理图片消息，缓存图片数据以备后续编辑使用"""
        context = e_context['context']
        session_id = context["session_id"]
        is_group = context.get("isgroup", False)
        
        # 获取发送者ID，确保群聊和单聊场景都能正确缓存
        sender_id = None
        if 'msg' in context.kwargs:
            msg = context.kwargs['msg']
            # 优先使用actual_user_id或from_user_id
            if hasattr(msg, 'actual_user_id') and msg.actual_user_id:
                sender_id = msg.actual_user_id
                logger.info(f"使用actual_user_id作为发送者ID: {sender_id}")
            elif hasattr(msg, 'from_user_id') and msg.from_user_id:
                sender_id = msg.from_user_id
                logger.info(f"使用from_user_id作为发送者ID: {sender_id}")
            # 检查是否在群聊中sender_id与session_id相同，如果相同说明获取发送者ID不正确
            if is_group and sender_id == session_id:
                # 尝试从其他属性获取发送者ID
                if hasattr(msg, 'sender_id') and msg.sender_id:
                    sender_id = msg.sender_id
                    logger.info(f"使用sender_id作为发送者ID: {sender_id}")
                elif hasattr(msg, 'sender_wxid') and msg.sender_wxid:
                    sender_id = msg.sender_wxid
                    logger.info(f"使用sender_wxid作为发送者ID: {sender_id}")
                elif hasattr(msg, 'self_display_name') and msg.self_display_name:
                    # 作为最后的备选方案，使用显示名称
                    sender_id = msg.self_display_name
                    logger.info(f"使用self_display_name作为发送者ID: {sender_id}")
        
        # 记录所有可能的用户标识符，便于调试
        if 'msg' in context.kwargs and hasattr(context.kwargs['msg'], '__dict__'):
            user_attrs = {}
            for attr in ['from_user_id', 'actual_user_id', 'sender_id', 'sender_wxid', 'from_user_nickname', 
                         'self_display_name', 'other_user_id']:
                if hasattr(context.kwargs['msg'], attr):
                    user_attrs[attr] = getattr(context.kwargs['msg'], attr)
            logger.info(f"消息对象中的用户标识符: {user_attrs}")
        
        # 如果仍然无法获取sender_id，使用session_id的一部分
        if not sender_id:
            sender_id = f"user_{hash(session_id) % 10000}"
            logger.info(f"使用生成的ID作为发送者ID: {sender_id}")
        
        # 生成缓存键，在群聊中使用群ID+用户ID组合，在单聊中使用用户ID
        if is_group:
            # 群聊场景：群ID_用户ID
            cache_key = f"{session_id}_{sender_id}"
        else:
            # 单聊场景：使用发送者ID
            cache_key = sender_id
        
        logger.info(f"图片缓存键: {cache_key} (群聊:{is_group})")
        
        try:
            # 获取图片数据
            image_data = None
            
            # 尝试从content获取文件路径并读取文件
            if hasattr(context, 'content') and context.content:
                file_path = context.content
                # 尝试将相对路径转换为绝对路径
                if not os.path.isabs(file_path):
                    abs_path = os.path.abspath(file_path)
                    logger.info(f"转换为绝对路径: {abs_path}")
                    if os.path.exists(abs_path):
                        file_path = abs_path
                
                logger.info(f"从content获取到文件路径: {file_path}")
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'rb') as f:
                            image_data = f.read()
                        logger.info(f"从文件路径读取到图片数据，大小: {len(image_data)} 字节")
                    except Exception as e:
                        logger.error(f"读取图片文件失败: {e}")
                else:
                    logger.warning(f"文件路径不存在: {file_path}")
            
            # 尝试从msg对象获取图片数据
            if not image_data and 'msg' in context.kwargs:
                msg = context.kwargs['msg']
                logger.info(f"MSG对象属性: {dir(msg)}")
                
                # 检查msg是否有download_image方法
                if hasattr(msg, 'download_image') and callable(getattr(msg, 'download_image')):
                    try:
                        image_data = msg.download_image()
                        logger.info(f"通过download_image方法获取到图片数据")
                    except Exception as e:
                        logger.error(f"download_image方法调用失败: {e}")
                
                # 检查msg是否有msg_data属性
                elif hasattr(msg, 'msg_data'):
                    try:
                        msg_data = msg.msg_data
                        logger.info(f"MSG.msg_data: {type(msg_data)}")
                        if isinstance(msg_data, dict) and 'image' in msg_data:
                            image_data = msg_data['image']
                            logger.info(f"从msg_data['image']获取到图片数据")
                        elif isinstance(msg_data, bytes):
                            image_data = msg_data
                            logger.info(f"从msg_data(bytes)获取到图片数据")
                    except Exception as e:
                        logger.error(f"获取msg_data失败: {e}")
                
                # 检查msg是否有img属性
                elif hasattr(msg, 'img') and msg.img:
                    image_data = msg.img
                    logger.info(f"从msg.img获取到图片数据")
                
                # 检查msg是否有文件内容属性
                elif hasattr(msg, 'content') and isinstance(msg.content, bytes):
                    image_data = msg.content
                    logger.info(f"从msg.content获取到图片数据，大小: {len(image_data)} 字节")
                
                # 检查msg对象中可能保存的地址
                if hasattr(msg, 'from_user_id') and hasattr(msg, 'msg_id'):
                    # 尝试构建通用图片保存路径
                    possible_paths = [
                        f"tmp/{msg.msg_id}.png",
                        f"tmp/{msg.msg_id}.jpg",
                        f"tmp/image_{msg.msg_id}.png",
                        f"tmp/image_{msg.from_user_id}_{msg.msg_id}.png"
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            try:
                                with open(path, 'rb') as f:
                                    image_data = f.read()
                                logger.info(f"从路径 {path} 读取到图片数据，大小: {len(image_data)} 字节")
                                break
                            except Exception as e:
                                logger.error(f"读取图片 {path} 失败: {e}")
            
            # 验证获取到的图片数据是否有效
            if image_data and len(image_data) > 100:
                # 尝试验证图片格式
                try:
                    Image.open(BytesIO(image_data))
                    
                    # 保存图片到缓存
                    self.image_cache[cache_key] = {
                        "content": image_data,
                        "timestamp": time.time()
                    }
                    logger.info(f"成功缓存图片数据，大小: {len(image_data)} 字节，缓存键: {cache_key}")
                    
                    # 静默处理图片，不发送任何提示消息
                except Exception as e:
                    logger.error(f"验证图片格式失败: {e}")
            else:
                logger.warning(f"未获取到有效的图片数据或数据太小: {image_data[:20] if image_data else 'None'}")
        except Exception as e:
            logger.error(f"处理图片消息失败: {str(e)}")
            logger.exception(e)
    
    def _get_recent_image(self, conversation_key: str) -> Optional[bytes]:
        """获取最近的图片数据，支持群聊和单聊场景
        
        Args:
            conversation_key: 会话标识，可能是session_id或用户ID
            
        Returns:
            Optional[bytes]: 图片数据或None
        """
        # 尝试从conversation_key直接获取缓存
        cache_data = self.image_cache.get(conversation_key)
        if cache_data and time.time() - cache_data["timestamp"] <= self.image_cache_timeout:
            logger.info(f"从缓存获取到图片数据，大小: {len(cache_data['content'])} 字节，缓存键: {conversation_key}")
            return cache_data["content"]
        
        # 群聊场景：尝试使用当前消息上下文中的发送者ID
        context = e_context['context'] if 'e_context' in locals() else None
        if not context and hasattr(self, 'current_context'):
            context = self.current_context
            
        if context and context.get("isgroup", False):
            sender_id = None
            if 'msg' in context.kwargs:
                msg = context.kwargs['msg']
                # 优先使用actual_user_id或from_user_id
                if hasattr(msg, 'actual_user_id') and msg.actual_user_id:
                    sender_id = msg.actual_user_id
                elif hasattr(msg, 'from_user_id') and msg.from_user_id:
                    sender_id = msg.from_user_id
                # 如果sender_id与session_id相同，尝试其他属性
                if sender_id == context.get("session_id"):
                    if hasattr(msg, 'sender_id') and msg.sender_id:
                        sender_id = msg.sender_id
                    elif hasattr(msg, 'sender_wxid') and msg.sender_wxid:
                        sender_id = msg.sender_wxid
                    elif hasattr(msg, 'self_display_name') and msg.self_display_name:
                        sender_id = msg.self_display_name
                
                if sender_id:
                    # 使用群ID_用户ID格式查找
                    group_key = f"{context.get('session_id')}_{sender_id}"
                    cache_data = self.image_cache.get(group_key)
                    if cache_data and time.time() - cache_data["timestamp"] <= self.image_cache_timeout:
                        logger.info(f"从群聊缓存键获取到图片数据，大小: {len(cache_data['content'])} 字节，缓存键: {group_key}")
                        return cache_data["content"]
        
        # 遍历所有缓存键，查找匹配的键
        for cache_key in self.image_cache:
            if cache_key.startswith(f"{conversation_key}_") or cache_key.endswith(f"_{conversation_key}"):
                cache_data = self.image_cache.get(cache_key)
                if cache_data and time.time() - cache_data["timestamp"] <= self.image_cache_timeout:
                    logger.info(f"从组合缓存键获取到图片数据，大小: {len(cache_data['content'])} 字节，缓存键: {cache_key}")
                    return cache_data["content"]
                
        # 如果没有找到，尝试其他方法
        if '_' in conversation_key:
            # 拆分组合键，可能是群ID_用户ID格式
            parts = conversation_key.split('_')
            for part in parts:
                cache_data = self.image_cache.get(part)
                if cache_data and time.time() - cache_data["timestamp"] <= self.image_cache_timeout:
                    logger.info(f"从拆分键部分获取到图片数据，大小: {len(cache_data['content'])} 字节，缓存键: {part}")
                    return cache_data["content"]
                    
        return None
    
    def _cleanup_image_cache(self):
        """清理过期的图片缓存"""
        current_time = time.time()
        expired_keys = []
        
        for key, cache_data in self.image_cache.items():
            if current_time - cache_data["timestamp"] > self.image_cache_timeout:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.image_cache[key]
            logger.debug(f"清理过期图片缓存: {key}")
    
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
    
    def _cleanup_temp_files(self, max_age_hours=24):
        """清理保存目录中的旧图片文件
        
        Args:
            max_age_hours: 文件最大保留时间（小时）
        """
        try:
            if not os.path.exists(self.save_dir):
                return
                
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            deleted_count = 0
            
            # 遍历save_dir目录下的所有文件
            for filename in os.listdir(self.save_dir):
                file_path = os.path.join(self.save_dir, filename)
                
                # 跳过目录
                if os.path.isdir(file_path):
                    continue
                    
                # 检查文件是否为插件生成的图片文件
                if filename.startswith(("gemini_", "edited_", "temp_")):
                    try:
                        # 获取文件修改时间
                        file_mod_time = os.path.getmtime(file_path)
                        file_age = current_time - file_mod_time
                        
                        # 如果文件超过最大保留时间，则删除
                        if file_age > max_age_seconds:
                            # 检查文件是否在最后生成的图片路径中
                            in_use = False
                            for last_image in self.last_images.values():
                                if file_path == last_image:
                                    in_use = True
                                    break
                            
                            # 如果文件不在使用中，删除它
                            if not in_use:
                                os.remove(file_path)
                                deleted_count += 1
                                logger.debug(f"清理临时图片文件: {file_path}")
                    except Exception as e:
                        logger.warning(f"清理临时文件时出错: {str(e)}")
            
            if deleted_count > 0:
                logger.info(f"共清理 {deleted_count} 个临时图片文件")
        except Exception as e:
            logger.error(f"清理临时文件失败: {str(e)}")
    
    def _generate_image(self, prompt: str, conversation_history: List[Dict] = None) -> Tuple[Optional[bytes], Optional[str]]:
        """调用Gemini API生成图片，返回图片数据和文本响应"""
        url = f"{self.base_url}/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent"
        headers = {
            "Content-Type": "application/json",
        }
        
        params = {
            "key": self.api_key
        }
        
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
        proxies = None
        if self.enable_proxy and self.proxy_url:
            proxies = {
                "http": self.proxy_url,
                "https": self.proxy_url
            }
        
        try:
            # 发送请求
            logger.info(f"开始调用Gemini API生成图片")
            response = requests.post(
                url, 
                headers=headers, 
                params=params, 
                json=data,
                proxies=proxies,
                timeout=60  # 增加超时时间到60秒
            )
            
            logger.info(f"Gemini API响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # 记录完整响应内容，方便调试
                logger.debug(f"Gemini API响应内容: {result}")
                
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
                        if "inlineData" in part:
                            inline_data = part.get("inlineData", {})
                            if inline_data and "data" in inline_data:
                                # 返回Base64解码后的图片数据
                                image_data = base64.b64decode(inline_data["data"])
                    
                    if not image_data:
                        logger.error(f"API响应中没有找到图片数据: {result}")
                    
                    return image_data, text_response
                
                logger.error(f"未找到生成的图片数据: {result}")
                return None, None
            else:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return None, None
        except Exception as e:
            logger.error(f"API调用异常: {str(e)}")
            logger.exception(e)
            return None, None
    
    def _edit_image(self, prompt: str, image_data: bytes, conversation_history: List[Dict] = None) -> Tuple[Optional[bytes], Optional[str]]:
        """调用Gemini API编辑图片，返回图片数据和文本响应"""
        url = f"{self.base_url}/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent"
        headers = {
            "Content-Type": "application/json",
        }
        
        params = {
            "key": self.api_key
        }
        
        # 将图片数据转换为Base64编码
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        
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
                                "text": prompt
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
        else:
            # 无会话历史，直接使用提示和图片
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
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
        
        # 创建代理配置
        proxies = None
        if self.enable_proxy and self.proxy_url:
            proxies = {
                "http": self.proxy_url,
                "https": self.proxy_url
            }
        
        try:
            # 发送请求
            logger.info(f"开始调用Gemini API编辑图片")
            response = requests.post(
                url, 
                headers=headers, 
                params=params, 
                json=data,
                proxies=proxies,
                timeout=60  # 增加超时时间到60秒
            )
            
            logger.info(f"Gemini API响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # 记录完整响应内容，方便调试
                logger.debug(f"Gemini API响应内容: {result}")
                
                # 检查是否有内容安全问题
                candidates = result.get("candidates", [])
                if candidates and len(candidates) > 0:
                    finish_reason = candidates[0].get("finishReason", "")
                    if finish_reason == "IMAGE_SAFETY":
                        logger.warning("Gemini API返回IMAGE_SAFETY，图片内容可能违反安全政策")
                        return None, json.dumps(result)  # 返回整个响应作为错误信息
                    
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
                        if "inlineData" in part:
                            inline_data = part.get("inlineData", {})
                            if inline_data and "data" in inline_data:
                                # 返回Base64解码后的图片数据
                                image_data = base64.b64decode(inline_data["data"])
                    
                    if not image_data:
                        logger.error(f"API响应中没有找到图片数据: {result}")
                    
                    return image_data, text_response
                
                logger.error(f"未找到编辑后的图片数据: {result}")
                return None, None
            else:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return None, None
        except Exception as e:
            logger.error(f"API调用异常: {str(e)}")
            logger.exception(e)
            return None, None
    
    def _translate_gemini_message(self, text: str) -> str:
        """将Gemini API的英文消息翻译成中文"""
        # 内容安全过滤消息
        if "finishReason" in text and "IMAGE_SAFETY" in text:
            return "抱歉，您的请求可能违反了内容安全政策，无法生成或编辑图片。请尝试修改您的描述，提供更为安全、合规的内容。"
        
        # 处理API响应中的特定错误
        if "finishReason" in text:
            return "抱歉，图片处理失败，请尝试其他描述或稍后再试。"
            
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
    
    def _load_config_template(self):
        """加载配置模板"""
        try:
            template_path = os.path.join(os.path.dirname(__file__), "config.json.template")
            if os.path.exists(template_path):
                with open(template_path, "r", encoding="utf-8") as f:
                    plugin_conf = json.load(f)
                    return plugin_conf
        except Exception as e:
            logger.exception(e)
            return self.DEFAULT_CONFIG
    
    def get_help_text(self, verbose=False, **kwargs):
        help_text = "基于Google Gemini的图像生成插件\n"
        help_text += "支持以下命令：\n"
        help_text += f"1. 生成图片：{' 或 '.join(self.commands)} [描述]\n"
        help_text += f"2. 编辑图片：{' 或 '.join(self.edit_commands)} [描述]\n"
        help_text += f"3. 结束对话：{' 或 '.join(self.exit_commands)}\n\n"
        
        if verbose:
            help_text += "使用说明：\n"
            help_text += "- 生成图片后会开始一个会话，可以通过发送命令继续修改图片\n"
            help_text += "- 每个会话的有效期为10分钟，超时需要重新开始\n"
            help_text += "- 发送结束对话命令可以立即结束当前会话\n"
        
        return help_text 
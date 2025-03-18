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
    version="1.0.0",
    author="Lingyuzhou",
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
        "admins": [],
        "enable_proxy": False,
        "proxy_url": "",
        "use_proxy_service": True,
        "proxy_service_url": ""
    }

    def __init__(self):
        """初始化插件配置"""
        try:
            super().__init__()
            
            # 载入配置
            self.config = super().load_config() or self._load_config_template()
            
            # 使用默认配置初始化
            for key, default_value in self.DEFAULT_CONFIG.items():
                if key not in self.config:
                    self.config[key] = default_value
            
            # 设置配置参数
            self.enable = self.config.get("enable", True)
            self.api_key = self.config.get("gemini_api_key", "")
            
            # 模型配置
            self.model = self.config.get("model", "gemini-2.0-flash-exp-image-generation")
            
            # 获取命令配置
            self.commands = self.config.get("commands", ["#生成图片", "#画图", "#图片生成"])
            self.edit_commands = self.config.get("edit_commands", ["#编辑图片", "#修改图片"])
            self.exit_commands = self.config.get("exit_commands", ["#结束对话", "#退出对话", "#关闭对话", "#结束"])
            
            # 获取积分配置
            self.enable_points = self.config.get("enable_points", False)
            self.generate_cost = self.config.get("generate_image_cost", 10)
            self.edit_cost = self.config.get("edit_image_cost", 15)
            
            # 获取图片保存配置
            self.save_path = self.config.get("save_path", "temp")
            self.save_dir = os.path.join(os.path.dirname(__file__), self.save_path)
            os.makedirs(self.save_dir, exist_ok=True)
            
            # 获取管理员列表
            self.admins = self.config.get("admins", [])
            
            # 获取代理配置
            self.enable_proxy = self.config.get("enable_proxy", False)
            self.proxy_url = self.config.get("proxy_url", "")
            
            # 获取代理服务配置
            self.use_proxy_service = self.config.get("use_proxy_service", True)
            self.proxy_service_url = self.config.get("proxy_service_url", "")
            
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
            
            logger.info("GeminiImage插件初始化成功")
            if self.enable_proxy:
                logger.info(f"GeminiImage插件已启用代理: {self.proxy_url}")
            
        except Exception as e:
            logger.error(f"GeminiImage插件初始化失败: {str(e)}")
            logger.exception(e)
            self.enable = False
    
    def on_handle_context(self, e_context: EventContext):
        """处理消息事件"""
        if not self.enable:
            return
        
        context = e_context['context']
        
        # 清理过期的会话和图片缓存
        self._cleanup_expired_conversations()
        self._cleanup_image_cache()
        
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
            
            # 获取图片数据
            image_data = None
            
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
            
            # 如果获取到图片数据，进行处理
            if image_data and len(image_data) > 1000:  # 确保数据大小合理
                try:
                    # 验证是否为有效的图片数据
                    Image.open(BytesIO(image_data))
                    
                    # 保存图片到缓存
                    self.image_cache[session_id] = {
                        "content": image_data,
                        "timestamp": time.time()
                    }
                    logger.info(f"成功缓存图片数据，大小: {len(image_data)} 字节，缓存键: {session_id}")
                    
                    # 静默处理图片，不发送任何提示消息
                except Exception as e:
                    logger.error(f"验证图片数据失败: {str(e)}")
            else:
                logger.warning(f"未获取到有效的图片数据或数据太小: {image_data[:20] if image_data else 'None'}")
    
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
    
    def _generate_image(self, prompt: str, conversation_history: List[Dict] = None) -> Tuple[Optional[bytes], Optional[str]]:
        """调用Gemini API生成图片，返回图片数据和文本响应"""
        # 根据配置决定使用直接调用还是通过代理服务调用
        if self.use_proxy_service and self.proxy_service_url:
            # 使用代理服务调用API
            url = f"{self.proxy_service_url.rstrip('/')}/v1beta/models/{self.model}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"  # 使用Bearer认证方式
            }
            params = {}  # 不需要在URL参数中传递API密钥
        else:
            # 直接调用Google API
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
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
        if self.enable_proxy and self.proxy_url and not self.use_proxy_service:
            # 只有在直接调用Google API且启用了代理时才使用代理
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
                # 先记录响应内容，便于调试
                response_text = response.text
                logger.debug(f"Gemini API原始响应内容长度: {len(response_text)}, 前100个字符: {response_text[:100] if response_text else '空'}")
                
                # 检查响应内容是否为空
                if not response_text.strip():
                    logger.error("Gemini API返回了空响应")
                    return None, "API返回了空响应，请检查网络连接或代理服务配置"
                
                try:
                    result = response.json()
                    # 记录解析后的JSON结构
                    logger.debug(f"Gemini API响应JSON结构: {result}")
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON解析错误: {str(json_err)}, 响应内容: {response_text[:200]}")
                    # 检查是否是代理服务问题
                    if self.use_proxy_service:
                        logger.error("可能是代理服务配置问题，尝试禁用代理服务或检查代理服务实现")
                        return None, "API响应格式错误，可能是代理服务配置问题。请检查代理服务实现或暂时禁用代理服务。"
                    return None, f"API响应格式错误: {str(json_err)}"
                
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
            elif response.status_code == 400:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return None, "API调用失败，请检查请求参数或网络连接"
            elif response.status_code == 401:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return None, "API调用失败，请检查API密钥或代理服务配置"
            elif response.status_code == 403:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return None, "API调用失败，请检查API密钥或代理服务配置"
            elif response.status_code == 429:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return None, "API调用失败，请稍后再试或检查代理服务配置"
            else:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return None, "API调用失败，请检查网络连接或代理服务配置"
        except Exception as e:
            logger.error(f"API调用异常: {str(e)}")
            logger.exception(e)
            return None, f"API调用异常: {str(e)}"
    
    def _edit_image(self, prompt: str, image_data: bytes, conversation_history: List[Dict] = None) -> Tuple[Optional[bytes], Optional[str]]:
        """调用Gemini API编辑图片，返回图片数据和文本响应"""
        # 根据配置决定使用直接调用还是通过代理服务调用
        if self.use_proxy_service and self.proxy_service_url:
            # 使用代理服务调用API
            url = f"{self.proxy_service_url.rstrip('/')}/v1beta/models/{self.model}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"  # 使用Bearer认证方式
            }
            params = {}  # 不需要在URL参数中传递API密钥
        else:
            # 直接调用Google API
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
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
        if self.enable_proxy and self.proxy_url and not self.use_proxy_service:
            # 只有在直接调用Google API且启用了代理时才使用代理
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
                # 先记录响应内容，便于调试
                response_text = response.text
                logger.debug(f"Gemini API原始响应内容长度: {len(response_text)}, 前100个字符: {response_text[:100] if response_text else '空'}")
                
                # 检查响应内容是否为空
                if not response_text.strip():
                    logger.error("Gemini API返回了空响应")
                    return None, "API返回了空响应，请检查网络连接或代理服务配置"
                
                try:
                    result = response.json()
                    # 记录解析后的JSON结构
                    logger.debug(f"Gemini API响应JSON结构: {result}")
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON解析错误: {str(json_err)}, 响应内容: {response_text[:200]}")
                    # 检查是否是代理服务问题
                    if self.use_proxy_service:
                        logger.error("可能是代理服务配置问题，尝试禁用代理服务或检查代理服务实现")
                        return None, "API响应格式错误，可能是代理服务配置问题。请检查代理服务实现或暂时禁用代理服务。"
                    return None, f"API响应格式错误: {str(json_err)}"
                
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
            elif response.status_code == 400:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return None, "API调用失败，请检查请求参数或网络连接"
            elif response.status_code == 401:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return None, "API调用失败，请检查API密钥或代理服务配置"
            elif response.status_code == 403:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return None, "API调用失败，请检查API密钥或代理服务配置"
            elif response.status_code == 429:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return None, "API调用失败，请稍后再试或检查代理服务配置"
            else:
                logger.error(f"Gemini API调用失败 (状态码: {response.status_code}): {response.text}")
                return None, "API调用失败，请检查网络连接或代理服务配置"
        except Exception as e:
            logger.error(f"API调用异常: {str(e)}")
            logger.exception(e)
            return None, f"API调用异常: {str(e)}"
    
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
            return {
                "enable": True,
                "gemini_api_key": "",
                "model": "gemini-2.0-flash-exp-image-generation",
                "commands": ["#生成图片", "#画图", "#图片生成"],
                "edit_commands": ["#编辑图片", "#修改图片"],
                "exit_commands": ["#结束对话", "#退出对话", "#关闭对话", "#结束"],
                "enable_points": False,
                "generate_image_cost": 10,
                "edit_image_cost": 15,
                "save_path": "temp",
                "admins": [],
                "enable_proxy": False,
                "proxy_url": "",
                "use_proxy_service": True,
                "proxy_service_url": ""
            }
    
    def get_help_text(self, verbose=False, **kwargs):
        help_text = "基于Google Gemini的图像生成插件\n"
        help_text += "可以生成和编辑图片，支持连续对话\n\n"
        help_text += "使用方法：\n"
        help_text += f"1. 生成图片：发送 {self.commands[0]} + 描述，例如：{self.commands[0]} 一只可爱的猫咪\n"
        help_text += f"2. 编辑图片：发送 {self.edit_commands[0]} + 描述 + 图片，例如：{self.edit_commands[0]} 把猫咪变成蓝色\n"
        help_text += f"3. 结束会话：发送 {self.exit_commands[0]}\n\n"
        
        if verbose:
            help_text += "当前配置：\n"
            help_text += f"当前模型：{self.model}\n"
            help_text += f"生成图片命令：{', '.join(self.commands)}\n"
            help_text += f"编辑图片命令：{', '.join(self.edit_commands)}\n"
            help_text += f"结束会话命令：{', '.join(self.exit_commands)}\n"
            
            if self.enable_proxy:
                help_text += "已启用HTTP代理\n"
            
            if self.use_proxy_service:
                help_text += "已启用代理服务\n"
        
        return help_text

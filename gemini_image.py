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
        "commands": ["g生成图片", "g画图", "g画一个"],
        "edit_commands": ["g编辑图片", "g改图"],
        "reference_edit_commands": ["g参考图", "g编辑参考图"],
        "merge_commands": ["g融图"],
        "image_reverse_commands": ["g反推提示", "g反推"],
        "image_analysis_commands": ["g分析图片", "g识图"],
        "follow_up_commands": ["g追问"],
        "exit_commands": ["g结束对话", "g结束"],
        "enable_points": False,
        "generate_image_cost": 10,
        "edit_image_cost": 15,
        "save_path": "temp",
        "admins": [],
        "enable_proxy": False,
        "proxy_url": "",
        "use_proxy_service": True,
        "proxy_service_url": "",
        "translate_api_base": "https://open.bigmodel.cn/api/paas/v4",
        "translate_api_key": "",
        "translate_model": "glm-4-flash",
        "enable_translate": True,
        "translate_on_commands": ["g开启翻译", "g启用翻译"],
        "translate_off_commands": ["g关闭翻译", "g禁用翻译"],
        "reverse_prompt": ""
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
            self.commands = self.config.get("commands", ["g生成图片", "g画图", "g画一个"])
            self.edit_commands = self.config.get("edit_commands", ["g编辑图片", "g改图"])
            self.reference_edit_commands = self.config.get("reference_edit_commands", ["g参考图", "g编辑参考图"])
            self.image_reverse_commands = self.config.get("image_reverse_commands", ["g反推提示", "g反推"])
            self.image_analysis_commands = self.config.get("image_analysis_commands", ["g分析图片", "g识图"])
            self.follow_up_commands = self.config.get("follow_up_commands", ["g追问"])
            self.exit_commands = self.config.get("exit_commands", ["g结束对话", "g结束"])
            
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
            
            # 获取翻译API配置
            self.enable_translate = self.config.get("enable_translate", True)
            self.translate_api_base = self.config.get("translate_api_base", "https://open.bigmodel.cn/api/paas/v4")
            self.translate_api_key = self.config.get("translate_api_key", "")
            self.translate_model = self.config.get("translate_model", "glm-4-flash")
            
            # 获取翻译控制命令配置
            self.translate_on_commands = self.config.get("translate_on_commands", ["g开启翻译", "g启用翻译"])
            self.translate_off_commands = self.config.get("translate_off_commands", ["g关闭翻译", "g禁用翻译"])
            
            # 用户翻译设置缓存，用于存储每个用户的翻译设置
            self.user_translate_settings = {}  # 用户ID -> 是否启用翻译
            
            # 初始化会话状态，用于保存上下文
            self.conversations = defaultdict(list)  # 用户ID -> 对话历史列表
            self.conversation_expiry = 600  # 会话过期时间(秒)
            self.last_conversation_time = {}  # 用户ID -> 最后对话时间
            self.last_images = {}  # 用户ID -> 最后生成的图片路径
            self.waiting_for_reference_image = {}  # 用户ID -> 等待参考图片的提示词
            self.waiting_for_reference_image_time = {}  # 用户ID -> 开始等待参考图片的时间戳
            self.reference_image_wait_timeout = 180  # 等待参考图片的超时时间(秒)，3分钟
            
            # 初始化图片分析状态
            self.waiting_for_reverse_image = {}  # 用户ID -> 是否等待反推图片
            self.waiting_for_reverse_image_time = {}  # 用户ID -> 开始等待反推图片的时间戳
            self.reverse_image_wait_timeout = 180  # 等待反推图片的超时时间(秒)，3分钟
            
            # 初始化识图状态
            self.waiting_for_analysis_image = {}  # 用户ID -> 等待识图的问题
            self.waiting_for_analysis_image_time = {}  # 用户ID -> 开始等待识图的时间戳
            self.analysis_image_wait_timeout = 180  # 等待识图的超时时间(秒)，3分钟
            
            # 初始化图片缓存，用于存储用户上传的图片
            self.image_cache = {}  # 会话ID/用户ID -> {"data": 图片数据, "timestamp": 时间戳}
            self.image_cache_timeout = 600  # 图片缓存过期时间(秒)
            
            # 初始化追问状态
            self.last_analysis_image = {}  # 用户ID -> 最后一次识图的图片数据
            self.last_analysis_time = {}  # 用户ID -> 最后一次识图的时间戳
            self.follow_up_timeout = 180  # 追问超时时间(秒)，3分钟
            
            # 获取图片分析提示词
            self.reverse_prompt = self.config.get("reverse_prompt", "请详细分析这张图片的内容，包括主要对象、场景、风格、颜色等关键特征。如果图片包含文字，也请提取出来。请用简洁清晰的中文进行描述。")
            
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
        
        # 获取用户ID
        user_id = context.get("from_user_id")
        session_id = context.get("session_id")
        is_group = context.get("isgroup", False)
        
        # 获取消息对象
        msg = None
        if 'msg' in context.kwargs:
            msg = context.kwargs['msg']
            # 在群聊中，优先使用actual_user_id作为用户标识
            if is_group and hasattr(msg, 'actual_user_id') and msg.actual_user_id:
                user_id = msg.actual_user_id
                logger.info(f"群聊中使用actual_user_id作为用户ID: {user_id}")
            elif not is_group:
                # 私聊中使用from_user_id
                if hasattr(msg, 'from_user_id') and msg.from_user_id:
                    user_id = msg.from_user_id
                    logger.info(f"私聊中使用from_user_id作为用户ID: {user_id}")
        
        # 会话标识: 用户ID
        conversation_key = user_id
        
        # 处理图片消息 - 用于缓存用户发送的图片
        if context.type == ContextType.IMAGE:
            self._handle_image_message(e_context)
            return
            
        # 处理文本消息
        if context.type != ContextType.TEXT:
            return
        
        content = context.content.strip()
        
        # 检查是否是反推提示词命令
        for cmd in self.image_reverse_commands:
            if content == cmd:
                # 设置等待图片状态
                self.waiting_for_reverse_image[user_id] = True
                self.waiting_for_reverse_image_time[user_id] = time.time()
                
                # 提示用户上传图片
                reply = Reply(ReplyType.TEXT, "请在3分钟内发送需要反推提示词的图片")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
                
        # 检查是否是识图命令
        for cmd in self.image_analysis_commands:
            # 检查是否包含问题
            if content.startswith(cmd):
                question = content[len(cmd):].strip()
                # 设置等待图片状态，并保存问题
                self.waiting_for_analysis_image[user_id] = question if question else "分析这张图片的内容，包括主要对象、场景、风格、颜色等关键特征，用简洁清晰的中文进行描述。"
                self.waiting_for_analysis_image_time[user_id] = time.time()
                
                # 提示用户上传图片
                reply = Reply(ReplyType.TEXT, "请在3分钟内发送需要gemini识别的图片")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
                
        # 检查是否是追问命令
        for cmd in self.follow_up_commands:
            if content.startswith(cmd):
                # 检查是否有最近的识图记录
                if user_id not in self.last_analysis_image or user_id not in self.last_analysis_time:
                    reply = Reply(ReplyType.TEXT, "没有找到最近的识图记录，请先使用识图功能")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 检查是否超时
                if time.time() - self.last_analysis_time[user_id] > self.follow_up_timeout:
                    # 清理状态
                    del self.last_analysis_image[user_id]
                    del self.last_analysis_time[user_id]
                    
                    reply = Reply(ReplyType.TEXT, "追问超时，请重新使用识图功能")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 提取追问问题
                question = content[len(cmd):].strip() if len(content) > len(cmd) else "请继续分析这张图片"
                # 添加中文回答要求
                question = question + "，请用简洁的中文进行回答。"
                
                try:
                    # 调用API分析图片
                    analysis_result = self._analyze_image(self.last_analysis_image[user_id], question)
                    if analysis_result:
                        # 更新时间戳
                        self.last_analysis_time[user_id] = time.time()
                        
                        # 添加追问提示
                        analysis_result += "\n💬3min内输入g追问+问题，可继续追问"
                        reply = Reply(ReplyType.TEXT, analysis_result)
                    else:
                        reply = Reply(ReplyType.TEXT, "图片分析失败，请稍后重试")
                except Exception as e:
                    logger.error(f"处理追问请求异常: {str(e)}")
                    logger.exception(e)
                    reply = Reply(ReplyType.TEXT, f"图片分析失败: {str(e)}")
                
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
        
        # 检查是否是翻译控制命令
        for cmd in self.translate_on_commands:
            if content == cmd:
                # 启用翻译
                self.user_translate_settings[user_id] = True
                reply = Reply(ReplyType.TEXT, "已开启前置翻译功能，接下来的图像生成和编辑将自动将中文提示词翻译成英文")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
        
        for cmd in self.translate_off_commands:
            if content == cmd:
                # 禁用翻译
                self.user_translate_settings[user_id] = False
                reply = Reply(ReplyType.TEXT, "已关闭前置翻译功能，接下来的图像生成和编辑将直接使用原始中文提示词")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
        
        # 检查是否在等待用户上传参考图片
        if user_id in self.waiting_for_reference_image:
            # 检查是否超时
            current_time = time.time()
            start_time = self.waiting_for_reference_image_time.get(user_id, 0)
            
            if current_time - start_time > self.reference_image_wait_timeout:
                # 超过3分钟，自动结束等待
                logger.info(f"用户 {user_id} 等待上传参考图片超时，自动结束流程")
                prompt = self.waiting_for_reference_image[user_id]
                
                # 清除等待状态
                del self.waiting_for_reference_image[user_id]
                if user_id in self.waiting_for_reference_image_time:
                    del self.waiting_for_reference_image_time[user_id]
                
                # 发送超时提示
                reply = Reply(ReplyType.TEXT, f"等待上传参考图片超时（超过{self.reference_image_wait_timeout//60}分钟），已自动取消操作。如需继续，请重新发送参考图编辑命令。")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
            
            # 获取之前保存的提示词
            prompt = self.waiting_for_reference_image[user_id]
            
            # 获取消息对象
            msg = None
            if 'msg' in context.kwargs:
                msg = context.kwargs['msg']
            
            # 先检查context.kwargs中是否有image_base64
            image_base64 = context.kwargs.get("image_base64")
            
            # 如果没有image_base64，使用统一的图片获取方法
            if not image_base64:
                # 使用统一的图片获取方法获取图片数据
                image_data = self._get_image_data(msg, "")  # 传入空字符串，让方法尝试从msg中获取图片
                
                # 如果获取到图片数据，转换为base64
                if image_data and len(image_data) > 1000:
                    try:
                        # 验证图片数据是否有效
                        Image.open(BytesIO(image_data))
                        image_base64 = base64.b64encode(image_data).decode('utf-8')
                        logger.info(f"成功获取图片数据并转换为base64，大小: {len(image_data)} 字节")
                    except Exception as img_err:
                        logger.error(f"获取的图片数据无效: {img_err}")
            
            # 如果成功获取到图片数据
            if image_base64:
                # 清除等待状态
                del self.waiting_for_reference_image[user_id]
                if user_id in self.waiting_for_reference_image_time:
                    del self.waiting_for_reference_image_time[user_id]
                
                # 发送成功获取图片的提示
                success_reply = Reply(ReplyType.TEXT, "成功获取图片，正在处理中...")
                e_context["reply"] = success_reply
                e_context.action = EventAction.BREAK_PASS
                e_context["channel"].send(success_reply, e_context["context"])
                
                # 处理参考图片编辑
                self._handle_reference_image_edit(e_context, user_id, prompt, image_base64)
                return
            else:
                # 用户没有上传图片，提醒用户
                reply = Reply(ReplyType.TEXT, "请上传一张图片作为参考图进行编辑。如果想取消操作，请发送\"g结束对话\"")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
        
        # 检查是否是结束对话命令
        if content in self.exit_commands:
            if conversation_key in self.conversations:
                # 清除会话数据
                del self.conversations[conversation_key]
                if conversation_key in self.last_conversation_time:
                    del self.last_conversation_time[conversation_key]
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
                    
                    # 翻译提示词
                    translated_prompt = self._translate_prompt(prompt, user_id)
                    
                    # 生成图片
                    image_data, text_response = self._generate_image(translated_prompt, conversation_history)
                    
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
                        self.last_conversation_time[conversation_key] = time.time()
                        
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
                        
                        # 翻译提示词
                        translated_prompt = self._translate_prompt(prompt, user_id)
                        
                        # 编辑图片
                        result_image, text_response = self._edit_image(translated_prompt, image_data, conversation_history)
                        
                        if result_image:
                            # 保存编辑后的图片
                            reply_text = text_response if text_response else "图片编辑成功！"
                            if not conversation_history or len(conversation_history) <= 2:  # 如果是新会话
                                reply_text += f"（已开始图像对话，可以继续发送命令修改图片。需要结束时请发送\"{self.exit_commands[0]}\"）"
                            
                            # 将回复文本添加到文件名中
                            clean_text = reply_text.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_")
                            clean_text = clean_text[:30] + "..." if len(clean_text) > 30 else clean_text
                            
                            image_path = os.path.join(self.save_dir, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}_{clean_text}.png")
                            with open(image_path, "wb") as f:
                                f.write(result_image)
                            
                            # 保存最后生成的图片路径
                            self.last_images[conversation_key] = image_path
                            
                            # 添加用户提示到会话
                            user_message = {"role": "user", "parts": [{"text": prompt}]}
                            conversation_history.append(user_message)
                            
                            # 添加助手回复到会话
                            assistant_message = {
                                "role": "model", 
                                "parts": [
                                    {"text": text_response if text_response else "我已编辑了图片"},
                                    {"image_url": image_path}
                                ]
                            }
                            conversation_history.append(assistant_message)
                            
                            # 限制会话历史长度
                            if len(conversation_history) > 10:  # 保留最近5轮对话
                                conversation_history = conversation_history[-10:]
                            
                            # 更新会话时间戳
                            self.last_conversation_time[conversation_key] = time.time()
                            
                            # 准备回复文本
                            reply_text = text_response if text_response else "图片编辑成功！"
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
                                reply = Reply(ReplyType.TEXT, "图片编辑失败，请稍后再试或修改提示词")
                                e_context["reply"] = reply
                                e_context.action = EventAction.BREAK_PASS
                    except Exception as e:
                        logger.error(f"编辑图片失败: {str(e)}")
                        logger.exception(e)
                        reply = Reply(ReplyType.TEXT, f"编辑图片失败: {str(e)}")
                        e_context["reply"] = reply
                        e_context.action = EventAction.BREAK_PASS
                    return
                else:
                    # 没有找到缓存的图片，检查是否有最后生成的图片
                    if conversation_key in self.last_images:
                        last_image_path = self.last_images[conversation_key]
                        if os.path.exists(last_image_path):
                            try:
                                # 发送处理中消息
                                processing_reply = Reply(ReplyType.TEXT, "正在编辑图片，请稍候...")
                                e_context["reply"] = processing_reply
                                
                                # 读取图片数据
                                with open(last_image_path, "rb") as f:
                                    image_data = f.read()
                                
                                # 获取会话上下文
                                conversation_history = self.conversations[conversation_key]
                                
                                # 翻译提示词
                                translated_prompt = self._translate_prompt(prompt, user_id)
                                
                                # 编辑图片
                                result_image, text_response = self._edit_image(translated_prompt, image_data, conversation_history)
                                
                                if result_image:
                                    # 保存编辑后的图片
                                    reply_text = text_response if text_response else "图片编辑成功！"
                                    
                                    # 将回复文本添加到文件名中
                                    clean_text = reply_text.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_")
                                    clean_text = clean_text[:30] + "..." if len(clean_text) > 30 else clean_text
                                    
                                    image_path = os.path.join(self.save_dir, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:8]}_{clean_text}.png")
                                    with open(image_path, "wb") as f:
                                        f.write(result_image)
                                    
                                    # 保存最后生成的图片路径
                                    self.last_images[conversation_key] = image_path
                                    
                                    # 添加用户提示到会话
                                    user_message = {"role": "user", "parts": [{"text": prompt}]}
                                    conversation_history.append(user_message)
                                    
                                    # 添加助手回复到会话
                                    assistant_message = {
                                        "role": "model", 
                                        "parts": [
                                            {"text": text_response if text_response else "我已编辑了图片"},
                                            {"image_url": image_path}
                                        ]
                                    }
                                    conversation_history.append(assistant_message)
                                    
                                    # 限制会话历史长度
                                    if len(conversation_history) > 10:  # 保留最近5轮对话
                                        conversation_history = conversation_history[-10:]
                                    
                                    # 更新会话时间戳
                                    self.last_conversation_time[conversation_key] = time.time()
                                    
                                    # 准备回复文本
                                    reply_text = text_response if text_response else "图片编辑成功！"
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
                                        reply = Reply(ReplyType.TEXT, "图片编辑失败，请稍后再试或修改提示词")
                                        e_context["reply"] = reply
                                        e_context.action = EventAction.BREAK_PASS
                            except Exception as e:
                                logger.error(f"编辑图片失败: {str(e)}")
                                logger.exception(e)
                                reply = Reply(ReplyType.TEXT, f"编辑图片失败: {str(e)}")
                                e_context["reply"] = reply
                                e_context.action = EventAction.BREAK_PASS
                            return
                        else:
                            # 图片文件已丢失
                            reply = Reply(ReplyType.TEXT, "找不到之前生成的图片，请重新生成图片后再编辑")
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                            return
                    else:
                        # 没有之前生成的图片
                        reply = Reply(ReplyType.TEXT, "请先使用生成图片命令生成一张图片，或者上传一张图片后再编辑")
                        e_context["reply"] = reply
                        e_context.action = EventAction.BREAK_PASS
                        return
                        
        # 检查是否是参考图编辑命令
        for cmd in self.reference_edit_commands:
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
                
                # 检查是否启用积分系统且用户积分不足
                if self.enable_points and user_id not in self.admins:
                    user_points = self.get_user_points(user_id)
                    if user_points < self.edit_image_cost:
                        reply = Reply(ReplyType.TEXT, f"您的积分不足，编辑图片需要{self.edit_image_cost}积分，您当前有{user_points}积分")
                        e_context["reply"] = reply
                        e_context.action = EventAction.BREAK_PASS
                        return
                
                # 记录用户正在等待上传参考图片
                self.waiting_for_reference_image[user_id] = prompt
                self.waiting_for_reference_image_time[user_id] = time.time()
                
                # 记录日志
                logger.info(f"用户 {user_id} 开始等待上传参考图片，提示词: {prompt}")
                
                # 发送提示消息
                reply = Reply(ReplyType.TEXT, "请发送需要编辑的参考图片")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
    
    def _handle_image_message(self, e_context: EventContext):
        """处理图片消息，缓存图片数据以备后续编辑使用"""
        context = e_context['context']
        session_id = context["session_id"]
        is_group = context.get("isgroup", False)
        
        # 获取发送者ID，确保群聊和单聊场景都能正确缓存
        sender_id = context.get("from_user_id")  # 默认使用from_user_id
        
        if 'msg' in context.kwargs:
            msg = context.kwargs['msg']
            
            # 在群聊中，优先使用actual_user_id作为用户标识
            if is_group and hasattr(msg, 'actual_user_id') and msg.actual_user_id:
                sender_id = msg.actual_user_id
                logger.info(f"群聊中使用actual_user_id作为发送者ID: {sender_id}")
            elif not is_group:
                # 私聊中使用from_user_id或session_id
                if hasattr(msg, 'from_user_id') and msg.from_user_id:
                    sender_id = msg.from_user_id
                    logger.info(f"私聊中使用from_user_id作为发送者ID: {sender_id}")
                else:
                    sender_id = session_id
                    logger.info(f"私聊中使用session_id作为发送者ID: {sender_id}")
            
            # 使用统一的图片获取方法获取图片数据
            image_data = self._get_image_data(msg, "")
            
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
                    
                    # 检查是否有用户在等待上传参考图片
                    if sender_id and sender_id in self.waiting_for_reference_image:
                        prompt = self.waiting_for_reference_image[sender_id]
                        logger.info(f"检测到用户 {sender_id} 正在等待上传参考图片，提示词: {prompt}")
                        
                        # 将图片转换为base64
                        image_base64 = base64.b64encode(image_data).decode('utf-8')
                        
                        # 清除等待状态
                        del self.waiting_for_reference_image[sender_id]
                        if sender_id in self.waiting_for_reference_image_time:
                            del self.waiting_for_reference_image_time[sender_id]
                        
                        # 发送成功获取图片的提示
                        success_reply = Reply(ReplyType.TEXT, "成功获取图片，正在处理中...")
                        e_context["reply"] = success_reply
                        e_context.action = EventAction.BREAK_PASS
                        e_context["channel"].send(success_reply, e_context["context"])
                        
                        # 处理参考图片编辑
                        self._handle_reference_image_edit(e_context, sender_id, prompt, image_base64)
                        return
                    # 检查是否有用户在等待反推提示词
                    elif sender_id and sender_id in self.waiting_for_reverse_image:
                        # 检查是否超时
                        if time.time() - self.waiting_for_reverse_image_time[sender_id] > self.reverse_image_wait_timeout:
                            # 清理状态
                            del self.waiting_for_reverse_image[sender_id]
                            del self.waiting_for_reverse_image_time[sender_id]
                            
                            reply = Reply(ReplyType.TEXT, "图片上传超时，请重新发送反推提示词命令")
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                            return
                        
                        try:
                            # 调用API分析图片
                            reverse_result = self._reverse_image(image_data)
                            if reverse_result:
                                reply = Reply(ReplyType.TEXT, reverse_result)
                            else:
                                reply = Reply(ReplyType.TEXT, "图片分析失败，请稍后重试")
                            
                            # 清理状态
                            del self.waiting_for_reverse_image[sender_id]
                            del self.waiting_for_reverse_image_time[sender_id]
                            
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                            return
                        except Exception as e:
                            logger.error(f"处理反推请求异常: {str(e)}")
                            logger.exception(e)
                            
                            # 清理状态
                            del self.waiting_for_reverse_image[sender_id]
                            del self.waiting_for_reverse_image_time[sender_id]
                            
                            reply = Reply(ReplyType.TEXT, f"图片分析失败: {str(e)}")
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                            return
                    # 检查是否有用户在等待识图
                    elif sender_id and sender_id in self.waiting_for_analysis_image:
                        # 检查是否超时
                        if time.time() - self.waiting_for_analysis_image_time[sender_id] > self.analysis_image_wait_timeout:
                            # 清理状态
                            del self.waiting_for_analysis_image[sender_id]
                            del self.waiting_for_analysis_image_time[sender_id]
                            
                            reply = Reply(ReplyType.TEXT, "图片上传超时，请重新发送识图命令")
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                            return
                        
                        try:
                            # 获取用户的问题或默认提示词
                            question = self.waiting_for_analysis_image[sender_id]
                            
                            # 调用API分析图片
                            analysis_result = self._analyze_image(image_data, question)
                            if analysis_result:
                                # 缓存图片数据和时间戳，用于后续追问
                                self.last_analysis_image[sender_id] = image_data
                                self.last_analysis_time[sender_id] = time.time()
                                
                                # 添加追问提示
                                analysis_result += "\n💬3min内输入g追问+问题，可继续追问"
                                reply = Reply(ReplyType.TEXT, analysis_result)
                            else:
                                reply = Reply(ReplyType.TEXT, "图片分析失败，请稍后重试")
                            
                            # 清理状态
                            del self.waiting_for_analysis_image[sender_id]
                            del self.waiting_for_analysis_image_time[sender_id]
                            
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                            return
                        except Exception as e:
                            logger.error(f"处理反推请求异常: {str(e)}")
                            logger.exception(e)
                            
                            # 清理状态
                            del self.waiting_for_reverse_image[sender_id]
                            del self.waiting_for_reverse_image_time[sender_id]
                            
                            reply = Reply(ReplyType.TEXT, f"图片分析失败: {str(e)}")
                            e_context["reply"] = reply
                            e_context.action = EventAction.BREAK_PASS
                            return
                except Exception as img_err:
                    logger.error(f"图片验证失败: {str(img_err)}")
                    reply = Reply(ReplyType.TEXT, "无法处理图片，请确保上传的是有效的图片文件。")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
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
        
        for key, timestamp in self.last_conversation_time.items():
            if current_time - timestamp > self.conversation_expiry:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self.conversations:
                del self.conversations[key]
            if key in self.last_conversation_time:
                del self.last_conversation_time[key]
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
    
    def _translate_prompt(self, prompt: str, user_id: str = None) -> str:
        """
        将中文提示词翻译成英文
        
        Args:
            prompt: 原始提示词
            user_id: 用户ID，用于获取用户的翻译设置
            
        Returns:
            翻译后的提示词，如果翻译失败则返回原始提示词
        """
        # 如果提示词为空，直接返回
        if not prompt or len(prompt.strip()) == 0:
            return prompt
            
        # 检查全局翻译设置
        if not self.enable_translate:
            return prompt
            
        # 检查用户个人翻译设置（如果有）
        if user_id is not None and user_id in self.user_translate_settings:
            if not self.user_translate_settings[user_id]:
                return prompt
        
        # 检查API密钥是否配置
        if not self.translate_api_key:
            logger.warning("翻译API密钥未配置，将使用原始提示词")
            return prompt
            
        try:
            # 构建请求数据
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.translate_api_key}"
            }
            
            data = {
                "model": self.translate_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的中英翻译专家。你的任务是将用户输入的中文提示词翻译成英文，用于AI图像生成。请确保翻译准确、自然，并保留原始提示词的意图和风格。不要添加任何解释或额外内容，只需提供翻译结果。"
                    },
                    {
                        "role": "user",
                        "content": f"请将以下中文提示词翻译成英文，用于AI图像生成：\n\n{prompt}"
                    }
                ]
            }
            
            # 发送请求
            url = f"{self.translate_api_base}/chat/completions"
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            # 解析响应
            if response.status_code == 200:
                result = response.json()
                translated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # 清理翻译结果，移除可能的引号和多余空格
                translated_text = translated_text.strip('"\'').strip()
                
                if translated_text:
                    logger.info(f"翻译成功: {prompt} -> {translated_text}")
                    return translated_text
            
            logger.warning(f"翻译失败: {response.status_code} {response.text}")
            return prompt
            
        except Exception as e:
            logger.error(f"翻译出错: {str(e)}")
            return prompt
    
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
                "commands": ["g生成图片", "g画图", "g画一个"],
                "edit_commands": ["g编辑图片", "g改图"],
                "reference_edit_commands": ["g参考图", "g编辑参考图"],
                "exit_commands": ["g结束对话", "g结束"],
                "enable_points": False,
                "generate_image_cost": 10,
                "edit_image_cost": 15,
                "save_path": "temp",
                "admins": [],
                "enable_proxy": False,
                "proxy_url": "",
                "use_proxy_service": True,
                "proxy_service_url": "",
                "translate_api_base": "https://open.bigmodel.cn/api/paas/v4",
                "translate_api_key": "",
                "translate_model": "glm-4-flash",
                "enable_translate": True,
                "translate_on_commands": ["g开启翻译", "g启用翻译"],
                "translate_off_commands": ["g关闭翻译", "g禁用翻译"]
            }

    def _get_image_data(self, msg, image_path_or_data):
        """
        统一的图片数据获取方法，参考QwenVision插件的实现
        
        Args:
            msg: 消息对象，可能包含图片数据或路径
            image_path_or_data: 可能是图片路径、URL或二进制数据
            
        Returns:
            bytes: 图片二进制数据，获取失败则返回None
        """
        try:
            # 如果已经是二进制数据，直接返回
            if isinstance(image_path_or_data, bytes):
                logger.debug(f"处理二进制数据，大小: {len(image_path_or_data)} 字节")
                return image_path_or_data
            
            logger.debug(f"开始处理图片，类型: {type(image_path_or_data)}")
            
            # 统一的文件读取函数
            def read_file(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        data = f.read()
                        logger.debug(f"成功读取文件: {file_path}, 大小: {len(data)} 字节")
                        return data
                except Exception as e:
                    logger.error(f"读取文件失败 {file_path}: {e}")
                    return None
            
            # 按优先级尝试不同的读取方式
            # 1. 如果是文件路径，直接读取
            if isinstance(image_path_or_data, str):
                if os.path.isfile(image_path_or_data):
                    data = read_file(image_path_or_data)
                    if data:
                        return data
                
                # 2. 处理URL，尝试下载
                if image_path_or_data.startswith(('http://', 'https://')):
                    try:
                        logger.debug(f"尝试从URL下载图片: {image_path_or_data}")
                        response = requests.get(image_path_or_data, timeout=10)
                        if response.status_code == 200:
                            data = response.content
                            if data and len(data) > 1000:
                                logger.debug(f"从URL下载图片成功，大小: {len(data)} 字节")
                                return data
                    except Exception as e:
                        logger.error(f"从URL下载图片失败: {e}")
                
                # 3. 尝试不同的路径组合
                if image_path_or_data.startswith('tmp/') and not os.path.exists(image_path_or_data):
                    # 尝试使用项目目录
                    project_path = os.path.join(os.path.dirname(__file__), image_path_or_data)
                    if os.path.exists(project_path):
                        data = read_file(project_path)
                        if data:
                            return data
                    
                    # 尝试使用临时目录
                    temp_path = os.path.join("temp", os.path.basename(image_path_or_data))
                    if os.path.exists(temp_path):
                        data = read_file(temp_path)
                        if data:
                            return data
            
            # 4. 从msg对象获取图片数据
            if msg:
                # 4.1 检查file_path属性
                if hasattr(msg, 'file_path') and msg.file_path:
                    file_path = msg.file_path
                    logger.debug(f"从msg.file_path获取到文件路径: {file_path}")
                    data = read_file(file_path)
                    if data:
                        return data
                
                # 4.2 检查msg.content
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, bytes):
                        logger.debug(f"使用msg.content中的二进制内容，大小: {len(msg.content)} 字节")
                        return msg.content
                    elif isinstance(msg.content, str) and os.path.isfile(msg.content):
                        data = read_file(msg.content)
                        if data:
                            return data
                
                # 4.3 尝试使用download_image方法
                if hasattr(msg, 'download_image') and callable(getattr(msg, 'download_image')):
                    try:
                        logger.debug("尝试使用msg.download_image()方法获取图片")
                        image_data = msg.download_image()
                        if image_data and len(image_data) > 1000:
                            logger.debug(f"通过download_image方法获取到图片数据，大小: {len(image_data)} 字节")
                            return image_data
                    except Exception as e:
                        logger.error(f"download_image方法调用失败: {e}")
                
                # 4.4 尝试从msg.img获取
                if hasattr(msg, 'img') and msg.img:
                    image_data = msg.img
                    if image_data and len(image_data) > 1000:
                        logger.debug(f"从msg.img获取到图片数据，大小: {len(image_data)} 字节")
                        return image_data
                
                # 4.5 尝试从msg.msg_data获取
                if hasattr(msg, 'msg_data'):
                    try:
                        msg_data = msg.msg_data
                        if isinstance(msg_data, dict) and 'image' in msg_data:
                            image_data = msg_data['image']
                            if image_data and len(image_data) > 1000:
                                logger.debug(f"从msg_data['image']获取到图片数据，大小: {len(image_data)} 字节")
                                return image_data
                        elif isinstance(msg_data, bytes):
                            image_data = msg_data
                            logger.debug(f"从msg_data(bytes)获取到图片数据，大小: {len(image_data)} 字节")
                            return image_data
                    except Exception as e:
                        logger.error(f"获取msg_data失败: {e}")
                
                # 4.6 微信特殊处理：尝试从_rawmsg获取图片路径
                if hasattr(msg, '_rawmsg') and isinstance(msg._rawmsg, dict):
                    try:
                        rawmsg = msg._rawmsg
                        logger.debug(f"获取到_rawmsg: {type(rawmsg)}")
                        
                        # 检查是否有图片文件路径
                        if 'file' in rawmsg and rawmsg['file']:
                            file_path = rawmsg['file']
                            logger.debug(f"从_rawmsg获取到文件路径: {file_path}")
                            data = read_file(file_path)
                            if data:
                                return data
                    except Exception as e:
                        logger.error(f"处理_rawmsg失败: {e}")
                
                # 4.7 尝试从image_url属性获取
                if hasattr(msg, 'image_url') and msg.image_url:
                    try:
                        image_url = msg.image_url
                        logger.debug(f"从msg.image_url获取图片URL: {image_url}")
                        response = requests.get(image_url, timeout=10)
                        if response.status_code == 200:
                            data = response.content
                            if data and len(data) > 1000:
                                logger.debug(f"从image_url下载图片成功，大小: {len(data)} 字节")
                                return data
                    except Exception as e:
                        logger.error(f"从image_url下载图片失败: {e}")
                
                # 4.8 如果文件未下载，尝试下载 (类似QwenVision的_prepare_fn处理)
                if hasattr(msg, '_prepare_fn') and hasattr(msg, '_prepared') and not msg._prepared:
                    logger.debug("尝试调用msg._prepare_fn()下载图片...")
                    try:
                        msg._prepare_fn()
                        msg._prepared = True
                        time.sleep(1)  # 等待文件准备完成
                        
                        # 再次尝试获取内容
                        if hasattr(msg, 'content'):
                            if isinstance(msg.content, bytes):
                                return msg.content
                            elif isinstance(msg.content, str) and os.path.isfile(msg.content):
                                data = read_file(msg.content)
                                if data:
                                    return data
                    except Exception as e:
                        logger.error(f"调用_prepare_fn下载图片失败: {e}")
            
            logger.error(f"无法获取图片数据: {image_path_or_data}")
            return None
            
        except Exception as e:
            logger.error(f"获取图片数据失败: {e}")
            return None

    def _reverse_image(self, image_data: bytes) -> Optional[str]:
        """调用Gemini API分析图片内容"""
        try:
            # 将图片转换为Base64格式
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            
            # 构建请求数据
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": image_base64
                                }
                            },
                            {
                                "text": self.reverse_prompt
                            }
                        ]
                    }
                ]
            }
            
            # 根据配置决定使用直接调用还是通过代理服务调用
            if self.use_proxy_service and self.proxy_service_url:
                url = f"{self.proxy_service_url.rstrip('/')}/v1beta/models/{self.model}:generateContent"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"  # 使用Bearer认证方式
                }
                params = {}
            else:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
                headers = {
                    "Content-Type": "application/json",
                }
                params = {
                    "key": self.api_key
                }
            
            # 创建代理配置
            proxies = None
            if self.enable_proxy and self.proxy_url and not self.use_proxy_service:
                proxies = {
                    "http": self.proxy_url,
                    "https": self.proxy_url
                }
            
            # 发送请求
            response = requests.post(
                url,
                headers=headers,
                params=params,
                json=data,
                proxies=proxies,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                candidates = result.get("candidates", [])
                if candidates and len(candidates) > 0:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    
                    # 提取文本响应
                    for part in parts:
                        if "text" in part:
                            return part["text"]
                
                return None
            else:
                logger.error(f"图片分析API调用失败 (状态码: {response.status_code}): {response.text}")
                return None
        except Exception as e:
            logger.error(f"图片分析异常: {str(e)}")
            logger.exception(e)
            return None

    def _analyze_image(self, image_data: bytes, question: Optional[str] = None) -> Optional[str]:
        """分析图片内容或回答关于图片的问题
        
        Args:
            image_data: 图片二进制数据
            question: 可选，用户关于图片的具体问题
            
        Returns:
            str: 分析结果或问题的回答
        """
        try:
            # 将图片数据转换为base64格式
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # 构建请求数据
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ]
            }
            
            # 如果有具体问题，添加到请求中
            if question:
                data["contents"][0]["parts"].append({"text": question})
            else:
                # 使用默认的分析提示词
                default_prompt = "请仔细观察这张图片的内容，然后用简洁清晰的中文回答用户的问题。如用户没有提出额外问题，则简单描述图片中的主体、场景、风格、颜色等关键要素。如果图片包含文字，也请提取出来。"
                data["contents"][0]["parts"].append({"text": default_prompt})
            
            # 根据配置决定使用直接调用还是通过代理服务调用
            if self.use_proxy_service and self.proxy_service_url:
                url = f"{self.proxy_service_url.rstrip('/')}/v1beta/models/{self.model}:generateContent"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"  # 使用Bearer认证方式
                }
                params = {}
            else:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
                headers = {
                    "Content-Type": "application/json",
                }
                params = {
                    "key": self.api_key
                }
            
            # 创建代理配置
            proxies = None
            if self.enable_proxy and self.proxy_url and not self.use_proxy_service:
                proxies = {
                    "http": self.proxy_url,
                    "https": self.proxy_url
                }
            
            # 发送请求
            response = requests.post(
                url,
                headers=headers,
                params=params,
                json=data,
                proxies=proxies,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                candidates = result.get("candidates", [])
                if candidates and len(candidates) > 0:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    
                    # 提取文本响应
                    for part in parts:
                        if "text" in part:
                            return part["text"]
                
                return None
            else:
                logger.error(f"图片分析API调用失败 (状态码: {response.status_code}): {response.text}")
                return None
        except Exception as e:
            logger.error(f"分析图片失败: {str(e)}")
            logger.exception(e)
            return None

    def _handle_reference_image_edit(self, e_context, user_id, prompt, image_base64):
        """
        处理参考图片编辑
        
        Args:
            e_context: 事件上下文
            user_id: 用户ID
            prompt: 编辑提示词
            image_base64: 图片的base64编码
        """
        try:
            # 获取会话标识
            session_id = e_context["context"].get("session_id")
            conversation_key = session_id or user_id
            
            # 注意：提示消息已在调用此方法前发送，此处不再重复发送
            
            # 检查图片数据是否有效
            if not image_base64 or len(image_base64) < 100:
                logger.error(f"无效的图片数据: {image_base64[:20] if image_base64 else 'None'}")
                reply = Reply(ReplyType.TEXT, "无法处理图片，请确保上传的是有效的图片文件。")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
            
            logger.info(f"收到有效的图片数据，长度: {len(image_base64)}")
            
            try:
                # 将base64转换为二进制数据
                image_data = base64.b64decode(image_base64)
                logger.info(f"成功解码图片数据，大小: {len(image_data)} 字节")
                
                # 验证图片数据是否有效
                try:
                    Image.open(BytesIO(image_data))
                    logger.info("图片数据验证成功")
                except Exception as img_err:
                    logger.error(f"图片数据无效: {str(img_err)}")
                    reply = Reply(ReplyType.TEXT, "无法处理图片，请确保上传的是有效的图片文件。")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
            except Exception as decode_err:
                logger.error(f"Base64解码失败: {str(decode_err)}")
                reply = Reply(ReplyType.TEXT, "图片数据解码失败，请重新上传图片。")
                e_context["reply"] = reply
                e_context.action = EventAction.BREAK_PASS
                return
            
            # 获取会话上下文，如果不存在则创建
            if conversation_key not in self.conversations:
                self.conversations[conversation_key] = []
            conversation_history = self.conversations[conversation_key]
            
            # 翻译提示词
            translated_prompt = self._translate_prompt(prompt, user_id)
            logger.info(f"翻译后的提示词: {translated_prompt}")
            
            # 编辑图片
            logger.info("开始调用_edit_image方法")
            result_image, text_response = self._edit_image(translated_prompt, image_data, conversation_history)
            
            if result_image:
                logger.info(f"图片编辑成功，结果大小: {len(result_image)} 字节")
                # 保存编辑后的图片
                reply_text = text_response if text_response else "参考图片编辑成功！"
                if not conversation_history or len(conversation_history) <= 2:  # 如果是新会话
                    reply_text += f"（已开始图像对话，可以继续发送命令修改图片。需要结束时请发送\"{self.exit_commands[0]}\"）"
                
                # 将回复文本添加到文件名中
                clean_text = reply_text.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_")
                clean_text = clean_text[:30] + "..." if len(clean_text) > 30 else clean_text
                
                image_path = os.path.join(self.save_dir, f"gemini_ref_{int(time.time())}_{uuid.uuid4().hex[:8]}_{clean_text}.png")
                with open(image_path, "wb") as f:
                    f.write(result_image)
                
                # 保存最后生成的图片路径
                self.last_images[conversation_key] = image_path
                
                # 添加用户提示到会话
                user_message = {"role": "user", "parts": [{"text": prompt}]}
                conversation_history.append(user_message)
                
                # 添加助手回复到会话
                assistant_message = {
                    "role": "model", 
                    "parts": [
                        {"text": text_response if text_response else "我已编辑了参考图片"},
                        {"image_url": image_path}
                    ]
                }
                conversation_history.append(assistant_message)
                
                # 限制会话历史长度
                if len(conversation_history) > 10:  # 保留最近5轮对话
                    conversation_history = conversation_history[-10:]
                
                # 更新会话时间戳
                self.last_conversation_time[conversation_key] = time.time()
                
                # 准备回复文本
                reply_text = text_response if text_response else "参考图片编辑成功！"
                if not conversation_history or len(conversation_history) <= 2:  # 如果是新会话
                    reply_text += f"（已开始图像对话，可以继续发送命令修改图片。需要结束时请发送\"{self.exit_commands[0]}\"）"
                
                # 先发送文本消息
                e_context["channel"].send(Reply(ReplyType.TEXT, reply_text), e_context["context"])
                
                # 创建文件对象，由框架负责关闭
                image_file = open(image_path, "rb")
                e_context["reply"] = Reply(ReplyType.IMAGE, image_file)
                e_context.action = EventAction.BREAK_PASS
            else:
                logger.error(f"图片编辑失败，API响应: {text_response}")
                # 检查是否有文本响应，可能是内容被拒绝
                if text_response:
                    # 内容审核拒绝的情况，翻译并发送拒绝消息
                    translated_response = self._translate_gemini_message(text_response)
                    reply = Reply(ReplyType.TEXT, translated_response)
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
                else:
                    reply = Reply(ReplyType.TEXT, "参考图片编辑失败，请稍后再试或修改提示词")
                    e_context["reply"] = reply
                    e_context.action = EventAction.BREAK_PASS
        except Exception as e:
            logger.error(f"处理参考图片编辑失败: {str(e)}")
            logger.exception(e)
            reply = Reply(ReplyType.TEXT, f"处理参考图片失败: {str(e)}")
            e_context["reply"] = reply
            e_context.action = EventAction.BREAK_PASS
    
    def get_help_text(self, verbose=False, **kwargs):
        help_text = "基于Google Gemini的图像生成插件\n"
        help_text += "可以生成和编辑图片，支持连续对话\n\n"
        help_text += "使用方法：\n"
        help_text += f"1. 生成图片：发送 {self.commands[0]} + 描述，例如：{self.commands[0]} 一只可爱的猫咪\n"
        help_text += f"2. 编辑图片：发送 {self.edit_commands[0]} + 描述，例如：{self.edit_commands[0]} 给猫咪戴上帽子\n"
        help_text += f"3. 参考图编辑：发送 {self.reference_edit_commands[0]} + 描述，然后上传图片\n"
        help_text += f"4. 融图：发送 {self.merge_commands[0]} + 描述，然后按顺序上传两张图片\n"
        help_text += f"5. 识图：发送 {self.image_analysis_commands[0]} 然后上传图片，或发送问题后上传图片\n"
        help_text += f"6. 反推提示：发送 {self.image_reverse_commands[0]} 然后上传图片，分析图片特征\n"
        help_text += f"7. 追问：发送 {self.follow_up_commands[0]} + 问题，对已识别的图片进行追加提问\n"
        help_text += f"8. 继续对话：直接发送描述，例如：把帽子换成红色的\n"
        help_text += f"9. 结束对话：发送 {self.exit_commands[0]}\n\n"
        
        if self.enable_translate:
            help_text += "特色功能：\n"
            help_text += "* 前置翻译：所有以g开头的指令会自动将中文提示词翻译成英文，然后再调用Gemini API进行图像生成或编辑，提高生成质量\n"
            help_text += f"* 开启翻译：发送 {self.translate_on_commands[0]} 可以开启前置翻译功能\n"
            help_text += f"* 关闭翻译：发送 {self.translate_off_commands[0]} 可以关闭前置翻译功能\n\n"
        
        if verbose:
            help_text += "配置说明：\n"
            help_text += "* 在config.json中可以自定义触发命令和其他设置\n"
            help_text += "* 可以设置代理或代理服务，解决网络访问问题\n"
            
            if self.enable_translate:
                help_text += "* 可以通过enable_translate选项开启或关闭前置翻译功能\n"
                help_text += "* 每个用户可以单独控制是否启用翻译功能\n"
            
            help_text += "\n注意事项：\n"
            help_text += "* 图片生成可能需要一些时间，请耐心等待\n"
            help_text += "* 会话有效期为10分钟，超时后需要重新开始\n"
            help_text += "* 不支持生成违反内容政策的图片\n"
            help_text += "* 识图和追问功能的等待时间为3分钟\n"
            help_text += "* 追问功能仅在最近一次识图后的3分钟内有效\n"
        
        return help_text

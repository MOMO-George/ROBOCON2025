#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用UDP服务器实现
支持多个接收服务器配置和灵活的日志管理
"""

import signal
import sys
import time
import threading
import json
import logging
import socket
from datetime import datetime
from typing import Callable, Dict, Any, Optional, List
from pathlib import Path
import os


class LogManager:
    """日志管理器"""

    DEFAULT_LOG_CONFIG = {
        "level": "INFO",
        "log_dir": "logs",
        "file_enabled": True,
        "file_level": "INFO",
        "file_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "console_enabled": True,
        "console_level": "INFO",
        "console_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self.DEFAULT_LOG_CONFIG
        self.loggers: Dict[str, logging.Logger] = {}
        self._setup_logging()

    def _setup_logging(self):
        """设置日志配置"""
        # 创建日志目录
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        # 配置根日志记录器
        logging.basicConfig(
            level=getattr(logging, self.config.get('level', 'INFO').upper()),
            format=self.config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[]
        )
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取指定名称的日志记录器"""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.handlers.clear()
            
            # 文件处理器
            if self.config.get('file_enabled', True):
                log_file = Path(self.config.get('log_dir', 'logs')) / f"{name}.log"
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(getattr(logging, self.config.get('file_level', 'INFO').upper()))
                file_formatter = logging.Formatter(
                    self.config.get('file_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
            
            # 控制台处理器
            if self.config.get('console_enabled', True):
                console_handler = logging.StreamHandler()
                console_handler.setLevel(getattr(logging, self.config.get('console_level', 'INFO').upper()))
                console_formatter = logging.Formatter(
                    self.config.get('console_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                )
                console_handler.setFormatter(console_formatter)
                logger.addHandler(console_handler)
            
            logger.setLevel(getattr(logging, self.config.get('level', 'INFO').upper()))
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def stop_logging(self):
        """停止所有日志记录器"""
        for logger in self.loggers.values():
            for handler in logger.handlers:
                handler.close()
            logger.handlers.clear()
        self.loggers.clear()


class ConfigManager:
    """配置管理器"""
    
    DEFAULT_CONFIG = {
        "server": {
            "name": "UDPServer",
            "local_host": "0.0.0.0",
            "local_port": 8888,
            "heartbeat_interval": 5.0,
            "data_exchange_interval": 10.0,
            "socket_timeout": 1.0,
            "max_buffer_size": 1024
        },
        "remotes": [
            {
                "name": "RemoteServer1",
                "host": "127.0.0.1",
                "port": 8889,
                "enabled": True
            }
        ],
        "logging": {
            "level": "INFO",
            "log_dir": "logs",
            "file_enabled": True,
            "file_level": "INFO",
            "file_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "console_enabled": True,
            "console_level": "INFO",
            "console_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "features": {
            "heartbeat_enabled": True,
            "data_exchange_enabled": True,
            "auto_reconnect": True,
            "status_display_interval": 30
        }
    }
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self.config_file.exists():
            print(f"配置文件 {self.config_file} 不存在，创建默认配置...")
            self._create_default_config()
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 合并默认配置，确保所有必需的键都存在
            return self._merge_config(self.DEFAULT_CONFIG, config)
        
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            print("使用默认配置...")
            return self.DEFAULT_CONFIG.copy()
    
    def _create_default_config(self):
        """创建默认配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
            print(f"默认配置文件已创建: {self.config_file}")
        except Exception as e:
            print(f"创建默认配置文件失败: {e}")
    
    def _merge_config(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """合并用户配置和默认配置"""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result
    
    def get_server_config(self) -> Dict[str, Any]:
        """获取服务器配置"""
        return self.config.get('server', {})
    
    def get_remote_configs(self) -> List[Dict[str, Any]]:
        """获取远程服务器配置列表"""
        return [remote for remote in self.config.get('remotes', []) if remote.get('enabled', True)]
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.config.get('logging', {})
    
    def get_features_config(self) -> Dict[str, Any]:
        """获取功能配置"""
        return self.config.get('features', {})
    
    def save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置文件失败: {e}")

class UDPServer:
    """通用UDP服务器类（单例）"""

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super(UDPServer, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        log_manager: Optional[LogManager] = None,
    ):
        # 避免重复初始化
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.config_manager = config_manager or ConfigManager()
        self.log_manager = log_manager or LogManager(
            self.config_manager.get_logging_config()
        )

        # 获取配置
        self.server_config = self.config_manager.get_server_config()
        self.remote_configs = self.config_manager.get_remote_configs()
        self.features_config = self.config_manager.get_features_config()

        # 服务器基本信息
        self.server_name = self.server_config["name"]
        self.local_host = self.server_config["local_host"]
        self.local_port = self.server_config["local_port"]
        self.heartbeat_interval = self.server_config["heartbeat_interval"]
        self.data_exchange_interval = self.server_config["data_exchange_interval"]
        self.socket_timeout = self.server_config["socket_timeout"]
        self.max_buffer_size = self.server_config["max_buffer_size"]

        # 获取日志记录器
        self.logger = self.log_manager.get_logger(self.server_name)

        # 创建UDP套接字
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        # 控制标志
        self.running = False
        self.connected_remotes = set()

        # 线程管理
        self.threads = []
        self.thread_lock = threading.Lock()

        # 统计信息
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "heartbeats_sent": 0,
            "heartbeats_received": 0,
            "last_heartbeat_times": {},  # 记录每个远程服务器的最后心跳时间
            "connection_start_time": None,
            "remote_stats": {},  # 每个远程服务器的统计信息
        }

        # 数据处理回调函数
        self.data_handlers: List[Callable[[Dict[str, Any]], None]] = []

        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # 标记初始化完成
        self._initialized = True

    def _signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.info(f"收到信号 {signum}，正在关闭服务器...")
        self.stop()
        sys.exit(0)
    
    def add_data_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """添加数据处理回调函数"""
        self.data_handlers.append(handler)
    
    def remove_data_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """移除数据处理回调函数"""
        if handler in self.data_handlers:
            self.data_handlers.remove(handler)
    
    def start(self):
        """启动服务器"""
        try:
            self.logger.info(f"正在启动服务器 {self.server_name}...")
            
            # 检查网络连接
            self._check_network_connectivity()
            
            # 绑定本地地址
            self.socket.bind((self.local_host, self.local_port))
            self.running = True
            self.stats["connection_start_time"] = datetime.now()
            
            self.logger.info(f"服务器启动成功，监听 {self.local_host}:{self.local_port}")
            self.logger.info(f"配置了 {len(self.remote_configs)} 个远程服务器")
            
            # 启动各种线程
            self._start_threads()
            
            self.logger.info("服务器完全启动，所有线程运行中")
            
        except Exception as e:
            self.logger.error(f"启动服务器失败: {e}")
            self.stop()
            raise
    
    def _check_network_connectivity(self):
        """检查网络连接性"""
        try:
            # 检查本地端口
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            test_socket.bind((self.local_host, self.local_port))
            test_socket.close()
            self.logger.info(f"本地端口检查通过: {self.local_host}:{self.local_port}")
            
            # 检查远程主机解析
            for remote in self.remote_configs:
                try:
                    remote_ip = socket.gethostbyname(remote['host'])
                    self.logger.info(f"远程主机 {remote['name']} 解析成功: {remote['host']} -> {remote_ip}")
                except Exception as e:
                    self.logger.warning(f"远程主机 {remote['name']} 解析失败: {e}")
            
        except Exception as e:
            self.logger.warning(f"网络连接检查警告: {e}")
    
    def _start_threads(self):
        """启动所有线程"""
        with self.thread_lock:
            # 接收线程
            receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            receive_thread.start()
            self.threads.append(receive_thread)
            
            # 心跳线程
            if self.features_config.get('heartbeat_enabled', True):
                heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
                heartbeat_thread.start()
                self.threads.append(heartbeat_thread)
            
            # 数据交换线程
            if self.features_config.get('data_exchange_enabled', True):
                data_exchange_thread = threading.Thread(target=self._data_exchange_loop, daemon=True)
                data_exchange_thread.start()
                self.threads.append(data_exchange_thread)
            
            # 状态显示线程
            status_thread = threading.Thread(target=self._status_loop, daemon=True)
            status_thread.start()
            self.threads.append(status_thread)
    
    def stop(self):
        """停止服务器"""
        if not self.running:
            return
        
        self.logger.info("正在停止服务器...")
        self.running = False
        
        # 关闭套接字
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        # 等待线程结束
        with self.thread_lock:
            for thread in self.threads:
                if thread.is_alive():
                    thread.join(timeout=1)
        
        # 停止日志
        self.log_manager.stop_logging()
        
        self.logger.info("服务器已停止")
    
    def _receive_loop(self):
        """接收数据循环"""
        self.logger.info("接收线程启动")
        
        while self.running:
            try:
                self.socket.settimeout(self.socket_timeout)
                data, addr = self.socket.recvfrom(self.max_buffer_size)
                
                # 解析消息
                try:
                    message = json.loads(data.decode("utf-8"))
                    self._handle_message(message, addr)
                except json.JSONDecodeError:
                    self.logger.warning(f"收到无效JSON数据: {data}")
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    self.logger.error(f"接收数据时出错: {e}")
    
    def _handle_message(self, message: Dict[str, Any], addr: tuple):
        """处理收到的消息"""
        msg_type = message.get("type", "unknown")
        sender = message.get("sender", "Unknown")
        
        if msg_type == "heartbeat":
            self._handle_heartbeat(message, addr)
        elif msg_type == "data":
            self._handle_data(message, addr)
        else:
            self.logger.warning(f"收到未知类型消息: {msg_type} from {sender}")
    
    def _handle_heartbeat(self, message: Dict[str, Any], addr: tuple):
        """处理心跳消息"""
        sender = message.get("sender", "Unknown")
        
        self.stats["heartbeats_received"] += 1
        self.stats["last_heartbeat_times"][sender] = datetime.now()
        self.connected_remotes.add(sender)
        
        self.logger.debug(f"收到来自 {sender} 的心跳 (来源: {addr})")
    
    def _handle_data(self, message: Dict[str, Any], addr: tuple):
        """处理数据消息"""
        sender = message.get("sender", "Unknown")
        data = message.get("data", {})
        timestamp = message.get("timestamp", "")
        
        self.stats["messages_received"] += 1
        
        # 更新远程服务器统计信息
        if sender not in self.stats["remote_stats"]:
            self.stats["remote_stats"][sender] = {"messages_received": 0}
        self.stats["remote_stats"][sender]["messages_received"] += 1
        
        self.logger.info(f"收到来自 {sender} 的数据 (时间: {timestamp})")
        self.logger.debug(f"数据内容: {data}")
        
        # 调用所有数据处理回调函数
        for handler in self.data_handlers:
            try:
                handler(message)
            except Exception as e:
                self.logger.error(f"数据处理回调函数 {handler} 出错: {e}")
    
    def _heartbeat_loop(self):
        """心跳循环"""
        self.logger.info("心跳线程启动")
        
        while self.running:
            try:
                self._send_heartbeat()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                self.logger.error(f"发送心跳时出错: {e}")
    
    def _send_heartbeat(self):
        """发送心跳到所有远程服务器"""
        message = {
            "type": "heartbeat",
            "sender": self.server_name,
            "timestamp": datetime.now().isoformat(),
        }
        
        for remote in self.remote_configs:
            try:
                self._send_message_to_remote(message, remote)
                self.stats["heartbeats_sent"] += 1
                self.logger.debug(f"发送心跳到 {remote['name']} ({remote['host']}:{remote['port']})")
            except Exception as e:
                self.logger.error(f"发送心跳到 {remote['name']} 失败: {e}")
    
    def _data_exchange_loop(self):
        """数据交换循环"""
        self.logger.info("数据交换线程启动")
        
        while self.running:
            try:
                self._send_data()
                time.sleep(self.data_exchange_interval)
            except Exception as e:
                self.logger.error(f"发送数据时出错: {e}")
    
    def _send_data(self):
        """发送数据到所有远程服务器"""
        # 生成服务器状态数据
        data = {
            "server_status": "running",
            "timestamp": datetime.now().isoformat(),
            "uptime": (
                datetime.now() - self.stats["connection_start_time"]
            ).total_seconds() if self.stats["connection_start_time"] else 0,
            "stats": self.get_stats(),
            "connected_remotes": list(self.connected_remotes),
        }
        
        message = {
            "type": "data",
            "sender": self.server_name,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        
        for remote in self.remote_configs:
            try:
                self._send_message_to_remote(message, remote)
                self.stats["messages_sent"] += 1
                self.logger.debug(f"发送数据到 {remote['name']}")
            except Exception as e:
                self.logger.error(f"发送数据到 {remote['name']} 失败: {e}")
    
    def _send_message_to_remote(self, message: Dict[str, Any], remote: Dict[str, Any]):
        """发送消息到指定远程服务器"""
        try:
            def default_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")
            
            data = json.dumps(
                message, ensure_ascii=False, default=default_serializer
            ).encode("utf-8")
            
            self.socket.sendto(data, (remote['host'], remote['port']))
            
        except Exception as e:
            self.logger.error(f"发送消息到 {remote['name']} 失败: {e}")
            raise
    
    def send_custom_data(self, data: Dict[str, Any], target_remotes: Optional[List[str]] = None):
        """
        发送自定义数据
        
        Args:
            data: 要发送的数据
            target_remotes: 目标远程服务器名称列表，None表示发送到所有服务器
        """
        message = {
            "type": "data",
            "sender": self.server_name,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        
        targets = self.remote_configs
        if target_remotes:
            targets = [r for r in self.remote_configs if r['name'] in target_remotes]
        
        for remote in targets:
            try:
                self._send_message_to_remote(message, remote)
                self.stats["messages_sent"] += 1
                self.logger.info(f"发送自定义数据到 {remote['name']}: {data}")
            except Exception as e:
                self.logger.error(f"发送自定义数据到 {remote['name']} 失败: {e}")
    
    def _status_loop(self):
        """状态显示循环"""
        self.logger.info("状态显示线程启动")
        interval = self.features_config.get('status_display_interval', 30)
        
        while self.running:
            try:
                time.sleep(interval)
                if self.running:
                    self._print_status()
            except Exception as e:
                self.logger.error(f"状态显示循环出错: {e}")
    
    def _print_status(self):
        """打印服务器状态"""
        stats = self.get_stats()
        
        status_lines = [
            f"\n=== {self.server_name} 状态 ===",
            f"运行状态: {'运行中' if stats['running'] else '已停止'}",
            f"本地地址: {self.local_host}:{self.local_port}",
            f"已连接远程服务器: {len(self.connected_remotes)}/{len(self.remote_configs)}",
            f"连接的服务器: {', '.join(self.connected_remotes) if self.connected_remotes else '无'}",
            f"发送消息数: {stats['messages_sent']}",
            f"接收消息数: {stats['messages_received']}",
            f"发送心跳数: {stats['heartbeats_sent']}",
            f"接收心跳数: {stats['heartbeats_received']}",
        ]
        
        if stats['connection_start_time']:
            uptime = (datetime.now() - stats['connection_start_time']).total_seconds()
            status_lines.append(f"运行时间: {uptime/60:.1f} 分钟")
        
        # 显示每个远程服务器的状态
        for remote_name, last_heartbeat in stats['last_heartbeat_times'].items():
            if last_heartbeat:
                elapsed = (datetime.now() - last_heartbeat).total_seconds()
                status = "在线" if elapsed < self.heartbeat_interval * 3 else "离线"
                status_lines.append(f"  {remote_name}: {status} (最后心跳: {elapsed:.1f}s前)")
        
        self.logger.info('\n'.join(status_lines))
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        stats['running'] = self.running
        stats['connected_remotes'] = list(self.connected_remotes)
        stats['remote_count'] = len(self.remote_configs)
        stats['connected_count'] = len(self.connected_remotes)
        return stats
    
    def is_connected_to(self, remote_name: str) -> bool:
        """检查是否连接到指定远程服务器"""
        if remote_name not in self.stats['last_heartbeat_times']:
            return False
        
        last_heartbeat = self.stats['last_heartbeat_times'][remote_name]
        if not last_heartbeat:
            return False
        
        # 如果超过3个心跳间隔没有收到心跳，认为连接断开
        timeout = self.heartbeat_interval * 3
        elapsed = (datetime.now() - last_heartbeat).total_seconds()
        return elapsed < timeout



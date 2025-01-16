# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os.path

from omegaconf import OmegaConf, DictConfig
from pathlib import Path, PureWindowsPath, PurePosixPath


def is_path_string(value: str) -> bool:
    """
    주어진 문자열이 파일 경로인지 확인합니다.

    경로 판단 기준:
    1. Windows 또는 POSIX 절대 경로 패턴 매칭
    2. 상대 경로 시작 패턴 (./, ../) 확인
    3. 경로 구성요소의 유효성
    """
    if not isinstance(value, str) or not value.strip():
        return False

    # 기본적인 구조 검사
    has_path_separators = '/' in value or '\\' in value
    if not has_path_separators:
        return False

    try:
        # 절대 경로 패턴 확인
        # Windows: C:\, D:\, etc.
        is_windows_abs = len(value) > 2 and value[1] == ':' and value[2] in ('\\', '/')
        # POSIX: /
        is_posix_abs = value.startswith('/')

        # 상대 경로 패턴 확인 (./, ../)
        is_relative = value.startswith('./') or value.startswith('../') or value.startswith('.\\') or value.startswith('..\\')

        # 명확한 경로 패턴인 경우만 처리
        if not (is_windows_abs or is_posix_abs or is_relative):
            return False

        # 경로 유효성 검증
        try:
            if is_windows_abs:
                PureWindowsPath(value)
            else:
                PurePosixPath(value)
        except Exception:
            return False

        # 경로 구성요소 유효성 검사
        path_parts = value.replace('\\', '/').split('/')
        for part in path_parts:
            if not part:
                continue
            # 경로에 허용되지 않는 특수문자 검사
            if any(char in part for char in '<>:"|?*\0'):
                return False

        return True

    except Exception:
        return False


def convert_path_string(path_str: str) -> str:
    """
    경로 문자열을 현재 OS에 맞는 형식으로 변환합니다.
    """
    try:
        # 경로 구분자 정규화
        normalized_path = path_str.replace('\\', '/')

        # Windows에서 Linux 경로를 처리할 때
        if os.name == 'nt' and path_str.startswith('/'):
            # 루트 경로를 현재 드라이브로 변경
            current_drive = os.getcwd()[:2]
            normalized_path = f"{current_drive}{normalized_path}"

        # Linux에서 Windows 경로를 처리할 때
        elif os.name != 'nt' and len(path_str) > 1 and path_str[1] == ':':
            # 드라이브 문자를 제거하고 절대 경로로 변환
            normalized_path = '/' + normalized_path[3:]

        return str(Path(normalized_path))
    except Exception:
        return path_str

class Config:
    def __init__(self, args):
        self.config = {}

        self.args = args
        user_config = self._build_opt_list(self.args.options)
        config = OmegaConf.load(self.args.cfg_path)
        config = OmegaConf.merge(config, user_config)
        config = self.chage_path_for_os(config)
        self.config = config
        self.os = os.name

    def chage_path_for_os(self, config):
        # Check each key in the config
        for key in config.keys():
            # If the key is a string, check if it is a path

            if isinstance(config[key], str) and is_path_string(config[key]):
                config[key] = convert_path_string(config[key])

            # If the key is a dictionary, recursively check the dictionary
            elif isinstance(config[key], DictConfig):
                config[key] = self.chage_path_for_os(config[key])

        return config

    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []

        if len(opts) == 0:
            return opts

        has_equal = opts[0].find("=") != -1

        if has_equal:
            return opts

        return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]

    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)

    def pretty_print(self):
        logging.info("\n=====  Running Parameters    =====")
        logging.info(self._convert_node_to_json(self.config.run))

        logging.info("\n======  Dataset Attributes  ======")
        logging.info(self._convert_node_to_json(self.config.datasets))

        logging.info("\n======  Model Attributes  ======")
        logging.info(self._convert_node_to_json(self.config.model))

    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)

    def to_dict(self):
        return OmegaConf.to_container(self.config)

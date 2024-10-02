from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests
import configparser
import json

class AimemberLLM(LLM):
    __setattr__ = object.__setattr__

    def __init__(self, config_path: str = 'config.ini', **kwargs):
        #TODO: 파라미터로 endpoint를 입력받아 apiurl, payload 설정
        super().__init__(**kwargs)
        config = configparser.ConfigParser()
        config.read(config_path)

        self.api_baseurl = config['AIMEMBER']['API_BASEURL']
        self.auth_token = config['AIMEMBER']['AUTH_TOKEN']

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # 별도 G/W앱으로 중계하는 시나리오
        headers = {
            #"Authorization": f"Bearer {self.api_key}",
            "X-Auth-Token": self.auth_token,
            "Content-Type": "application/json"
        }

        # lottegpt 호출하는 시나리오
        payload = {
            "query": prompt,
            "history": ''
        }

        response = requests.post(
            self.api_baseurl, 
            json=payload, 
            headers=headers
        )

        response.raise_for_status()
        data = json.loads(response.json())
        text = data.get('message','Internal Server Error ' + str(data.get('status_code','900')))

        # 'stop' 토큰 처리
        if stop:
            for stop_token in stop:
                text = text.split(stop_token)[0]

        return text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """LLM을 식별하는 데 필요한 파라미터를 반환합니다."""
        return {
            "api_baseurl": self.api_baseurl,
            "auth_token": self.auth_token[:5] + '*' * (len(self.auth_token) - 5)
        }

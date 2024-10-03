from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests
import configparser
import json

class AimemberLLM(LLM):
    __setattr__ = object.__setattr__

    def __init__(self, endpoint: str, config_path: str = 'config.ini', **kwargs):
        super().__init__(**kwargs)
        config = configparser.ConfigParser()
        config.read(config_path)

        endpoint = endpoint.strip()
        if not endpoint:
            raise ValueError("Endpoint cannot be null or empty")
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint

        self.api_baseurl = config['AIMEMBER']['API_BASEURL']
        self.auth_token = config['AIMEMBER']['AUTH_TOKEN']
        self.endpoint = endpoint

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "X-Auth-Token": self.auth_token,
            "Content-Type": "application/json"
        }

        payload = self._create_payload(prompt)

        response = requests.post(
            self.api_baseurl + self.endpoint, 
            json=payload, 
            headers=headers
        )

        response.raise_for_status()
        data = json.loads(response.json())
        text = data.get('message', 'Internal Server Error ' + str(data.get('status_code','900')))

        if stop:
            for stop_token in stop:
                text = text.split(stop_token)[0]

        return text

    def _create_payload(self, prompt: str) -> dict:
        """
        prompt: str

        Create payload from the prompt JSON.
        """

        payload = json.loads(prompt)
        if self.endpoint in ["/lottegpt", "/chatgpt"]:
            return {
                "query": payload.get("query", ""),
                "history": payload.get("history", "")
            }
        elif self.endpoint == "/lottegpt/search":
            return {
                "query": payload.get("query", ""),
                "search_resource": payload.get("search_resource", {})
            }
        elif self.endpoint == "/summarization":
            return {
                "document": payload.get("document", "")
            }
        elif self.endpoint == "/recommendation":
            return {
                "question": payload.get("question", ""),
                "answer": payload.get("answer", "")
            }
        elif self.endpoint in ["/gemini/nostream", "/gemini/claude", "/codegenerate", "/wordner", "/worddetector"]:
            return {
                "query": payload.get("query", "")
            }
        elif self.endpoint == "/translate":
            return {
                "doc": payload.get("doc", ""),
                "src_lang": payload.get("src_lang", ""),
                "tgt_lang": payload.get("tgt_lang", "")
            }
        else:
            return {
                "query": payload.get("query", ""),
                "history": payload.get("history", "")
            }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "api_baseurl": self.api_baseurl,
            "auth_token": self.auth_token[:5] + '*' * (len(self.auth_token) - 5),
            "endpoint": self.endpoint
        }

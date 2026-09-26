from submission_paths import path as _submission_path
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from code_strategy3.config import Strategy3Config
from code_strategy3.config import load_strategy3_runtime_config
from code_strategy3.llm_client import build_provider
from code_strategy3.profiles_v1 import STRATEGY3_PROFILES_V1
from code_strategy3.providers import QwenProvider
WORKSPACE_DIR = _submission_path('project', '')
RESULTS_ROOT = WORKSPACE_DIR / 'results' / 'strategy3'

class Strategy3VersionedProfilesTest(unittest.TestCase):

    def test_versioned_profile_source_contains_no_api_key_field(self):
        for profile in STRATEGY3_PROFILES_V1.values():
            self.assertNotIn('api_key', profile)

    def test_existing_qwen_profile_resolution_is_unchanged(self):
        config = Strategy3Config.from_profile('qwen')
        self.assertEqual(config.provider, 'qwen')
        self.assertEqual(config.model, 'qwen3.5-plus')
        self.assertEqual(config.api_base, 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1')
        self.assertEqual(config.results_dir, RESULTS_ROOT / 'qwen')

    def test_kimi_profile_uses_qwen_adapter_and_isolated_paths(self):
        config = Strategy3Config.from_profile('kimi-k2.6')
        self.assertEqual(config.provider, 'qwen')
        self.assertEqual(config.model, 'kimi/kimi-k2.6')
        self.assertEqual(config.api_base, 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.assertEqual(config.results_dir, RESULTS_ROOT / 'kimi_k26')
        self.assertEqual(config.cache_dir, RESULTS_ROOT / 'kimi_k26' / 'cache')

    def test_minimax_alias_resolves_exact_remote_model(self):
        config = Strategy3Config.from_profile('minimax-m27')
        self.assertEqual(config.provider, 'qwen')
        self.assertEqual(config.model, 'MiniMax/MiniMax-M2.7')
        self.assertEqual(config.results_dir, RESULTS_ROOT / 'minimax_m27')
        self.assertEqual(config.cache_dir, RESULTS_ROOT / 'minimax_m27' / 'cache')

    def test_environment_profile_uses_dashscope_key_without_persisting_it(self):
        env = {'STRATEGY3_PROFILE': 'kimi', 'DASHSCOPE_API_KEY': 'offline-test-key'}
        with patch.dict(os.environ, env, clear=True):
            config = Strategy3Config.from_env()
        self.assertEqual(config.provider, 'qwen')
        self.assertEqual(config.model, 'kimi/kimi-k2.6')
        self.assertEqual(config.api_key, 'offline-test-key')

    def test_profile_plus_use_env_honors_profile_and_environment_key(self):
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'offline-test-key'}, clear=True):
            config = load_strategy3_runtime_config(use_env=True, profile_name='minimax-m2.7')
        self.assertEqual(config.provider, 'qwen')
        self.assertEqual(config.model, 'MiniMax/MiniMax-M2.7')
        self.assertEqual(config.api_key, 'offline-test-key')
        self.assertNotIn('STRATEGY3_PROFILE', os.environ)

    @patch('code_strategy3.providers.qwen.requests.post')
    def test_building_provider_is_offline(self, requests_post):
        with patch.dict(os.environ, {'STRATEGY3_PROFILE': 'kimi-k2.6', 'DASHSCOPE_API_KEY': 'offline-test-key'}, clear=True):
            config = Strategy3Config.from_env()
            provider = build_provider(config)
        self.assertIsInstance(provider, QwenProvider)
        requests_post.assert_not_called()
if __name__ == '__main__':
    unittest.main()

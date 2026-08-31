# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Tests for llm_router config dataclasses and parsing.

Sections:
  • KVCAwareStrategyConfig — normal construction, validation, Hydra instantiation
  • KVCAwareConfig.from_config — normal inputs, abnormal inputs, other behaviour
  • E2E — packaged YAML composition
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from uni_agent.llm_router.config import (
    CacheStoreConfig,
    CollectorConfig,
    ConfigError,
    KVCAwareConfig,
    KVCAwareStrategyConfig,
    StrategyConfig,
)
from uni_agent.llm_router.types import OverloadMode, SlowCut

pytestmark = [pytest.mark.level0, pytest.mark.cpu]


# ============================================================
# KVCAwareStrategyConfig
# ============================================================

# -- (1) Normal construction --


def test_strategy_default_config():
    """
    Feature: slow_cut accepts a YAML string and coerces to the SlowCut enum
    Description: construct strategy config with slow_cut="CAPACITY_TOKEN_AWARE"
    Expectation: cfg.slow_cut == SlowCut.CAPACITY_TOKEN_AWARE
    """
    cfg = KVCAwareStrategyConfig()
    assert cfg.slow_cut == SlowCut.CAPACITY_TOKEN_AWARE
    assert cfg.overload_mode == OverloadMode.KV_CACHE_USAGE_PERC
    assert cfg.do_shortcut is True


def test_strategy_normal_fields_parse():
    """
    Feature: strategy fields assign explicit values and fall back to defaults
    Description: construct KVCAwareStrategyConfig with varied kwargs
    Expectation: each field equals the expected explicit or default value
    """
    cfg = KVCAwareStrategyConfig(load_threshold=0.5)
    assert cfg.load_threshold == 0.5


# -- (2) Abnormal construction --


@pytest.mark.parametrize("load_threshold", [0, -1, 1.0, 80])
def test_strategy_load_threshold_out_of_range_raises_config_error(load_threshold):
    """
    Feature: load_threshold outside (0, 1) triggers validation error
    Description: construct strategy config with load_threshold = 0 / -1 / 1.0 / 80
    Expectation: raises ConfigError matching "load_threshold"
    """
    with pytest.raises(ConfigError, match="load_threshold"):
        KVCAwareStrategyConfig(load_threshold=load_threshold)


# -- (3) Hydra instantiation --


def test_strategy_instantiates_with_explicit_fields_and_defaults():
    """
    Feature: Hydra instantiate produces a fully-populated KVCAwareStrategyConfig
    Description: instantiate a strategy entry with only _target_
    Expectation: result is a KVCAwareStrategyConfig/StrategyConfig, defaults filled
    """
    from hydra.utils import instantiate

    entry = OmegaConf.create(
        {
            "_target_": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
        }
    )
    result = instantiate(entry)

    # concrete type and StrategyConfig base
    assert isinstance(result, KVCAwareStrategyConfig)
    assert isinstance(result, StrategyConfig)
    # defaults filled in
    assert result.do_shortcut is True
    assert result.slow_cut == "capacity-token-aware"
    assert result.overload_mode == "kv_cache_usage_perc"
    assert result.load_threshold == 0.9


# ============================================================
# KVCAwareConfig.from_config — normal input
# ============================================================


@pytest.mark.parametrize("wrap", [lambda d: OmegaConf.create(d), lambda d: d], ids=["omegaconf", "plain_dict"])
def test_from_config_assembles_three_sections(wrap):
    """
    Feature: from_config assembles the three sections into a KVCAwareConfig
    Description: from_config with a full strategies/collector/cache_store config (OmegaConf or plain dict)
    Expectation: result is a KVCAwareConfig with correctly typed sections
    """
    full_config = {
        "strategies": [
            {
                "_target_": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
                "load_threshold": 0.7,
            },
        ]
    }
    result = KVCAwareConfig.from_config(full_config)
    assert isinstance(result, KVCAwareConfig)
    assert isinstance(result.strategies[0], KVCAwareStrategyConfig)
    assert isinstance(result.collector, CollectorConfig)
    assert isinstance(result.cache_store, CacheStoreConfig)


def test_from_config_strategies_dict_converts_to_list():
    """
    Feature: strategies given as a mapping auto-converts to a list
    Description: from_config with strategies in dict form
    Expectation: result.strategies is a list of KVCAwareStrategyConfig with correct values
    """
    kwargs = OmegaConf.create(
        {
            "strategies": {
                "kvc_aware": {
                    "_target_": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
                },
            },
        }
    )
    result = KVCAwareConfig.from_config(kwargs)
    assert isinstance(result.strategies, list)
    assert len(result.strategies) == 1
    assert isinstance(result.strategies[0], KVCAwareStrategyConfig)


def test_from_config_omitted_sections_take_defaults():
    """
    Feature: omitted collector/cache_store take their Config defaults
    Description: from_config with a config containing only strategies
    Expectation: result.collector/cache_store are CollectorConfig/CacheStoreConfig instances
    """
    kwargs = OmegaConf.create(
        {
            "strategies": [
                {
                    "_target_": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
                },
            ],
        }
    )
    result = KVCAwareConfig.from_config(kwargs)
    assert isinstance(result.collector, CollectorConfig)
    assert isinstance(result.cache_store, CacheStoreConfig)


# ============================================================
# KVCAwareConfig.from_config — abnormal input
# ============================================================


@pytest.mark.parametrize(
    "strategies",
    [
        [],  # empty list
        "kvc_aware",  # not a list
        ["kvc_aware"],  # item not dict
    ],
)
def test_strategies_invalid_raises_config_error(strategies):
    """
    Feature: invalid strategies list triggers from_config validation error
    Description: from_config with empty list / non-list / item not dict
    Expectation: raises ConfigError
    """
    kwargs = OmegaConf.create({"strategies": strategies})
    with pytest.raises(ConfigError):
        KVCAwareConfig.from_config(kwargs)


@pytest.mark.parametrize(
    "strategies,match",
    [
        # missing _target_ on a strategy item (list form)
        ({"strategies": [{}]}, "_target_"),
        # missing _target_ on a strategy item (dict form)
        ({"strategies": {"kvc_aware": {}}}, "_target_"),
        # _target_ pointing to a non-StrategyConfig subclass
        (
            {
                "strategies": [
                    {
                        "_target_": "uni_agent.llm_router.config.CacheStoreConfig",
                        "kv_cache_store_type": "list",
                        "ttl": 30,
                    },
                ]
            },
            "StrategyConfig",
        ),
        # empty config — no strategies key at all
        ({}, "strategies"),
        # strategies explicitly null
        ({"strategies": None}, "strategies"),
    ],
)
def test_from_config_rejects_invalid_strategies(strategies, match):
    """
    Feature: from_config rejects invalid strategies configuration
    Description: missing _target_ / non-StrategyConfig _target_ / empty config / null strategies
    Expectation: raises ConfigError matching the relevant keyword
    """
    kwargs = OmegaConf.create(strategies)
    with pytest.raises(ConfigError, match=match):
        KVCAwareConfig.from_config(kwargs)


@pytest.mark.parametrize("bad_cfg", ["invalid", None, 42])
def test_from_config_rejects_non_dict_input(bad_cfg):
    """
    Feature: from_config rejects inputs that are neither DictConfig nor dict
    Description: from_config with str / None / int input
    Expectation: raises ConfigError matching "cfg"
    """
    with pytest.raises(ConfigError, match="cfg"):
        KVCAwareConfig.from_config(bad_cfg)


def test_from_config_aggregates_multiple_errors():
    """
    Feature: from_config raises on multiple aggregated errors
    Description: from_config with empty strategies and invalid collector/cache_store
    Expectation: raises ConfigError whose message contains a relevant field name
    """
    kwargs = OmegaConf.create(
        {
            "strategies": [],
            "collector": {"http_polling": {"polling_interval": -1}},
            "cache_store": {"ttl": 0},
        }
    )
    with pytest.raises(ConfigError) as exc_info:
        KVCAwareConfig.from_config(kwargs)
    error_msg = str(exc_info.value)
    assert "strategies" in error_msg or "polling_interval" in error_msg or "ttl" in error_msg


# ============================================================
# KVCAwareConfig other behaviour
# ============================================================


def test_top_level_unknown_keys_ignored():
    """
    Feature: from_config ignores top-level keys outside the config domain
    Description: from_config with a config carrying extra non-domain keys
    Expectation: result is KVCAwareConfig and the extra keys are dropped
    """
    kwargs = OmegaConf.create(
        {
            "_unknown_top_level": "ignored",
            "router_strategy": "should_be_dropped",
            "strategies": [
                {
                    "_target_": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
                },
            ],
        }
    )
    result = KVCAwareConfig.from_config(kwargs)
    assert isinstance(result, KVCAwareConfig)
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(result)}
    assert "_unknown_top_level" not in field_names
    assert "router_strategy" not in field_names


def test_compact_repr():
    """
    Feature: KVCAwareConfig renders a multi-line indented repr
    Description: from_config then call repr and inspect output
    Expectation: repr contains newline, starts with KVCAwareConfig(, and includes fields
    """
    kwargs = OmegaConf.create(
        {
            "strategies": [
                {
                    "_target_": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
                },
            ],
            "collector": {
                "http_polling": {"polling_interval": 5, "http_timeout": 10},
                "long_connection": {
                    "base_retry_delay": 1.0,
                    "max_retry_delay": 30.0,
                    "max_retry_attempts": 5,
                    "retry_backoff_factor": 2.0,
                },
            },
            "cache_store": {"kv_cache_store_type": "list", "ttl": 30},
        }
    )
    result = KVCAwareConfig.from_config(kwargs)
    r = repr(result)
    assert "\n" in r
    assert r.startswith("KVCAwareConfig(")


# ============================================================
# E2E integration
# ============================================================


def _load_pkg_router_yaml() -> dict:
    """Compose ``uni_agent/llm_router/configs/router.yaml`` via Hydra.

    Expands the YAML defaults block into one config dictionary.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    import uni_agent.llm_router.configs as _cfg_pkg

    config_dir = str(next(iter(_cfg_pkg.__path__)))
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="router")
    return OmegaConf.to_container(cfg, resolve=True)


def test_packaged_yaml_e2e():
    """
    Feature: Hydra composition of the packaged router YAML parses end-to-end
    Description: compose configs/router.yaml, then from_config
    Expectation: all fields match the YAML config and non-domain keys are dropped
    """
    loaded = _load_pkg_router_yaml()

    result = KVCAwareConfig.from_config(loaded)

    # ── strategies ──
    assert isinstance(result.strategies, list)
    assert len(result.strategies) == 1
    strategy = result.strategies[0]
    assert isinstance(strategy, KVCAwareStrategyConfig)
    assert strategy.alpha == 0.7
    assert strategy.load_threshold == 0.9
    assert strategy.layer_weights == {"gpu": 0.7, "cpu": 0.2, "ssd": 0.1}
    # FQN top-level keys are harmless to from_config (extra keys are dropped)
    assert loaded["router_class"] == "uni_agent.llm_router.balancer.KVCAwareBalancer"
    assert not hasattr(result, "router_class")

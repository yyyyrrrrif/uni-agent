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

Per §5 of detailed_config.md, each module has 5 test categories:
① Input/output normal cases
② Input/output abnormal cases
③ Hydra parsing normal cases
④ Hydra parsing abnormal cases
⑤ Other cases
"""

from __future__ import annotations

import pytest
from hydra.errors import InstantiationException
from omegaconf import OmegaConf

from uni_agent.llm_router.config import (
    CacheStoreConfig,
    CollectorConfig,
    ConfigError,
    KVCAwareConfig,
    KVCAwareStrategyConfig,
    StrategyConfig,
)
from uni_agent.llm_router.types import SlowCut

# Default collector_names for strategy construction (required field)
_CN = ["vllm_zmq"]


# ============================================================
# 5.1 StrategyConfig / KVCAwareStrategyConfig
# ============================================================

# -- ① Input/output normal cases --


pytestmark = [pytest.mark.ut, pytest.mark.cpu]


class TestStrategyNormalInput:
    """Normal construction: field assignment, defaults, collector binding."""

    @pytest.mark.parametrize(
        "kwargs,attr,expected",
        [
            # weight explicit value
            ({"weight": 1.0, "collector_names": _CN}, "weight", 1.0),
            # alpha default when omitted
            ({"weight": 1.0, "collector_names": _CN}, "alpha", 0.7),
            # load_threshold explicit value
            ({"weight": 1.0, "load_threshold": 0.5, "collector_names": _CN}, "load_threshold", 0.5),
            # layer_weights three-tier dict
            (
                {"weight": 1.0, "layer_weights": {"gpu": 0.7, "cpu": 0.2, "ssd": 0.1}, "collector_names": _CN},
                "layer_weights",
                {"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
            ),
            # memory_overload_filter explicit False
            ({"weight": 1.0, "memory_overload_filter": False, "collector_names": _CN}, "memory_overload_filter", False),
        ],
    )
    def test_normal_fields_parse(self, kwargs, attr, expected):
        """
        Feature: strategy fields assign explicit values and fall back to defaults
        Description: construct KVCAwareStrategyConfig with varied kwargs
        Expectation: each field equals the expected explicit or default value
        """
        cfg = KVCAwareStrategyConfig(**kwargs)
        assert getattr(cfg, attr) == expected

    def test_slow_cut_string_coerced(self):
        """
        Feature: slow_cut accepts a YAML string and coerces to the SlowCut enum
        Description: construct strategy config with slow_cut="least-inflight"
        Expectation: cfg.slow_cut == SlowCut.LEAST_INFLIGHT
        """
        cfg = KVCAwareStrategyConfig(weight=1.0, slow_cut="least-inflight", collector_names=_CN)
        assert cfg.slow_cut == SlowCut.LEAST_INFLIGHT

    def test_collector_names_binds_multiple(self):
        """
        Feature: collector_names binds multiple collector identifiers in order
        Description: construct strategy config with two collector names
        Expectation: cfg.collector_names equals the input list
        """
        cfg = KVCAwareStrategyConfig(weight=1.0, collector_names=["vllm_zmq", "mooncake_prometheus"])
        assert cfg.collector_names == ["vllm_zmq", "mooncake_prometheus"]

    def test_multi_strategy_weights_sum_to_one(self):
        """
        Feature: weights across multiple strategies sum to ~1.0
        Description: construct two strategies with weights 0.6 and 0.4
        Expectation: sum of weights == pytest.approx(1.0)
        """
        s1 = KVCAwareStrategyConfig(weight=0.6, collector_names=["vllm_zmq"])
        s2 = KVCAwareStrategyConfig(weight=0.4, collector_names=["mooncake_prometheus"])
        assert s1.weight + s2.weight == pytest.approx(1.0)


# -- ② Input/output abnormal cases --


class TestStrategyAbnormalInput:
    """Abnormal construction: per-field validation errors."""

    @pytest.mark.parametrize("weight", [0.0, 1.5, -1.0])
    def test_weight_out_of_range_raises_config_error(self, weight):
        """
        Feature: weight outside (0, 1) triggers validation error
        Description: construct strategy config with weight = 0.0 / 1.5 / -1.0
        Expectation: raises ConfigError matching "weight"
        """
        with pytest.raises(ConfigError, match="weight"):
            KVCAwareStrategyConfig(weight=weight, collector_names=_CN)

    @pytest.mark.parametrize("weight", ["0.7", None])
    def test_weight_wrong_type_or_missing_raises_type_error(self, weight):
        """
        Feature: weight of wrong type or missing triggers TypeError
        Description: construct strategy config with weight="0.7" or without weight
        Expectation: raises TypeError
        """
        if weight is None:
            with pytest.raises(TypeError):
                KVCAwareStrategyConfig(collector_names=_CN)
        else:
            with pytest.raises(TypeError):
                KVCAwareStrategyConfig(weight=weight, collector_names=_CN)

    @pytest.mark.parametrize("load_threshold", [0, -1, 1.0, 80])
    def test_load_threshold_out_of_range_raises_config_error(self, load_threshold):
        """
        Feature: load_threshold outside (0, 1) triggers validation error
        Description: construct strategy config with load_threshold = 0 / -1 / 1.0 / 80
        Expectation: raises ConfigError matching "load_threshold"
        """
        with pytest.raises(ConfigError, match="load_threshold"):
            KVCAwareStrategyConfig(weight=1.0, load_threshold=load_threshold, collector_names=_CN)

    @pytest.mark.parametrize(
        "layer_weights",
        [
            {"gpu": 0.7, "cpu": 0.2, "disk": 0.1},  # illegal key
            {"gpu": 0.7, "cpu": 0.3},  # missing tier
            {"gpu": 0.5, "cpu": 0.2, "ssd": 0.1},  # sum < 1
            {"gpu": 0.7, "cpu": 0.2, "ssd": 0.2},  # sum > 1
        ],
    )
    def test_layer_weights_invalid_raises_config_error(self, layer_weights):
        """
        Feature: invalid layer_weights triggers validation error
        Description: construct strategy config with illegal key / missing tier / sum below or above 1.0
        Expectation: raises ConfigError matching "layer_weights"
        """
        with pytest.raises(ConfigError, match="layer_weights"):
            KVCAwareStrategyConfig(weight=1.0, layer_weights=layer_weights, collector_names=_CN)

    def test_collector_names_not_list_raises_config_error(self):
        """
        Feature: non-list collector_names triggers validation error
        Description: construct strategy config with collector_names="vllm_zmq"
        Expectation: raises ConfigError matching "collector_names must be a list"
        """
        with pytest.raises(ConfigError, match="collector_names must be a list"):
            KVCAwareStrategyConfig(weight=1.0, collector_names="vllm_zmq")

    def test_collector_names_missing_raises_type_error(self):
        """
        Feature: missing required collector_names triggers TypeError
        Description: construct strategy config without collector_names
        Expectation: raises TypeError
        """
        with pytest.raises(TypeError):
            KVCAwareStrategyConfig(weight=1.0)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("memory_overload_filter", "yes"),
            ("slow_cut", "random"),
        ],
    )
    def test_optional_field_wrong_type_or_value_raises_config_error(self, field, value):
        """
        Feature: optional field of wrong type or value triggers validation error
        Description: construct strategy config with memory_overload_filter="yes" or slow_cut="random"
        Expectation: raises ConfigError matching the field name
        """
        with pytest.raises(ConfigError, match=field):
            KVCAwareStrategyConfig(weight=1.0, **{field: value}, collector_names=_CN)

    @pytest.mark.parametrize(
        "strategies",
        [
            # weights not summing to 1
            [
                {
                    "_target": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
                    "weight": 0.4,
                    "collector_names": ["vllm_zmq"],
                },
                {
                    "_target": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
                    "weight": 0.4,
                    "collector_names": ["mooncake_prometheus"],
                },
            ],
            [],  # empty list
            "kvc_aware",  # not a list
            ["kvc_aware"],  # item not dict
        ],
    )
    def test_strategies_invalid_raises_config_error(self, strategies):
        """
        Feature: invalid strategies list triggers from_config validation error
        Description: from_config with weights not summing to 1 / empty list / non-list / item not dict
        Expectation: raises ConfigError
        """
        kwargs = OmegaConf.create({"strategies": strategies})
        with pytest.raises(ConfigError):
            KVCAwareConfig.from_config(kwargs)


# -- ③ Hydra parsing normal cases --


class TestStrategyHydraNormal:
    """Hydra instantiate produces a fully-populated KVCAwareStrategyConfig."""

    def test_strategy_instantiates_with_explicit_fields_and_defaults(self):
        """
        Feature: Hydra instantiate produces a fully-populated KVCAwareStrategyConfig
        Description: instantiate a strategy entry with weight and collector_names
        Expectation: result is a KVCAwareStrategyConfig/StrategyConfig, explicit fields preserved, defaults filled
        """
        from hydra.utils import instantiate

        entry = OmegaConf.create(
            {
                "_target_": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
                "weight": 1.0,
                "collector_names": ["vllm_zmq"],
            }
        )
        result = instantiate(entry)

        # concrete type and StrategyConfig base
        assert isinstance(result, KVCAwareStrategyConfig)
        assert isinstance(result, StrategyConfig)
        # explicit field preserved
        assert result.weight == 1.0
        assert result.collector_names == ["vllm_zmq"]
        # defaults filled in
        assert result.alpha == 0.7
        assert result.load_threshold == 0.9
        assert result.layer_weights == {"gpu": 0.7, "cpu": 0.2, "ssd": 0.1}


# -- ④ Hydra parsing abnormal cases --


class TestStrategyHydraAbnormal:
    """Invalid _target_ / strategies configuration is rejected."""

    def test_instantiate_rejects_nonexistent_module(self):
        """
        Feature: instantiate fails when _target_ module does not exist
        Description: instantiate an entry whose _target_ points to nonexistent.Module.Class
        Expectation: raises InstantiationException/ImportError/ConfigError
        """
        from hydra.utils import instantiate

        entry = OmegaConf.create({"_target_": "nonexistent.Module.Class"})
        with pytest.raises((InstantiationException, ImportError, ConfigError)):
            instantiate(entry)

    @pytest.mark.parametrize(
        "strategies,match",
        [
            # missing _target_ on a strategy item
            ({"strategies": [{"weight": 1.0}]}, "_target_"),
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
    def test_from_config_rejects_invalid_strategies(self, strategies, match):
        """
        Feature: from_config rejects invalid strategies configuration
        Description: missing _target_ / non-StrategyConfig _target_ / empty config / null strategies
        Expectation: raises ConfigError matching the relevant keyword
        """
        kwargs = OmegaConf.create(strategies)
        with pytest.raises(ConfigError, match=match):
            KVCAwareConfig.from_config(kwargs)


# ============================================================
# 5.2 CollectorConfig
# ============================================================

# -- ① Input/output normal cases --


class TestMetricsNormalInput:
    """CollectorConfig normal construction: defaults, custom values, dataclass shape."""

    @pytest.mark.parametrize(
        "kwargs,attr,expected",
        [
            # http_polling default
            ({}, "http_polling", {"polling_interval": 5.0, "http_timeout": 10.0}),
            # http_polling custom
            (
                {"http_polling": {"polling_interval": 3.0, "http_timeout": 15.0}},
                "http_polling",
                {"polling_interval": 3.0, "http_timeout": 15.0},
            ),
            # long_connection default
            (
                {},
                "long_connection",
                {
                    "base_retry_delay": 1.0,
                    "max_retry_delay": 30.0,
                    "max_retry_attempts": 5,
                    "retry_backoff_factor": 2.0,
                },
            ),
            # long_connection custom
            (
                {
                    "long_connection": {
                        "base_retry_delay": 2.0,
                        "max_retry_delay": 60.0,
                        "max_retry_attempts": 10,
                        "retry_backoff_factor": 3.0,
                    }
                },
                "long_connection",
                {
                    "base_retry_delay": 2.0,
                    "max_retry_delay": 60.0,
                    "max_retry_attempts": 10,
                    "retry_backoff_factor": 3.0,
                },
            ),
        ],
    )
    def test_fields_parse_default_and_custom(self, kwargs, attr, expected):
        """
        Feature: CollectorConfig fields accept custom values and fall back to defaults
        Description: construct CollectorConfig with varied http_polling / long_connection kwargs
        Expectation: each field equals the expected explicit or default value
        """
        cfg = CollectorConfig(**kwargs)
        assert getattr(cfg, attr) == expected

    def test_collector_config_is_dataclass_with_connection_fields(self):
        """
        Feature: CollectorConfig is a dataclass exposing connection-type fields
        Description: inspect the dataclass type and field set of CollectorConfig
        Expectation: is_dataclass is True; field set is {http_polling, long_connection}
        """
        import dataclasses

        assert dataclasses.is_dataclass(CollectorConfig)
        assert {f.name for f in dataclasses.fields(CollectorConfig)} == {"http_polling", "long_connection"}


# -- ② Input/output abnormal cases --


class TestMetricsAbnormalInput:
    """CollectorConfig abnormal: per-field validation errors."""

    @pytest.mark.parametrize(
        "http_polling,match",
        [
            ({"polling_interval": 0, "http_timeout": 10}, "polling_interval"),
            ({"polling_interval": -1, "http_timeout": 10}, "polling_interval"),
            ({"polling_interval": 5, "http_timeout": 0}, "http_timeout"),
        ],
    )
    def test_http_polling_invalid_raises_config_error(self, http_polling, match):
        """
        Feature: invalid http_polling values trigger validation error
        Description: construct CollectorConfig with polling_interval=0/-1 or http_timeout=0
        Expectation: raises ConfigError matching the offending field name
        """
        with pytest.raises(ConfigError, match=match):
            CollectorConfig(http_polling=http_polling)

    @pytest.mark.parametrize(
        "long_connection,match",
        [
            ({"base_retry_delay": -1}, "base_retry_delay"),
            ({"base_retry_delay": 5, "max_retry_delay": 3}, "max_retry_delay"),
            ({"max_retry_attempts": 0}, "max_retry_attempts"),
            ({"retry_backoff_factor": -1}, "retry_backoff_factor"),
        ],
    )
    def test_long_connection_invalid_raises_config_error(self, long_connection, match):
        """
        Feature: invalid long_connection values trigger validation error
        Description: construct CollectorConfig with negative/zero/below-base retry params
        Expectation: raises ConfigError matching the offending field name
        """
        with pytest.raises(ConfigError, match=match):
            CollectorConfig(long_connection=long_connection)


# -- ③ Hydra parsing normal cases --
# (No collector sub-config instantiate — concrete subclasses removed)


# -- ⑤ Other cases --


class TestMetricsOther:
    """from_config: collector defaults and overrides."""

    _STRATEGY = [
        {
            "_target_": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
            "weight": 1.0,
            "collector_names": ["vllm_zmq"],
        }
    ]

    @pytest.mark.parametrize("collector", [None, {"http_polling": {"polling_interval": 3}}])
    def test_collector_defaults_or_override(self, collector):
        """
        Feature: collector takes defaults when omitted/null and accepts partial overrides
        Description: from_config with collector=None or a partial http_polling override
        Expectation: result.collector is a CollectorConfig with default or overridden values
        """
        kwargs = OmegaConf.create({"strategies": self._STRATEGY, "collector": collector})
        result = KVCAwareConfig.from_config(kwargs)
        assert isinstance(result.collector, CollectorConfig)
        if collector and "http_polling" in collector:
            assert result.collector.http_polling["polling_interval"] == 3.0
        else:
            assert result.collector.http_polling == {"polling_interval": 5.0, "http_timeout": 10.0}


# ============================================================
# 5.3 CacheStoreConfig
# ============================================================

# -- ① Input/output normal cases --


class TestCacheStoreNormalInput:
    """CacheStoreConfig normal construction: store type and ttl defaults."""

    @pytest.mark.parametrize(
        "kwargs,attr,expected",
        [
            ({"kv_cache_store_type": "list"}, "kv_cache_store_type", "list"),
            ({"kv_cache_store_type": "radix_tree"}, "kv_cache_store_type", "radix_tree"),
            ({}, "kv_cache_store_type", "list"),  # default store type
            ({"ttl": 30.0}, "ttl", 30.0),
            ({}, "ttl", 30.0),  # default ttl
        ],
    )
    def test_fields_parse_default_and_custom(self, kwargs, attr, expected):
        """
        Feature: CacheStoreConfig fields accept custom values and fall back to defaults
        Description: construct CacheStoreConfig with varied kv_cache_store_type / ttl kwargs
        Expectation: each field equals the expected explicit or default value
        """
        cfg = CacheStoreConfig(**kwargs)
        assert getattr(cfg, attr) == expected


# -- ② Input/output abnormal cases --


class TestCacheStoreAbnormalInput:
    """CacheStoreConfig abnormal: per-field validation errors."""

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"kv_cache_store_type": "unknown"}, "kv_cache_store_type"),
            ({"ttl": 0}, "ttl"),
            ({"ttl": -1}, "ttl"),
        ],
    )
    def test_field_invalid_raises_config_error(self, kwargs, match):
        """
        Feature: invalid kv_cache_store_type or ttl triggers validation error
        Description: construct CacheStoreConfig with unknown type / ttl=0 / ttl=-1
        Expectation: raises ConfigError matching the offending field name
        """
        with pytest.raises(ConfigError, match=match):
            CacheStoreConfig(**kwargs)

    def test_cache_store_not_dict_raises_config_error(self):
        """
        Feature: non-dict cache_store triggers from_config validation error
        Description: from_config with cache_store="list"
        Expectation: raises ConfigError matching "cache_store"
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    }
                ],
                "cache_store": "list",
            }
        )
        with pytest.raises(ConfigError, match="cache_store"):
            KVCAwareConfig.from_config(kwargs)


# -- ⑤ Other cases --


class TestCacheStoreOther:
    """from_config: cache_store defaults when omitted or null."""

    _STRATEGY = [
        {
            "_target_": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
            "weight": 1.0,
            "collector_names": ["vllm_zmq"],
        }
    ]

    def test_cache_store_defaults_when_omitted_or_null(self):
        """
        Feature: cache_store takes defaults when omitted or null
        Description: from_config with cache_store omitted or explicitly None
        Expectation: result.cache_store is a CacheStoreConfig with default type/ttl
        """
        for cfg in ({"strategies": self._STRATEGY}, {"strategies": self._STRATEGY, "cache_store": None}):
            result = KVCAwareConfig.from_config(OmegaConf.create(cfg))
            assert isinstance(result.cache_store, CacheStoreConfig)
            assert result.cache_store.kv_cache_store_type == "list"
            assert result.cache_store.ttl == 30.0


# ============================================================
# 5.4 KVCAwareConfig top-level
# ============================================================

# -- ① Input/output normal cases --


class TestKVCAwareNormalInput:
    """from_config assembles the three sections into a KVCAwareConfig."""

    @staticmethod
    def _full_config():
        return {
            "strategies": [
                {
                    "_target_": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
                    "weight": 1.0,
                    "alpha": 0.7,
                    "collector_names": ["vllm_zmq", "mooncake_prometheus"],
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

    @pytest.mark.parametrize("wrap", [lambda d: OmegaConf.create(d), lambda d: d], ids=["omegaconf", "plain_dict"])
    def test_full_config_assembles_three_sections(self, wrap):
        """
        Feature: from_config assembles the three sections into a KVCAwareConfig
        Description: from_config with a full strategies/collector/cache_store config (OmegaConf or plain dict)
        Expectation: result is a KVCAwareConfig with correctly typed sections
        """
        result = KVCAwareConfig.from_config(wrap(self._full_config()))
        assert isinstance(result, KVCAwareConfig)
        assert isinstance(result.strategies[0], KVCAwareStrategyConfig)
        assert isinstance(result.collector, CollectorConfig)
        assert isinstance(result.cache_store, CacheStoreConfig)


# -- ② Input/output abnormal cases --


class TestKVCAwareAbnormalInput:
    """from_config rejects malformed top-level / section inputs."""

    def test_from_config_rejects_non_dict_collector(self):
        """
        Feature: non-dict collector triggers from_config validation error
        Description: from_config with collector="vllm"
        Expectation: raises ConfigError matching "collector"
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    }
                ],
                "collector": "vllm",
            }
        )
        with pytest.raises(ConfigError, match="collector"):
            KVCAwareConfig.from_config(kwargs)

    def test_from_config_rejects_collector_unknown_key(self):
        """
        Feature: collector with an undefined key triggers structural validation error
        Description: from_config with collector containing unknown_key
        Expectation: raises OmegaConfBaseException matching "unknown_key"
        """
        from omegaconf.errors import OmegaConfBaseException

        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    }
                ],
                "collector": {"unknown_key": 123, "http_polling": {"polling_interval": 5}},
            }
        )
        with pytest.raises(OmegaConfBaseException, match="unknown_key"):
            KVCAwareConfig.from_config(kwargs)

    def test_from_config_aggregates_multiple_errors(self):
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


# -- ③ Hydra parsing normal cases --


class TestKVCAwareHydraNormal:
    """from_config handles OmegaConf inputs and section defaults."""

    def test_strategies_dict_converts_to_list(self):
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
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    },
                },
            }
        )
        result = KVCAwareConfig.from_config(kwargs)
        assert isinstance(result.strategies, list)
        assert len(result.strategies) == 1
        assert isinstance(result.strategies[0], KVCAwareStrategyConfig)
        assert result.strategies[0].weight == 1.0

    def test_omitted_sections_take_defaults(self):
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
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
                    },
                ],
            }
        )
        result = KVCAwareConfig.from_config(kwargs)
        assert isinstance(result.collector, CollectorConfig)
        assert isinstance(result.cache_store, CacheStoreConfig)


# -- ④ Hydra parsing abnormal cases --


class TestKVCAwareHydraAbnormal:
    """from_config drops non-domain top-level keys; custom repr is preserved."""

    def test_top_level_unknown_keys_ignored(self):
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
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq"],
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

    def test_compact_repr(self):
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
                        "weight": 1.0,
                        "collector_names": ["vllm_zmq", "mooncake_prometheus"],
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
        assert "weight=1.0" in r
        assert "collector_names" in r


# -- ⑤ Other cases --


def _load_pkg_router_yaml() -> dict:
    """Compose ``uni_agent/llm_router/configs/kvc_aware_router.yaml`` via Hydra.

    Expands the YAML defaults block into one config dictionary.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    import uni_agent.llm_router.configs as _cfg_pkg

    config_dir = str(next(iter(_cfg_pkg.__path__)))
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="kvc_aware_router")
    return OmegaConf.to_container(cfg, resolve=True)


class TestKVCAwareOther:
    """multi-strategy assembly and packaged YAML end-to-end."""

    def test_multi_strategy_instances(self):
        """
        Feature: from_config with multiple strategies yields a list of StrategyConfig
        Description: from_config with a config containing two strategies
        Expectation: each strategy is a StrategyConfig instance
        """
        kwargs = OmegaConf.create(
            {
                "strategies": [
                    {
                        "_target_": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
                        "weight": 0.7,
                        "collector_names": ["vllm_zmq"],
                    },
                    {
                        "_target_": "uni_agent.llm_router.config.strategy.KVCAwareStrategyConfig",
                        "weight": 0.3,
                        "collector_names": ["vllm_prometheus"],
                    },
                ],
            }
        )
        result = KVCAwareConfig.from_config(kwargs)
        assert len(result.strategies) == 2
        for s in result.strategies:
            assert isinstance(s, StrategyConfig)

    def test_packaged_yaml_e2e(self):
        """
        Feature: Hydra composition of the packaged router YAML parses end-to-end
        Description: compose configs/kvc_aware_router.yaml, then from_config
        Expectation: all fields match the YAML config and non-domain keys are dropped
        """
        loaded = _load_pkg_router_yaml()

        result = KVCAwareConfig.from_config(loaded)

        # ── strategies ──
        assert isinstance(result.strategies, list)
        assert len(result.strategies) == 1
        strategy = result.strategies[0]
        assert isinstance(strategy, KVCAwareStrategyConfig)
        assert strategy.weight == 1.0
        assert strategy.alpha == 0.3
        assert strategy.load_threshold == 0.6
        assert strategy.layer_weights == {"gpu": 0.7, "cpu": 0.2, "ssd": 0.1}
        assert strategy.collector_names == [
            "vllm_zmq",
            "vllm_metrics",
            "sticky_stat",
            "inflight_stat",
        ]
        # FQN 顶层键对 from_config 无害（多余键被丢弃）
        assert loaded["router_class"] == "uni_agent.llm_router.balancer.KVCAwareBalancer"
        assert not hasattr(result, "router_class")

"""Platform capability profiles for multi-platform parallel dispatch.

This module is intentionally **static and pure configuration**: it never reads
dataset files, never calls an LLM, and never talks to a real platform API.

Design constraints taken from the project brief:

* The dataset now carries a **simulated platform per user**
  (``UserProfile.platform`` ∈ ``facebook_simulated`` / ``instagram_simulated``
  / ``twitter_simulated``). Platform keys therefore describe the simulated
  platform a digital human actually belongs to. Every profile still carries an
  explicit ``policy_disclaimer`` stating that all platforms are simulated and
  no real accounts are involved.
* Platform differences must change role assignment and content strategy, not
  just the platform label. Each profile therefore defines its own role
  vocabulary, content template sections, time blueprint and signal weights.

All score-like values in this project are normalised to ``[0, 1]`` by
``Scorer._clip_score``; the ``signal_weights`` / ``role_affinity`` /
``dispatch_score_weights`` defined below are all relative weights over that
same normalised space, so they can be combined safely without rescaling.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

_TRUE_TOLERANCE = 1e-6

#: 数据集中每个数字人可归属的三个模拟平台 key（小写、无后缀）。
SIMULATED_PLATFORM_KEYS: tuple[str, ...] = ("facebook", "instagram", "twitter")

#: 数据集 ``UserProfile.platform`` 字段的合法取值（带 ``_simulated`` 后缀）。
DATASET_PLATFORM_VALUES: tuple[str, ...] = tuple(
    f"{key}_simulated" for key in SIMULATED_PLATFORM_KEYS
)


def normalize_platform_key(value: str | None) -> str:
    """把任意书写形式的平台名归一化成注册表 key。

    * 大小写与首尾空格会被清理；
    * ``facebook_simulated`` / ``Facebook Simulated`` 等带模拟后缀的写法
      会归一化成 ``facebook``；
    * 无法识别时返回空字符串，由调用方决定如何报错。
    """

    if value is None:
        return ""
    normalized = str(value).strip().lower()
    if normalized.endswith("_simulated"):
        normalized = normalized[: -len("_simulated")]
    return normalized


def dataset_platform_value(platform_key: str) -> str:
    """把注册表 key 转成数据集 ``platform`` 字段的取值（加模拟后缀）。"""

    return f"{normalize_platform_key(platform_key)}_simulated"


def is_valid_dataset_platform(value: str | None) -> bool:
    return normalize_platform_key(value) in SIMULATED_PLATFORM_KEYS


class UnsupportedPlatformError(ValueError):
    """Raised when a platform key is not present in the registry."""


class PlatformContentSection(BaseModel):
    """One section of a platform content template (e.g. Twitter thread opening)."""

    model_config = ConfigDict(extra="forbid")

    section_key: str
    label: str
    guidance: str


class PlatformContentTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_modes: list[str] = Field(default_factory=list)
    sections: list[PlatformContentSection] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)


class PlatformTimeStageSpec(BaseModel):
    """A stage blueprint expressed as ratios of the campaign window."""

    model_config = ConfigDict(extra="forbid")

    stage_key: str
    label: str
    ratio_start: float = Field(ge=0.0, le=1.0)
    ratio_end: float = Field(ge=0.0, le=1.0)
    focus: str

    @model_validator(mode="after")
    def _validate_window(self) -> "PlatformTimeStageSpec":
        if self.ratio_end <= self.ratio_start:
            raise ValueError(
                f"time stage '{self.stage_key}' must satisfy ratio_end > ratio_start"
            )
        return self


class PlatformProfile(BaseModel):
    """Static description of how one platform should be used as a strategy target."""

    model_config = ConfigDict(extra="forbid")

    platform_key: str
    display_name: str
    platform_roles: list[str] = Field(default_factory=list)
    content_template: PlatformContentTemplate
    time_blueprint: list[PlatformTimeStageSpec] = Field(default_factory=list)
    # role_hint (from FeatureBuilder) -> affinity in [0, 1]
    role_affinity: dict[str, float] = Field(default_factory=dict)
    # normalised signal name -> weight; must sum to 1.0
    signal_weights: dict[str, float] = Field(default_factory=dict)
    # convex combination of the two normalised scores; must sum to 1.0
    dispatch_score_weights: dict[str, float] = Field(default_factory=dict)
    cadence_notes: list[str] = Field(default_factory=list)
    fit_limitations: list[str] = Field(default_factory=list)
    policy_disclaimer: str = ""

    @model_validator(mode="after")
    def _validate_profile(self) -> "PlatformProfile":
        if self.platform_key != self.platform_key.lower().strip():
            raise ValueError(
                f"platform_key must be a lowercase key, got '{self.platform_key}'"
            )
        if not self.platform_roles:
            raise ValueError(f"platform '{self.platform_key}' must define platform_roles")
        if len(set(self.platform_roles)) != len(self.platform_roles):
            raise ValueError(
                f"platform '{self.platform_key}' has duplicated platform_roles"
            )
        if not self.content_template.sections:
            raise ValueError(
                f"platform '{self.platform_key}' must define content template sections"
            )
        if not self.time_blueprint:
            raise ValueError(f"platform '{self.platform_key}' must define a time_blueprint")

        self._validate_weights(self.signal_weights, "signal_weights")
        self._validate_weights(self.dispatch_score_weights, "dispatch_score_weights")

        for role_hint, affinity in self.role_affinity.items():
            if not 0.0 <= affinity <= 1.0:
                raise ValueError(
                    f"platform '{self.platform_key}' role_affinity['{role_hint}'] "
                    f"must be within [0, 1], got {affinity}"
                )

        for stage in self.time_blueprint:
            if stage.ratio_end > 1.0 + _TRUE_TOLERANCE:
                raise ValueError(
                    f"platform '{self.platform_key}' stage '{stage.stage_key}' "
                    "exceeds the campaign window"
                )
        return self

    def _validate_weights(self, weights: dict[str, float], field_name: str) -> None:
        if not weights:
            raise ValueError(
                f"platform '{self.platform_key}' must define {field_name}"
            )
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-3:
            raise ValueError(
                f"platform '{self.platform_key}' {field_name} must sum to 1.0, got {total:.4f}"
            )
        for key, value in weights.items():
            if value < 0.0:
                raise ValueError(
                    f"platform '{self.platform_key}' {field_name}['{key}'] must be >= 0"
                )


_DISCLAIMER = (
    "本平台为模拟平台：数字人的平台归属来自数据集中的 platform 字段（模拟账号），"
    "不代表该数字人在真实平台拥有真实账号或真实粉丝数；"
    "本结果只包含策略与内容草案，不包含真实平台发布行为。"
)

_FIT_LIMITATIONS = [
    "所有平台均为模拟平台，数字人平台归属来自数据集 platform 字段的模拟分配。",
    "平台适配分衡量的是数字人画像与平台策略偏好的匹配程度，不等同于真实粉丝数。",
]


FACEBOOK_PROFILE = PlatformProfile(
    platform_key="facebook",
    display_name="模拟Facebook",
    platform_roles=["community_discussion", "story_explainer", "community_moderator"],
    content_template=PlatformContentTemplate(
        content_modes=["社群讨论帖", "经验分享长文", "背景解释"],
        sections=[
            PlatformContentSection(
                section_key="post_theme",
                label="帖子主题",
                guidance="围绕「{event_title}」写一条面向{audience}的社群讨论帖主题，先说明讨论缘起。",
            ),
            PlatformContentSection(
                section_key="core_viewpoint",
                label="核心观点",
                guidance="用 2-3 句给出核心观点，说明它如何服务于目标「{target_goal}」，不写入未经证实的信息。",
            ),
            PlatformContentSection(
                section_key="discussion_question",
                label="讨论问题",
                guidance="提出 1 个开放式问题，邀请{audience}分享经验与背景，而不是争论立场。",
            ),
            PlatformContentSection(
                section_key="risk_note",
                label="风险提示",
                guidance="标注当前话题敏感度（事件风险等级 {risk_level}），提醒避免情绪化、对立化表达。",
            ),
        ],
        guardrails=[
            "不宣称该数字人在 Facebook 拥有真实账号或真实粉丝数。",
            "以经验分享和背景解释为主，不做煽动式号召。",
        ],
    ),
    time_blueprint=[
        PlatformTimeStageSpec(
            stage_key="community_open",
            label="社群开帖期",
            ratio_start=0.0,
            ratio_end=0.15,
            focus="发布社群讨论帖，铺垫话题并邀请第一轮经验分享。",
        ),
        PlatformTimeStageSpec(
            stage_key="discussion_sustain",
            label="讨论维持期",
            ratio_start=0.15,
            ratio_end=0.60,
            focus="持续回复评论、补充背景解释，维持长尾讨论。",
        ),
        PlatformTimeStageSpec(
            stage_key="background_share",
            label="背景沉淀期",
            ratio_start=0.60,
            ratio_end=1.0,
            focus="沉淀背景材料与案例，形成可复用的解释性内容。",
        ),
    ],
    role_affinity={
        "core_broadcast": 0.70,
        "interaction_response": 0.50,
        "amplification": 0.30,
        "general_support": 0.90,
    },
    signal_weights={
        "topic_match_score": 0.30,
        "stability_score": 0.25,
        "influence_score": 0.15,
        "diffusion_score": 0.10,
        "activity_score": 0.10,
        "role_affinity": 0.10,
    },
    dispatch_score_weights={"final_score": 0.60, "platform_fit_score": 0.40},
    cadence_notes=[
        "Facebook 适合慢节奏、长内容的社群讨论与背景解释。",
        "单日建议不超过 2 条主帖，优先保证评论回复质量。",
    ],
    fit_limitations=_FIT_LIMITATIONS,
    policy_disclaimer=_DISCLAIMER,
)


TWITTER_PROFILE = PlatformProfile(
    platform_key="twitter",
    display_name="模拟Twitter",
    platform_roles=["thread_writer", "retweet_amplifier", "reply_engager"],
    content_template=PlatformContentTemplate(
        content_modes=["推文串（thread）", "短评转发", "回复互动"],
        sections=[
            PlatformContentSection(
                section_key="thread_opening",
                label="推文串开头",
                guidance="用 1 句结论式开头（不超过 140 字）概括「{event_title}」的核心信息，不加情绪化修饰。",
            ),
            PlatformContentSection(
                section_key="thread_body",
                label="推文串主体",
                guidance="拆成 3-5 条连续推文，每条一个要点，只使用事件中已确认的事实。",
            ),
            PlatformContentSection(
                section_key="evidence_note",
                label="事实引用",
                guidance="对每个关键结论标注来源或依据，无法核实的说法不得写入。",
            ),
            PlatformContentSection(
                section_key="reply_prompt",
                label="回复引导",
                guidance="用 1 条推文引导{audience}在回复中补充背景或提问，不做立场煽动。",
            ),
        ],
        guardrails=[
            "不得为追求转发量而捏造事件事实、数据或当事人表态。",
            "不宣称该数字人在 Twitter 拥有真实账号或真实粉丝数。",
        ],
    ),
    time_blueprint=[
        PlatformTimeStageSpec(
            stage_key="thread_launch",
            label="推文串首发期",
            ratio_start=0.0,
            ratio_end=0.20,
            focus="集中发布推文串首条与主体，抢占话题早期注意力。",
        ),
        PlatformTimeStageSpec(
            stage_key="retweet_ride",
            label="转发扩散期",
            ratio_start=0.20,
            ratio_end=0.60,
            focus="围绕核心事实做短评转发与多角度补充推文。",
        ),
        PlatformTimeStageSpec(
            stage_key="reply_engage",
            label="回复互动期",
            ratio_start=0.60,
            ratio_end=1.0,
            focus="以回复互动为核心，处理高频疑问并纠正误读。",
        ),
    ],
    role_affinity={
        "core_broadcast": 0.60,
        "interaction_response": 0.80,
        "amplification": 0.70,
        "general_support": 0.40,
    },
    signal_weights={
        "topic_match_score": 0.25,
        "stability_score": 0.15,
        "influence_score": 0.20,
        "diffusion_score": 0.20,
        "activity_score": 0.10,
        "role_affinity": 0.10,
    },
    dispatch_score_weights={"final_score": 0.55, "platform_fit_score": 0.45},
    cadence_notes=[
        "Twitter 传播节奏快，建议把最关键事实放在推文串前两条。",
        "单日推文可略密，但同一话题不宜重复同质推文。",
    ],
    fit_limitations=_FIT_LIMITATIONS,
    policy_disclaimer=_DISCLAIMER,
)


INSTAGRAM_PROFILE = PlatformProfile(
    platform_key="instagram",
    display_name="模拟Instagram",
    platform_roles=["visual_storyteller", "reels_creator", "story_interactor"],
    content_template=PlatformContentTemplate(
        content_modes=["视觉叙事", "轮播图", "caption", "Stories 互动建议"],
        sections=[
            PlatformContentSection(
                section_key="carousel_theme",
                label="轮播图主题",
                guidance="给出轮播图主题，与「{event_title}」形成一条视觉叙事主线。",
            ),
            PlatformContentSection(
                section_key="carousel_frames",
                label="轮播分镜",
                guidance="设计 3-5 张轮播图分镜（画面内容 + 单张要点），第一张负责抓取注意力。",
            ),
            PlatformContentSection(
                section_key="caption_requirements",
                label="caption 要求",
                guidance="写一段 caption，包含 1 句核心观点与 2-3 个话题标签，措辞克制。",
            ),
            PlatformContentSection(
                section_key="stories_interaction",
                label="Stories 互动建议（可选）",
                guidance="给出可选的 Stories 问答或投票问题，用于回收{audience}的反馈。",
            ),
        ],
        guardrails=[
            "视觉叙事不得使用暗示真实事件结论的合成画面。",
            "不宣称该数字人在 Instagram 拥有真实账号或真实粉丝数。",
        ],
    ),
    time_blueprint=[
        PlatformTimeStageSpec(
            stage_key="visual_teaser",
            label="视觉预告期",
            ratio_start=0.0,
            ratio_end=0.20,
            focus="发布视觉预告与 Stories 互动，测试话题关注度。",
        ),
        PlatformTimeStageSpec(
            stage_key="carousel_drop",
            label="轮播发布期",
            ratio_start=0.20,
            ratio_end=0.55,
            focus="发布主轮播图，完整呈现视觉叙事主线。",
        ),
        PlatformTimeStageSpec(
            stage_key="stories_followup",
            label="Stories 跟进期",
            ratio_start=0.55,
            ratio_end=1.0,
            focus="用 Stories 问答或投票回收反馈并延长可见度。",
        ),
    ],
    role_affinity={
        "core_broadcast": 0.80,
        "interaction_response": 0.50,
        "amplification": 0.40,
        "general_support": 0.70,
    },
    signal_weights={
        "topic_match_score": 0.30,
        "stability_score": 0.20,
        "influence_score": 0.15,
        "diffusion_score": 0.10,
        "activity_score": 0.15,
        "role_affinity": 0.10,
    },
    dispatch_score_weights={"final_score": 0.60, "platform_fit_score": 0.40},
    cadence_notes=[
        "Instagram 以视觉为主，轮播图与 Stories 建议同日配合发布。",
        "caption 与分镜要点需一致，避免视觉与文本信息冲突。",
    ],
    fit_limitations=_FIT_LIMITATIONS,
    policy_disclaimer=_DISCLAIMER,
)


_PLATFORM_REGISTRY: dict[str, PlatformProfile] = {
    profile.platform_key: profile
    for profile in (FACEBOOK_PROFILE, INSTAGRAM_PROFILE, TWITTER_PROFILE)
}


def list_supported_platforms() -> list[str]:
    """Return the sorted list of supported lowercase platform keys."""

    return sorted(_PLATFORM_REGISTRY)


def get_platform_profile(platform_key: str) -> PlatformProfile:
    """Look up a platform profile by key.

    The lookup is case-insensitive and trims surrounding whitespace. Unknown
    platforms raise :class:`UnsupportedPlatformError` with a clear message that
    lists the supported keys.
    """

    if platform_key is None:
        raise UnsupportedPlatformError(
            f"平台键不能为空，当前支持的平台为：{', '.join(list_supported_platforms()) or '无'}。"
        )
    normalized = normalize_platform_key(platform_key)
    if not normalized:
        raise UnsupportedPlatformError(
            f"平台键不能为空，当前支持的平台为：{', '.join(list_supported_platforms()) or '无'}。"
        )
    profile = _PLATFORM_REGISTRY.get(normalized)
    if profile is None:
        supported = ", ".join(list_supported_platforms())
        raise UnsupportedPlatformError(
            f"不支持的平台 '{platform_key}'，当前支持的平台为：{supported}。"
        )
    return profile

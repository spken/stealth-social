"""Build immediate generation requests for configured autopost campaigns."""

from __future__ import annotations

from bot.config import AutopostCampaignSettings, Settings
from bot.content.models import ContentRequest, FactRequirement, GenerationType
from bot.content.requests import configured_account_context
from bot.models import Platform


def build_autopost_request(
    settings: Settings,
    campaign_id: str,
    campaign: AutopostCampaignSettings,
    topic: str,
) -> ContentRequest:
    """Translate one validated campaign occurrence into an immediate request."""

    generation_type = (
        GenerationType.X_POST
        if campaign.platform is Platform.X
        else GenerationType.REDDIT_POST
    )
    candidate_count = (
        campaign.candidate_count
        if campaign.candidate_count is not None
        else settings.content_generation.candidate_count
    )
    return ContentRequest(
        generation_type=generation_type,
        platform=campaign.platform,
        account_name=campaign.account,
        content_purpose=campaign.purpose,
        topic=topic,
        goal=campaign.goal,
        product_context=campaign.product_context,
        project_context=campaign.project_context,
        target_audience=campaign.target_audience,
        tone=campaign.tone,
        desired_length=campaign.desired_length,
        call_to_action=campaign.call_to_action,
        subreddit=campaign.subreddit,
        required_facts=tuple(
            FactRequirement(statement=fact) for fact in campaign.required_facts
        ),
        forbidden_claims=campaign.forbidden_claims,
        forbidden_phrases=campaign.forbidden_phrases,
        keywords=campaign.keywords,
        additional_instructions=campaign.additional_instructions,
        candidate_count=candidate_count,
        profile_name=campaign.profile_name,
        campaign_id=campaign_id,
        unattended_approval_requested=True,
        account_context=configured_account_context(
            settings,
            campaign.platform,
            campaign.account,
        ),
        resolved_parameters={"content_purpose_explicit": True},
    )


__all__ = ["build_autopost_request"]

"""Focused Typer command groups for the social bot."""

from bot.commands.actions import create_app, register_action_commands
from bot.commands.auth import register_auth_commands
from bot.commands.candidates import register_candidate_commands
from bot.commands.examples import register_example_commands
from bot.commands.generate import register_generation_commands
from bot.commands.ollama import register_ollama_commands
from bot.commands.topics import register_topic_commands
from bot.commands.worker import register_worker_command, run_publishing_worker

__all__ = [
    "create_app",
    "register_action_commands",
    "register_auth_commands",
    "register_candidate_commands",
    "register_example_commands",
    "register_generation_commands",
    "register_ollama_commands",
    "register_topic_commands",
    "register_worker_command",
    "run_publishing_worker",
]

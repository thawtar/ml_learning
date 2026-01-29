"""Tests for the intent classifier."""

import pytest
from openfoam_llm_wrapper.core.intent_classifier import IntentClassifier, Intent


def test_generate_case_intent():
    """Test classification of case generation intent."""
    classifier = IntentClassifier()

    intents = [
        "Generate a case for pipe flow",
        "Create a new simulation",
        "Setup an incompressible flow case",
    ]

    for text in intents:
        result = classifier.classify(text)
        assert result == Intent.GENERATE_CASE


def test_explain_error_intent():
    """Test classification of error explanation intent."""
    classifier = IntentClassifier()

    intents = [
        "What does this error mean?",
        "Explain this error: ...",
        "I got an error, can you help?",
    ]

    for text in intents:
        result = classifier.classify(text)
        assert result == Intent.EXPLAIN_ERROR


def test_solver_recommendation_intent():
    """Test classification of solver recommendation intent."""
    classifier = IntentClassifier()

    intents = [
        "Recommend a solver for incompressible flow",
        "Which solver should I use?",
        "I need a compressible solver",
    ]

    for text in intents:
        result = classifier.classify(text)
        assert result == Intent.RECOMMEND_SOLVER


def test_help_intent():
    """Test classification of help intent."""
    classifier = IntentClassifier()

    intents = [
        "Help, I don't know what to do",
        "How do I use this tool?",
        "Tutorial please",
    ]

    for text in intents:
        result = classifier.classify(text)
        assert result == Intent.HELP

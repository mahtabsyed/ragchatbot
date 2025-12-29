"""
Shared mock data and fixtures for testing the RAG system
"""
import sys
import os
from unittest.mock import Mock

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import SearchResults
from models import Course, Lesson, CourseChunk


class MockFixtures:
    """Factory class for creating consistent mock objects across tests"""

    @staticmethod
    def create_sample_search_results():
        """Create sample SearchResults for testing"""
        return SearchResults(
            documents=[
                "MCP is Model Context Protocol, a standard for connecting AI assistants to data sources.",
                "It enables AI assistants to securely access local and remote resources.",
                "MCP supports tools, prompts, and resources as first-class primitives."
            ],
            metadata=[
                {"course_title": "MCP: Build Rich-Context AI Apps with Anthropic", "lesson_number": 1, "chunk_index": 0},
                {"course_title": "MCP: Build Rich-Context AI Apps with Anthropic", "lesson_number": 2, "chunk_index": 1},
                {"course_title": "MCP: Build Rich-Context AI Apps with Anthropic", "lesson_number": 3, "chunk_index": 2}
            ],
            distances=[0.3, 0.5, 0.7]
        )

    @staticmethod
    def create_empty_search_results():
        """Create empty SearchResults"""
        return SearchResults(documents=[], metadata=[], distances=[])

    @staticmethod
    def create_error_search_results(error_msg):
        """Create SearchResults with error"""
        return SearchResults.empty(error_msg)

    @staticmethod
    def create_sample_course():
        """Create sample Course object"""
        return Course(
            title="MCP: Build Rich-Context AI Apps with Anthropic",
            course_link="https://www.deeplearning.ai/short-courses/mcp/",
            instructor="Alex Reibman",
            lessons=[
                Lesson(lesson_number=0, title="Introduction", lesson_link="https://example.com/lesson0"),
                Lesson(lesson_number=1, title="Why MCP", lesson_link="https://example.com/lesson1"),
                Lesson(lesson_number=2, title="MCP Architecture", lesson_link="https://example.com/lesson2")
            ]
        )

    @staticmethod
    def create_anthropic_response_no_tool():
        """Mock Anthropic response without tool use"""
        response = Mock()
        response.stop_reason = "end_turn"
        text_block = Mock()
        text_block.text = "This is a direct answer without using tools."
        response.content = [text_block]
        return response

    @staticmethod
    def create_anthropic_response_with_tool(tool_name="search_course_content", query="test query"):
        """Mock Anthropic response with tool use"""
        response = Mock()
        response.stop_reason = "tool_use"

        # Create tool use block
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = tool_name
        tool_block.input = {"query": query}
        tool_block.id = "toolu_test123"

        response.content = [tool_block]
        return response

    @staticmethod
    def create_anthropic_final_response(text="This is the synthesized answer based on search results."):
        """Mock Anthropic final response after tool execution"""
        response = Mock()
        response.stop_reason = "end_turn"
        text_block = Mock()
        text_block.text = text
        response.content = [text_block]
        return response

    @staticmethod
    def create_sample_course_chunks():
        """Create sample CourseChunk objects"""
        return [
            CourseChunk(
                course_title="MCP: Build Rich-Context AI Apps with Anthropic",
                lesson_number=1,
                chunk_index=0,
                content="MCP is Model Context Protocol..."
            ),
            CourseChunk(
                course_title="MCP: Build Rich-Context AI Apps with Anthropic",
                lesson_number=1,
                chunk_index=1,
                content="It enables AI assistants to connect..."
            )
        ]

    @staticmethod
    def create_anthropic_response_with_custom_tool(tool_name, **tool_params):
        """
        Mock Anthropic response with tool use and custom parameters

        Args:
            tool_name: Name of the tool being called
            **tool_params: Parameters to pass to the tool

        Returns:
            Mock response object with tool_use stop_reason
        """
        response = Mock()
        response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = tool_name
        tool_block.input = tool_params
        tool_block.id = f"toolu_{tool_name}_{hash(frozenset(tool_params.items())) % 10000}"
        response.content = [tool_block]
        return response

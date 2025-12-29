"""
Integration tests for RAGSystem query handling

Tests cover:
- Query processing with/without sessions
- Session history integration
- Source retrieval and reset flow
- End-to-end query flow with/without search
"""
import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from fixtures import MockFixtures


class TestRAGSystem(unittest.TestCase):
    """Integration tests for RAGSystem query handling"""

    def setUp(self):
        """Set up test fixtures before each test"""
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.CHUNK_SIZE = 800
        self.mock_config.CHUNK_OVERLAP = 100
        self.mock_config.CHROMA_PATH = "./test_chroma"
        self.mock_config.EMBEDDING_MODEL = "test-model"
        self.mock_config.MAX_RESULTS = 5
        self.mock_config.ANTHROPIC_API_KEY = "test-key"
        self.mock_config.ANTHROPIC_MODEL = "test-model"
        self.mock_config.MAX_HISTORY = 2

    def _create_rag_system_with_mocks(self):
        """Helper to create RAGSystem with all dependencies mocked"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager'), \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):

            rag_system = RAGSystem(self.mock_config)

            # Replace with fresh mocks for testing
            rag_system.ai_generator = Mock()
            rag_system.tool_manager = Mock()
            rag_system.session_manager = Mock()

            return rag_system

    def test_query_without_session(self):
        """Test query processing without session ID"""
        # Arrange
        rag_system = self._create_rag_system_with_mocks()
        rag_system.ai_generator.generate_response.return_value = "This is the answer."
        rag_system.tool_manager.get_last_sources.return_value = []

        # Act
        response, sources = rag_system.query(query="What is MCP?", session_id=None)

        # Assert
        self.assertEqual(response, "This is the answer.")
        self.assertEqual(sources, [])

        # Verify no history used
        rag_system.session_manager.get_conversation_history.assert_not_called()

        # Verify prompt formatting
        call_kwargs = rag_system.ai_generator.generate_response.call_args.kwargs
        self.assertIn("Answer this question about course materials:", call_kwargs['query'])
        self.assertIn("What is MCP?", call_kwargs['query'])

    def test_query_with_session_history(self):
        """Test query with existing session and conversation history"""
        # Arrange
        rag_system = self._create_rag_system_with_mocks()
        session_id = "session_test_123"
        history = "User: What is MCP?\nAssistant: MCP is Model Context Protocol."

        rag_system.session_manager.get_conversation_history.return_value = history
        rag_system.ai_generator.generate_response.return_value = "More details about MCP..."
        rag_system.tool_manager.get_last_sources.return_value = []

        # Act
        response, sources = rag_system.query(
            query="Tell me more",
            session_id=session_id
        )

        # Assert
        self.assertEqual(response, "More details about MCP...")

        # Verify history was retrieved
        rag_system.session_manager.get_conversation_history.assert_called_once_with(session_id)

        # Verify history passed to AIGenerator
        call_kwargs = rag_system.ai_generator.generate_response.call_args.kwargs
        self.assertEqual(call_kwargs['conversation_history'], history)

        # Verify session updated with new exchange (stores original query, not formatted prompt)
        rag_system.session_manager.add_exchange.assert_called_once_with(
            session_id,
            "Tell me more",  # Original query, not formatted prompt
            "More details about MCP..."
        )

    def test_query_sources_retrieved_and_reset(self):
        """Test that sources are properly retrieved and then reset - CRITICAL"""
        # Arrange
        rag_system = self._create_rag_system_with_mocks()

        # Mock sources from search
        mock_sources = [
            {"label": "MCP Course - Lesson 1", "link": "https://example.com/lesson1"},
            {"label": "MCP Course - Lesson 2", "link": "https://example.com/lesson2"}
        ]

        rag_system.ai_generator.generate_response.return_value = "Answer based on search results."
        rag_system.tool_manager.get_last_sources.return_value = mock_sources

        # Act
        response, sources = rag_system.query(query="What is MCP?")

        # Assert
        self.assertEqual(sources, mock_sources)

        # Verify get_last_sources was called
        rag_system.tool_manager.get_last_sources.assert_called_once()

        # Verify reset_sources was called AFTER getting sources
        rag_system.tool_manager.reset_sources.assert_called_once()

        # Verify call order: get_last_sources before reset_sources
        manager_calls = rag_system.tool_manager.method_calls
        get_index = next(i for i, call in enumerate(manager_calls) if call[0] == 'get_last_sources')
        reset_index = next(i for i, call in enumerate(manager_calls) if call[0] == 'reset_sources')
        self.assertLess(get_index, reset_index, "get_last_sources should be called before reset_sources")

    def test_query_end_to_end_with_search(self):
        """Test complete query flow when search tool is used"""
        # Arrange
        rag_system = self._create_rag_system_with_mocks()

        # Mock the complete flow
        mock_sources = [{"label": "MCP Course", "link": "https://example.com"}]
        rag_system.ai_generator.generate_response.return_value = "MCP is Model Context Protocol based on the search results."
        rag_system.tool_manager.get_last_sources.return_value = mock_sources
        rag_system.tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search courses"}
        ]

        # Act
        response, sources = rag_system.query(query="What is MCP?")

        # Assert
        self.assertIn("MCP is Model Context Protocol", response)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]['label'], "MCP Course")

        # Verify AIGenerator was called with tools
        call_kwargs = rag_system.ai_generator.generate_response.call_args.kwargs
        self.assertIsNotNone(call_kwargs['tools'])
        self.assertIsNotNone(call_kwargs['tool_manager'])

    def test_query_end_to_end_without_search(self):
        """Test complete query flow when no search tool is used"""
        # Arrange
        rag_system = self._create_rag_system_with_mocks()

        # Mock direct answer (no tool use)
        rag_system.ai_generator.generate_response.return_value = "2 plus 2 equals 4."
        rag_system.tool_manager.get_last_sources.return_value = []  # No sources when no search

        # Act
        response, sources = rag_system.query(query="What is 2+2?")

        # Assert
        self.assertEqual(response, "2 plus 2 equals 4.")
        self.assertEqual(sources, [])

        # Verify get_last_sources was still called (even if empty)
        rag_system.tool_manager.get_last_sources.assert_called_once()

        # Verify reset still called
        rag_system.tool_manager.reset_sources.assert_called_once()

    def test_prompt_formatting(self):
        """Test that query is properly formatted as a prompt"""
        # Arrange
        rag_system = self._create_rag_system_with_mocks()
        rag_system.ai_generator.generate_response.return_value = "Answer"
        rag_system.tool_manager.get_last_sources.return_value = []

        # Act
        response, sources = rag_system.query(query="Test query")

        # Assert
        call_kwargs = rag_system.ai_generator.generate_response.call_args.kwargs
        query_arg = call_kwargs['query']

        # Verify prompt structure
        self.assertEqual(query_arg, "Answer this question about course materials: Test query")

        # Verify other parameters passed correctly
        self.assertEqual(call_kwargs['tools'], rag_system.tool_manager.get_tool_definitions.return_value)
        self.assertEqual(call_kwargs['tool_manager'], rag_system.tool_manager)


if __name__ == '__main__':
    unittest.main()

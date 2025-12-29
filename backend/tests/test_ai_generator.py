"""
Unit tests for AIGenerator tool calling mechanism

Tests cover:
- Response generation without tools
- Response generation with conversation history
- Tool use decision logic
- Tool execution flow and message structure
- API parameter consistency
"""
import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from tests.fixtures import MockFixtures


class TestAIGenerator(unittest.TestCase):
    """Test cases for AIGenerator tool calling mechanism"""

    def setUp(self):
        """Set up test fixtures before each test"""
        self.mock_client = Mock()
        self.api_key = "test-api-key"
        self.model = "claude-sonnet-4-20250514"

    def _create_generator_with_mock_client(self):
        """Helper to create AIGenerator with mocked client"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = self.mock_client
            generator = AIGenerator(self.api_key, self.model)
            # Replace the client with our mock
            generator.client = self.mock_client
            return generator

    def test_generate_response_without_tools(self):
        """Test response generation without tools provided"""
        # Arrange
        generator = self._create_generator_with_mock_client()
        mock_response = MockFixtures.create_anthropic_response_no_tool()
        self.mock_client.messages.create.return_value = mock_response

        # Act
        result = generator.generate_response(
            query="What is 2+2?",
            tools=None,
            tool_manager=None
        )

        # Assert
        self.assertEqual(result, "This is a direct answer without using tools.")
        self.mock_client.messages.create.assert_called_once()

        # Verify no tools in API call
        call_kwargs = self.mock_client.messages.create.call_args.kwargs
        self.assertNotIn('tools', call_kwargs)
        self.assertNotIn('tool_choice', call_kwargs)

    def test_generate_response_with_conversation_history(self):
        """Test that conversation history is included in system prompt"""
        # Arrange
        generator = self._create_generator_with_mock_client()
        mock_response = MockFixtures.create_anthropic_response_no_tool()
        self.mock_client.messages.create.return_value = mock_response

        history = "User: What is MCP?\nAssistant: MCP is Model Context Protocol."

        # Act
        result = generator.generate_response(
            query="Tell me more",
            conversation_history=history,
            tools=None,
            tool_manager=None
        )

        # Assert
        call_kwargs = self.mock_client.messages.create.call_args.kwargs
        system_content = call_kwargs['system']

        # Verify history is in system prompt
        self.assertIn("Previous conversation:", system_content)
        self.assertIn("What is MCP?", system_content)
        self.assertIn("MCP is Model Context Protocol", system_content)

    def test_generate_response_no_tool_use_needed(self):
        """Test when tools are available but Claude doesn't use them"""
        # Arrange
        generator = self._create_generator_with_mock_client()
        mock_response = MockFixtures.create_anthropic_response_no_tool()
        self.mock_client.messages.create.return_value = mock_response

        mock_tool_manager = Mock()
        tools = [{"name": "search_course_content", "description": "Search courses"}]

        # Act
        result = generator.generate_response(
            query="What is 2+2?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert
        self.assertEqual(result, "This is a direct answer without using tools.")

        # Verify tool manager was NOT called
        mock_tool_manager.execute_tool.assert_not_called()

        # Verify only one API call
        self.assertEqual(self.mock_client.messages.create.call_count, 1)

    def test_generate_response_with_tool_use(self):
        """Test complete tool calling flow - CRITICAL TEST"""
        # Arrange
        generator = self._create_generator_with_mock_client()

        # Mock first response (tool_use)
        mock_initial_response = MockFixtures.create_anthropic_response_with_tool(
            tool_name="search_course_content",
            query="What is MCP?"
        )

        # Mock final response
        mock_final_response = MockFixtures.create_anthropic_final_response(
            text="MCP is Model Context Protocol, which enables AI assistants to connect to data sources."
        )

        # Set up sequential responses
        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "[MCP Course]\nMCP is Model Context Protocol..."

        tools = [{"name": "search_course_content", "description": "Search courses"}]

        # Act
        result = generator.generate_response(
            query="What is MCP?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert - Final result
        self.assertEqual(result, "MCP is Model Context Protocol, which enables AI assistants to connect to data sources.")

        # Assert - Two API calls made
        self.assertEqual(self.mock_client.messages.create.call_count, 2)

        # Assert - First call has tools
        first_call_kwargs = self.mock_client.messages.create.call_args_list[0].kwargs
        self.assertIn('tools', first_call_kwargs)
        self.assertEqual(first_call_kwargs['tool_choice']['type'], 'auto')

        # Assert - Tool execution
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="What is MCP?"
        )

        # Assert - Second call DOES have tools (allows sequential calling up to MAX_TOOL_ROUNDS)
        second_call_kwargs = self.mock_client.messages.create.call_args_list[1].kwargs
        self.assertIn('tools', second_call_kwargs)  # Updated for sequential tool calling
        self.assertIn('tool_choice', second_call_kwargs)  # Updated for sequential tool calling

    def test_handle_tool_execution_message_structure(self):
        """Test that message structure matches API spec exactly"""
        # Arrange
        generator = self._create_generator_with_mock_client()

        # Create tool use block with exact structure
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "test query", "course_name": "MCP"}
        tool_block.id = "toolu_abc123"

        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [tool_block]

        mock_final_response = MockFixtures.create_anthropic_final_response()

        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results here"

        tools = [{"name": "search_course_content"}]

        # Act
        result = generator.generate_response(
            query="test",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert - Verify message structure in second API call
        second_call_kwargs = self.mock_client.messages.create.call_args_list[1].kwargs
        messages = second_call_kwargs['messages']

        # Should have 3 messages
        self.assertEqual(len(messages), 3)

        # Message 1: User query
        self.assertEqual(messages[0]['role'], 'user')
        self.assertIn('test', messages[0]['content'])

        # Message 2: Assistant with tool_use
        self.assertEqual(messages[1]['role'], 'assistant')
        self.assertEqual(messages[1]['content'][0].type, 'tool_use')

        # Message 3: User with tool_result
        self.assertEqual(messages[2]['role'], 'user')
        tool_results = messages[2]['content']
        self.assertEqual(len(tool_results), 1)
        self.assertEqual(tool_results[0]['type'], 'tool_result')
        self.assertEqual(tool_results[0]['tool_use_id'], 'toolu_abc123')
        self.assertEqual(tool_results[0]['content'], 'Search results here')

    def test_multiple_tool_calls_in_response(self):
        """Test handling of multiple tool use blocks in one response"""
        # Arrange
        generator = self._create_generator_with_mock_client()

        # Create two tool use blocks
        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.input = {"query": "MCP"}
        tool_block_1.id = "tool_1"

        tool_block_2 = Mock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "get_course_outline"
        tool_block_2.input = {"course_title": "MCP"}
        tool_block_2.id = "tool_2"

        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [tool_block_1, tool_block_2]

        mock_final_response = MockFixtures.create_anthropic_final_response()

        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Search result 1",
            "Outline result 2"
        ]

        tools = [{"name": "search_course_content"}, {"name": "get_course_outline"}]

        # Act
        result = generator.generate_response(
            query="test",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert - Both tools executed
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)

        # Verify first tool call
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="MCP")

        # Verify second tool call
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_title="MCP")

        # Verify both results sent back in one message
        second_call_kwargs = self.mock_client.messages.create.call_args_list[1].kwargs
        tool_results = second_call_kwargs['messages'][2]['content']
        self.assertEqual(len(tool_results), 2)
        self.assertEqual(tool_results[0]['tool_use_id'], 'tool_1')
        self.assertEqual(tool_results[1]['tool_use_id'], 'tool_2')

    def test_api_parameters_consistency(self):
        """Test that API parameters are consistent across calls"""
        # Arrange
        generator = self._create_generator_with_mock_client()
        mock_response = MockFixtures.create_anthropic_response_no_tool()
        self.mock_client.messages.create.return_value = mock_response

        # Act
        result = generator.generate_response(query="test")

        # Assert
        call_kwargs = self.mock_client.messages.create.call_args.kwargs

        # Verify base parameters
        self.assertEqual(call_kwargs['model'], "claude-sonnet-4-20250514")
        self.assertEqual(call_kwargs['temperature'], 0)
        self.assertEqual(call_kwargs['max_tokens'], 800)

        # Verify messages structure
        self.assertIn('messages', call_kwargs)
        self.assertEqual(len(call_kwargs['messages']), 1)
        self.assertEqual(call_kwargs['messages'][0]['role'], 'user')

        # Verify system prompt
        self.assertIn('system', call_kwargs)
        self.assertIn('AI assistant specialized in course materials', call_kwargs['system'])

    def test_sequential_tool_calling_one_round(self):
        """Test single tool call followed by text answer (1 tool round)"""
        # Arrange
        generator = self._create_generator_with_mock_client()

        # Round 1: Claude uses tool
        mock_tool_use_response = MockFixtures.create_anthropic_response_with_tool(
            tool_name="search_course_content",
            query="What is MCP?"
        )

        # Round 2: Claude provides text answer
        mock_text_response = MockFixtures.create_anthropic_final_response(
            text="MCP is Model Context Protocol."
        )

        self.mock_client.messages.create.side_effect = [
            mock_tool_use_response,
            mock_text_response
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "MCP documentation..."

        tools = [{"name": "search_course_content"}]

        # Act
        result = generator.generate_response(
            query="What is MCP?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert
        self.assertEqual(result, "MCP is Model Context Protocol.")
        self.assertEqual(self.mock_client.messages.create.call_count, 2)
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 1)

        # Verify second call includes tools (allows round 2 if needed)
        second_call_kwargs = self.mock_client.messages.create.call_args_list[1].kwargs
        self.assertIn('tools', second_call_kwargs)

    def test_sequential_tool_calling_two_rounds(self):
        """Test two sequential tool calls followed by answer (2 tool rounds - MAX)"""
        # Arrange
        generator = self._create_generator_with_mock_client()

        # Round 1: Claude uses search tool
        mock_round1_response = MockFixtures.create_anthropic_response_with_tool(
            tool_name="search_course_content",
            query="MCP basics"
        )

        # Round 2: Claude uses search tool again with refined query
        mock_round2_response = MockFixtures.create_anthropic_response_with_custom_tool(
            tool_name="search_course_content",
            query="MCP architecture details",
            lesson_number=2
        )

        # Round 3: Claude provides text answer (no more tools allowed)
        mock_final_response = MockFixtures.create_anthropic_final_response(
            text="MCP is Model Context Protocol with client-server architecture."
        )

        self.mock_client.messages.create.side_effect = [
            mock_round1_response,
            mock_round2_response,
            mock_final_response
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "MCP is Model Context Protocol...",
            "MCP uses client-server architecture..."
        ]

        tools = [{"name": "search_course_content"}]

        # Act
        result = generator.generate_response(
            query="Explain MCP architecture in detail",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert
        self.assertEqual(result, "MCP is Model Context Protocol with client-server architecture.")
        self.assertEqual(self.mock_client.messages.create.call_count, 3)
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)

        # Verify API call structure
        # Call 1: Initial (has tools)
        # Call 2: After first tool result (has tools - allows round 2)
        # Call 3: After second tool result (NO tools - forces answer)

        first_call = self.mock_client.messages.create.call_args_list[0].kwargs
        second_call = self.mock_client.messages.create.call_args_list[1].kwargs
        third_call = self.mock_client.messages.create.call_args_list[2].kwargs

        self.assertIn('tools', first_call)
        self.assertIn('tools', second_call)  # CRITICAL: Tools still available
        self.assertNotIn('tools', third_call)  # CRITICAL: Tools removed to force answer

    def test_max_rounds_enforcement(self):
        """Test that system stops after 2 rounds even if Claude wants more tools"""
        # Arrange
        generator = self._create_generator_with_mock_client()

        # Simulate Claude wanting to use tools indefinitely
        mock_tool_response = MockFixtures.create_anthropic_response_with_tool()

        # Setup: Claude would use tools 3 times if allowed
        # But system should force text response after round 2
        mock_final_response = MockFixtures.create_anthropic_final_response("Forced answer based on available results")

        self.mock_client.messages.create.side_effect = [
            mock_tool_response,  # Round 1
            mock_tool_response,  # Round 2 (last round with tools)
            mock_final_response  # Round 3 (no tools, forced answer)
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search result"

        tools = [{"name": "search_course_content"}]

        # Act
        result = generator.generate_response(
            query="Complex query",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert
        self.assertIsNotNone(result)
        self.assertEqual(self.mock_client.messages.create.call_count, 3)
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)  # Only 2 tools executed

        # Verify third call has no tools (enforcement)
        third_call = self.mock_client.messages.create.call_args_list[2].kwargs
        self.assertNotIn('tools', third_call)

    def test_message_structure_sequential_calls(self):
        """Test that message history accumulates correctly across tool rounds"""
        # Arrange
        generator = self._create_generator_with_mock_client()

        # Create distinct tool blocks for each round
        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.input = {"query": "first query"}
        tool_block_1.id = "tool_1"

        tool_block_2 = Mock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "get_course_outline"
        tool_block_2.input = {"course_title": "MCP"}
        tool_block_2.id = "tool_2"

        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        round1_response.content = [tool_block_1]

        round2_response = Mock()
        round2_response.stop_reason = "tool_use"
        round2_response.content = [tool_block_2]

        final_response = MockFixtures.create_anthropic_final_response()

        self.mock_client.messages.create.side_effect = [
            round1_response,
            round2_response,
            final_response
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["result1", "result2"]

        tools = [{"name": "search_course_content"}, {"name": "get_course_outline"}]

        # Act
        result = generator.generate_response(
            query="original query",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert - Check message structure in final call
        final_call = self.mock_client.messages.create.call_args_list[2].kwargs
        messages = final_call['messages']

        # Expected structure:
        # [0] user: original query
        # [1] assistant: tool_use (round 1)
        # [2] user: tool_results (round 1)
        # [3] assistant: tool_use (round 2)
        # [4] user: tool_results (round 2)

        self.assertEqual(len(messages), 5)
        self.assertEqual(messages[0]['role'], 'user')  # Original query
        self.assertEqual(messages[1]['role'], 'assistant')  # Round 1 tool use
        self.assertEqual(messages[2]['role'], 'user')  # Round 1 results
        self.assertEqual(messages[3]['role'], 'assistant')  # Round 2 tool use
        self.assertEqual(messages[4]['role'], 'user')  # Round 2 results

        # Verify tool result IDs match
        self.assertEqual(messages[2]['content'][0]['tool_use_id'], 'tool_1')
        self.assertEqual(messages[4]['content'][0]['tool_use_id'], 'tool_2')

    def test_tool_execution_error_handling(self):
        """Test graceful handling of tool execution errors"""
        # Arrange
        generator = self._create_generator_with_mock_client()

        mock_tool_use = MockFixtures.create_anthropic_response_with_tool()
        mock_final = MockFixtures.create_anthropic_final_response(
            "Unable to retrieve complete information due to error."
        )

        self.mock_client.messages.create.side_effect = [
            mock_tool_use,
            mock_final
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Database connection failed")

        tools = [{"name": "search_course_content"}]

        # Act
        result = generator.generate_response(
            query="test",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert - Should complete without crashing
        self.assertIsNotNone(result)
        self.assertEqual(self.mock_client.messages.create.call_count, 2)

        # Verify error was passed to Claude as tool_result
        second_call = self.mock_client.messages.create.call_args_list[1].kwargs
        tool_result = second_call['messages'][2]['content'][0]
        self.assertIn('Error executing tool', tool_result['content'])
        self.assertTrue(tool_result.get('is_error', False))

    def test_no_tool_use_with_sequential_enabled(self):
        """Test that direct answers still work when sequential tool calling is enabled"""
        # Arrange
        generator = self._create_generator_with_mock_client()
        mock_response = MockFixtures.create_anthropic_response_no_tool()
        self.mock_client.messages.create.return_value = mock_response

        mock_tool_manager = Mock()
        tools = [{"name": "search_course_content"}]

        # Act
        result = generator.generate_response(
            query="What is 2+2?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert
        self.assertEqual(result, "This is a direct answer without using tools.")
        self.assertEqual(self.mock_client.messages.create.call_count, 1)
        mock_tool_manager.execute_tool.assert_not_called()


if __name__ == '__main__':
    unittest.main()

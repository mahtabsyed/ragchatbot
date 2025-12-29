import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **search_course_content**: Search for specific course content and detailed educational materials
2. **get_course_outline**: Retrieve course outline showing course title, course link, and all lessons with their numbers and titles

Tool Usage Guidelines:
- Use **search_course_content** for questions about specific course content or detailed educational materials
- Use **get_course_outline** for questions about course structure, lesson list, course overview, or table of contents
- **You may make up to 2 sequential tool calls** if the first tool's results indicate more information is needed
- Use sequential calls when:
  * Initial search returns partial information requiring broader/narrower context
  * Need to check multiple courses or lessons sequentially
  * First tool reveals need for different search parameters
- Synthesize all tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Use search_course_content tool, then answer
- **Course outline/structure questions**: Use get_course_outline tool, then provide the course title, course link, and formatted lesson list
- **Complex queries**: Make initial tool call, evaluate results, make second call if needed to provide complete answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the tool results" or "I made two searches"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls with support for sequential tool calling.
        Supports up to MAX_TOOL_ROUNDS (2) rounds of tool use.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters (includes tools, system, messages)
            tool_manager: Manager to execute tools

        Returns:
            Final response text after all tool executions
        """
        from config import config

        MAX_TOOL_ROUNDS = config.MAX_TOOL_ROUNDS
        current_round = 1  # Initial response counts as round 1

        # Start with existing messages from base_params
        messages = base_params["messages"].copy()

        # Add AI's initial tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        current_response = initial_response

        # Iterative loop: Continue while tools are being used and under round limit
        while current_response.stop_reason == "tool_use" and current_round <= MAX_TOOL_ROUNDS:
            # Execute all tool calls in current response
            tool_results = []
            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    try:
                        tool_result = tool_manager.execute_tool(
                            content_block.name,
                            **content_block.input
                        )

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result
                        })
                    except Exception as e:
                        # Handle tool execution errors gracefully
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Error executing tool: {str(e)}",
                            "is_error": True
                        })

            # Add tool results as user message
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Decide whether to include tools in next API call
            # CRITICAL: Keep tools available if under round limit
            if current_round < MAX_TOOL_ROUNDS:
                # Keep tools available for potential next round
                follow_up_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": base_params["system"],
                    "tools": base_params["tools"],
                    "tool_choice": {"type": "auto"}
                }
            else:
                # Final round: Remove tools to force text response
                follow_up_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": base_params["system"]
                }

            # Make follow-up API call
            current_response = self.client.messages.create(**follow_up_params)

            # If Claude used tools, add assistant response and continue loop
            if current_response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": current_response.content})
                current_round += 1
            # else: Claude provided text response, loop will exit

        # Extract final text response
        # Handle case where response has text content
        for content_block in current_response.content:
            if hasattr(content_block, 'text'):
                return content_block.text

        # Fallback if no text found
        return "Unable to generate response."
"""
Unit tests for CourseSearchTool.execute() method

Tests cover:
- Successful searches with various filter combinations
- Empty results handling
- Error handling
- Source tracking
- Result formatting
"""
import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool
from vector_store import SearchResults, VectorStore
from fixtures import MockFixtures


class TestCourseSearchTool(unittest.TestCase):
    """Test cases for CourseSearchTool.execute() method"""

    def setUp(self):
        """Set up test fixtures before each test"""
        self.mock_vector_store = Mock(spec=VectorStore)
        self.search_tool = CourseSearchTool(self.mock_vector_store)

    def test_search_with_query_only_success(self):
        """Test successful search with query only (no filters)"""
        # Arrange
        mock_results = MockFixtures.create_sample_search_results()
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/lesson1",
            "https://example.com/lesson2",
            "https://example.com/lesson3"
        ]

        # Act
        result = self.search_tool.execute(query="What is MCP?")

        # Assert
        self.assertIn("[MCP: Build Rich-Context AI Apps with Anthropic - Lesson 1]", result)
        self.assertIn("MCP is Model Context Protocol", result)
        self.mock_vector_store.search.assert_called_once_with(
            query="What is MCP?",
            course_name=None,
            lesson_number=None
        )

        # Verify sources tracking
        self.assertEqual(len(self.search_tool.last_sources), 3)
        self.assertEqual(self.search_tool.last_sources[0]['label'],
                        "MCP: Build Rich-Context AI Apps with Anthropic - Lesson 1")
        self.assertEqual(self.search_tool.last_sources[0]['link'],
                        "https://example.com/lesson1")

    def test_search_with_course_filter(self):
        """Test search with course name filter"""
        # Arrange
        mock_results = SearchResults(
            documents=["Content from MCP course"],
            metadata=[{"course_title": "MCP: Build Rich-Context AI Apps with Anthropic", "lesson_number": 1}],
            distances=[0.3]
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        # Act
        result = self.search_tool.execute(
            query="protocols",
            course_name="MCP"
        )

        # Assert
        self.mock_vector_store.search.assert_called_once_with(
            query="protocols",
            course_name="MCP",
            lesson_number=None
        )
        self.assertIn("Content from MCP course", result)

    def test_search_with_lesson_filter(self):
        """Test search with lesson number filter"""
        # Arrange
        mock_results = SearchResults(
            documents=["Lesson 1 content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.2]
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        # Act
        result = self.search_tool.execute(
            query="introduction",
            lesson_number=1
        )

        # Assert
        self.mock_vector_store.search.assert_called_once_with(
            query="introduction",
            course_name=None,
            lesson_number=1
        )
        self.assertIn("[Test Course - Lesson 1]", result)

    def test_search_with_both_filters(self):
        """Test search with both course and lesson filters"""
        # Arrange
        mock_results = SearchResults(
            documents=["Specific lesson content"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 2}],
            distances=[0.1]
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson2"

        # Act
        result = self.search_tool.execute(
            query="API",
            course_name="MCP",
            lesson_number=2
        )

        # Assert
        self.mock_vector_store.search.assert_called_once_with(
            query="API",
            course_name="MCP",
            lesson_number=2
        )
        self.assertIn("Specific lesson content", result)
        self.assertIn("[MCP Course - Lesson 2]", result)

    def test_search_returns_empty_results(self):
        """Test handling of empty search results"""
        # Arrange
        mock_results = MockFixtures.create_empty_search_results()
        self.mock_vector_store.search.return_value = mock_results

        # Act - Test 1: No filters
        result = self.search_tool.execute(query="nonexistent topic")

        # Assert
        self.assertEqual(result, "No relevant content found.")

        # Act - Test 2: With course filter
        result = self.search_tool.execute(
            query="nonexistent topic",
            course_name="Test Course"
        )

        # Assert
        self.assertIn("No relevant content found in course 'Test Course'", result)

        # Act - Test 3: With lesson filter
        result = self.search_tool.execute(
            query="nonexistent topic",
            lesson_number=5
        )

        # Assert
        self.assertIn("No relevant content found in lesson 5", result)

    def test_search_with_error(self):
        """Test handling of search errors"""
        # Arrange
        error_message = "No course found matching 'Invalid Course'"
        mock_results = MockFixtures.create_error_search_results(error_message)
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.search_tool.execute(
            query="test query",
            course_name="Invalid Course"
        )

        # Assert
        self.assertEqual(result, error_message)
        # Verify sources not populated on error
        self.assertEqual(len(self.search_tool.last_sources), 0)

    def test_last_sources_tracking(self):
        """Test that last_sources is correctly populated"""
        # Arrange
        mock_results = SearchResults(
            documents=["Doc 1", "Doc 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": None}
            ],
            distances=[0.1, 0.2]
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/lesson1",
            None  # No link for course without lesson
        ]

        # Act
        result = self.search_tool.execute(query="test")

        # Assert
        self.assertEqual(len(self.search_tool.last_sources), 2)

        # First source (with lesson)
        self.assertEqual(self.search_tool.last_sources[0]['label'], "Course A - Lesson 1")
        self.assertEqual(self.search_tool.last_sources[0]['link'], "https://example.com/lesson1")

        # Second source (no lesson)
        self.assertEqual(self.search_tool.last_sources[1]['label'], "Course B")
        self.assertIsNone(self.search_tool.last_sources[1]['link'])

    def test_format_results_with_lesson_links(self):
        """Test that lesson links are properly attached to sources"""
        # Arrange
        mock_results = SearchResults(
            documents=["Content with link"],
            metadata=[{"course_title": "Test Course", "lesson_number": 3}],
            distances=[0.15]
        )
        lesson_link = "https://learn.deeplearning.ai/lesson3"
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = lesson_link

        # Act
        result = self.search_tool.execute(query="test")

        # Assert
        # Verify get_lesson_link was called with correct parameters
        self.mock_vector_store.get_lesson_link.assert_called_once_with("Test Course", 3)

        # Verify link attached to source
        self.assertEqual(len(self.search_tool.last_sources), 1)
        self.assertEqual(self.search_tool.last_sources[0]['link'], lesson_link)

        # Verify formatting
        self.assertIn("[Test Course - Lesson 3]", result)
        self.assertIn("Content with link", result)


if __name__ == '__main__':
    unittest.main()

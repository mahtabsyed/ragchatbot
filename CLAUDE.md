# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A full-stack RAG (Retrieval-Augmented Generation) system for querying course materials. The system uses vector-based semantic search with ChromaDB and Anthropic's Claude API for intelligent question answering.

**Tech Stack:**
- Backend: Python 3.13, FastAPI, ChromaDB, Anthropic SDK
- Frontend: Vanilla JavaScript, Marked.js for markdown rendering
- Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
- AI Model: Claude Sonnet 4 (claude-sonnet-4-20250514)

## Essential Commands

### Running the Application

```bash
# Quick start (from root)
./run.sh

# Manual start (from root)
cd backend && uv run uvicorn app:app --reload --port 8000

# Access points
# - Web UI: http://localhost:8000
# - API docs: http://localhost:8000/docs
```

### Development Setup

**IMPORTANT: Always use `uv` for package management and running commands. Never use `pip` or `python` directly.**

```bash
# Install dependencies
uv sync

# Create .env file with required API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# Run Python files - ALWAYS use uv run
uv run python <file.py>

# Examples:
uv run python main.py
uv run python backend/app.py
uv run uvicorn app:app --reload

# WRONG - Do NOT do this:
# python script.py  ❌
# pip install package  ❌
```

### Working with ChromaDB

```bash
# Clear and rebuild vector database
# Delete the chroma_db folder and restart the app
rm -rf backend/chroma_db
./run.sh

# The app auto-loads course documents from docs/ on startup
```

## Architecture Overview

### Request Flow: Query Processing

The system uses a **tool-calling architecture** where Claude decides when to search:

```
User Query → FastAPI Endpoint → RAG System → AI Generator
    ↓
AI Generator makes Claude API call with search tool definition
    ↓
Claude decides: Search needed?
    ├─ NO  → Direct answer (1 API call)
    └─ YES → Execute search tool → Return results → Claude synthesizes answer (2 API calls)
```

### Core Components Interaction

**RAG System (`rag_system.py`)** - Central orchestrator
- Coordinates all components
- Manages query flow
- Handles session/conversation context

**AI Generator (`ai_generator.py`)** - Claude API wrapper
- Makes initial API call with tool definitions
- Handles tool execution loop if Claude requests search
- Makes second API call with search results to get final answer

**Search Tools (`search_tools.py`)** - Tool-based search
- Implements `search_course_content` tool that Claude can invoke
- Executes semantic search via Vector Store
- Formats results with course/lesson context
- Tracks sources for UI display

**Vector Store (`vector_store.py`)** - ChromaDB interface
- Two collections: `course_catalog` (metadata) and `course_content` (chunks)
- Handles semantic search with filtering by course name/lesson number
- Smart course name resolution using vector similarity

**Document Processor (`document_processor.py`)** - Parsing and chunking
- Parses structured course files (specific format required)
- Chunks content: 800 chars with 100 char overlap
- Sentence-aware splitting

**Session Manager (`session_manager.py`)** - Conversation history
- Maintains last 2 Q&A exchanges per session
- Context injected into Claude's system prompt

### Data Flow: Document Indexing

```
Startup → Load docs/*.txt files → Document Processor
    ↓
Parse metadata (title, instructor, lessons) + chunk content
    ↓
Vector Store adds to ChromaDB
    ├─ course_catalog: Course metadata with lessons
    └─ course_content: Text chunks with course/lesson metadata
```

### Key Architectural Patterns

**Tool-Based RAG:**
- Claude API receives tool definition for `search_course_content`
- Claude autonomously decides when to search vs. use general knowledge
- Tool parameters: `query` (required), `course_name` (optional), `lesson_number` (optional)
- System prompt in `ai_generator.py` instructs: "One search per query maximum"

**Two-Collection Strategy:**
- `course_catalog`: Enables fuzzy course name matching (e.g., "MCP" matches full course title)
- `course_content`: Stores actual searchable content chunks
- Smart filtering: Resolve course name first, then search content with filters

**Session Management:**
- Each chat creates a session ID
- Last 2 exchanges stored and included in system prompt as "Previous conversation:"
- Enables context-aware follow-up questions

## Course Document Format

Files in `docs/` must follow this exact structure:

```
Course Title: [Title]
Course Link: [URL]
Course Instructor: [Name]

Lesson 0: [Lesson Title]
Lesson Link: [URL]
[Lesson content...]

Lesson 1: [Next Lesson Title]
Lesson Link: [URL]
[Content...]
```

**Adding new courses:** Drop `.txt` files in `docs/` folder and restart app. The startup event (`app.py:88`) auto-processes all `.txt`, `.pdf`, `.docx` files.

## Configuration

All settings in `backend/config.py`:

- **CHUNK_SIZE**: 800 chars (tuned for course content)
- **CHUNK_OVERLAP**: 100 chars (maintains context across chunks)
- **MAX_RESULTS**: 5 (top-k semantic search results)
- **MAX_HISTORY**: 2 (conversation exchanges to remember)
- **CHROMA_PATH**: `./chroma_db` (local persistent storage)
- **ANTHROPIC_MODEL**: `claude-sonnet-4-20250514`
- **EMBEDDING_MODEL**: `all-MiniLM-L6-v2` (used by ChromaDB)

## Important Implementation Details

### ChromaDB Setup
- Uses `PersistentClient` (local file-based storage, not HTTP client)
- `anonymized_telemetry=False` for privacy
- Embeddings computed via SentenceTransformer integration
- Collection IDs: Course titles for catalog, `{title}_{chunk_index}` for content

### Claude API Usage
- Temperature: 0 (deterministic responses)
- Max tokens: 800 (concise answers)
- Tool choice: "auto" (Claude decides when to search)
- System prompt emphasizes: brief, educational, no meta-commentary

### Frontend-Backend Contract
```typescript
// Request
POST /api/query
{
  query: string,
  session_id?: string  // Optional, created if not provided
}

// Response
{
  answer: string,      // Markdown-formatted
  sources: string[],   // ["Course Title - Lesson N", ...]
  session_id: string   // For subsequent requests
}
```

### Error Handling Patterns
- Empty search results → Claude informed, responds with "no information found"
- Course name not found → `SearchResults.empty()` with error message
- Tool execution errors → Caught and returned as tool result string
- API errors → Propagated to FastAPI endpoint → 500 response

## Development Patterns

**When modifying search logic:**
- Update tool definition in `search_tools.py:30-50`
- Update system prompt in `ai_generator.py:7-29` to reflect new capabilities
- Test both with and without filters (course_name, lesson_number)

**When changing chunking strategy:**
- Modify `CHUNK_SIZE` and `CHUNK_OVERLAP` in `config.py`
- Clear ChromaDB: `rm -rf backend/chroma_db`
- Restart app to re-index all documents

**When updating AI behavior:**
- Edit system prompt in `ai_generator.py` (static variable, not rebuilt each call)
- Adjust `temperature` or `max_tokens` in `base_params`
- Test with conversation history to ensure context handling works

**When adding new document types:**
- Extend `add_course_folder()` file extension check in `rag_system.py:81`
- Implement parser in `document_processor.py` for new format
- Maintain `Course` and `CourseChunk` model structure from `models.py`

## Common Gotchas

1. **Always use `uv`, never `pip` or `python` directly** - This project uses `uv` for package management. Run ALL Python files and commands with `uv run` prefix (e.g., `uv run python script.py`, `uv run uvicorn app:app`). Never use `python script.py` or `pip install` directly.
2. **App must run from root directory** - `run.sh` expects `backend/` and `docs/` as siblings
3. **ChromaDB persists data** - Changes to chunking won't reflect unless DB is cleared
4. **Course deduplication** - App skips re-processing courses with same title on restart
5. **Frontend served by FastAPI** - No separate frontend server; FastAPI serves static files
6. **Tool calling is two-step** - Always account for both initial request and result synthesis in Claude API usage
7. **Session history format** - Formatted as single string in system prompt, not as separate messages
8. **Source tracking** - Must call `tool_manager.get_last_sources()` before `reset_sources()` in RAG system

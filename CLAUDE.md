# Catanatron Development Reference

## Build/Test/Lint Commands
- Run all tests: `uv run pytest`
- Run single test: `uv run pytest tests/path/to/test_file.py::test_function_name`
- Play simulation: `uv run catanatron-play --code src/catanatron_experimental/llm_player.py --players=R,R,R,LLM --num=1`
- To run Python code for iterative debugging, run: `uv run python -c`
- Use `uv run pdb` for iterative debugging with Python debugger

## Code Style Guidelines
- Use Python 3.11+ features
- Follow PEP 8 naming conventions (snake_case for variables/functions, PascalCase for classes)
- Include type annotations from `typing` module for function parameters and returns
- Document classes and functions with docstrings
- Use `black` formatter for consistent code style
- Prefer immutable data structures where possible
- Cache expensive operations with `@functools.lru_cache` decorator
- Use comprehensive error handling with descriptive error messages
- Organize imports: standard library first, then third party, then local modules
- Keep lines under 88 characters
- TEST TEST TEST. Write TESTS before implementing new features.

## Code Design Philosophy
- Keep functions small and focused on a single responsibility
- Favor readability and maintainability over premature optimization
- Use clear, descriptive names that reveal intent
- Minimize code complexity by breaking problems into smaller parts
- Write modular code that can be tested in isolation
- Document assumptions and design decisions in comments
- Follow the principle of least surprise with intuitive interfaces
- Create abstractions that improve understanding, not just reduce lines of code
- Prioritize simplicity until profiling shows optimization is needed

## Project Structure
- `src/catanatron`: Core game logic implementation
- `src/catanatron_gym`: OpenAI Gym interface
- `src/catanatron_server`: Web server for game visualization
- `src/catanatron_experimental`: Experimental AI implementations
- `tests/`: Unit and integration tests

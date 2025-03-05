# Catanatron Development Reference

## Build/Test/Lint Commands
- Run all tests: `pytest tests/`
- Run single test: `pytest tests/path/to/test_file.py::test_function_name`
- Run benchmark tests: `pytest --benchmark-compare`
- Run with test coverage: `coverage run --source=catanatron -m pytest tests/ && coverage report`
- Watch tests during development: `ptw --ignore=tests/integration_tests/ --nobeep`
- Play simulation: `catanatron-play --players=R,R,R,W --num=100`
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
- Ensure test coverage for new features

## Project Structure
- `src/catanatron`: Core game logic implementation
- `src/catanatron_gym`: OpenAI Gym interface
- `src/catanatron_server`: Web server for game visualization
- `src/catanatron_experimental`: Experimental AI implementations
- `tests/`: Unit and integration tests

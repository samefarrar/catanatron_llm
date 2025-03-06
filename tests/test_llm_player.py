import io
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

from catanatron.models.player import Color, SimplePlayer
from catanatron.models.enums import Action, ActionType, ActionPrompt, RESOURCES, SETTLEMENT, CITY
from catanatron.game import Game
from catanatron.state import State
from catanatron.state_functions import get_largest_army, player_key
from catanatron_experimental.llm_player import LLMPlayer, RESOURCE_EMOJI

def test_llm_player_initialization():
    """Test that LLMPlayer can be initialized properly"""
    # Save original env var to restore later
    original_api_key = os.environ.get('ANTHROPIC_API_KEY')
    original_model = os.environ.get('ANTHROPIC_MODEL')
    
    try:
        # Test with API key
        os.environ['ANTHROPIC_API_KEY'] = 'test_api_key'
        player = LLMPlayer(Color.RED)
        assert player.color == Color.RED
        assert player.is_bot is True
        assert player.api_key == 'test_api_key'
        assert player.api_calls == 0
        assert player.api_tokens_used == 0
        assert len(player.decision_times) == 0
        
        # Test default model (should be the latest Claude)
        assert player.model == 'claude-3-7-sonnet-20250219'
        
        # Test with custom model
        os.environ['ANTHROPIC_MODEL'] = 'claude-3-opus-20240229'
        player = LLMPlayer(Color.RED)
        assert player.model == 'claude-3-opus-20240229'
        
    finally:
        # Restore original env vars
        if original_api_key:
            os.environ['ANTHROPIC_API_KEY'] = original_api_key
        else:
            os.environ.pop('ANTHROPIC_API_KEY', None)
            
        if original_model:
            os.environ['ANTHROPIC_MODEL'] = original_model
        else:
            os.environ.pop('ANTHROPIC_MODEL', None)


def test_action_description_formatting():
    """Test that action descriptions are properly formatted"""
    player = LLMPlayer(Color.RED)

    # Test roll action description
    roll_action = Action(Color.RED, ActionType.ROLL, None)
    desc = player._get_action_description(roll_action)
    assert desc == "Roll the dice"

    # Test build road action description
    road_action = Action(Color.RED, ActionType.BUILD_ROAD, (0, 1))
    desc = player._get_action_description(road_action)
    assert "Build a road at edge (0, 1)" in desc
    assert "🌲" in desc  # Wood emoji
    assert "🧱" in desc  # Brick emoji

    # Test end turn action description
    end_turn_action = Action(Color.RED, ActionType.END_TURN, None)
    desc = player._get_action_description(end_turn_action)
    assert desc == "End your turn"


def test_cost_formatting():
    """Test that costs are correctly formatted with emojis"""
    player = LLMPlayer(Color.RED)

    road_cost = player._format_cost("ROAD")
    assert "1 🌲WOOD" in road_cost
    assert "1 🧱BRICK" in road_cost

    settlement_cost = player._format_cost("SETTLEMENT")
    assert "1 🌲WOOD" in settlement_cost
    assert "1 🧱BRICK" in settlement_cost
    assert "1 🌾WHEAT" in settlement_cost
    assert "1 🐑SHEEP" in settlement_cost

    city_cost = player._format_cost("CITY")
    assert "2 🌾WHEAT" in city_cost
    assert "3 ⛏️ORE" in city_cost


def test_format_game_state_for_llm():
    """Test that the complete game state is formatted for the LLM correctly"""
    player = LLMPlayer(Color.RED)
    
    # Create a game with a more explicit setup to avoid randomization issues
    red_player = SimplePlayer(Color.RED)
    blue_player = SimplePlayer(Color.BLUE)
    game = Game([red_player, blue_player], initialize=True)
    
    # Set up the test conditions
    game.state.num_turns = 5
    game.state.current_prompt = ActionPrompt.PLAY_TURN
    game.state.colors = (Color.RED, Color.BLUE)  # Force the colors order
    game.state.color_to_index = {Color.RED: 0, Color.BLUE: 1}  # Force the mapping
    game.state.current_player_index = 0  # Make sure RED is current
    
    # Verify our setup
    assert game.state.current_color() == Color.RED
    
    # Create some sample actions
    actions = [
        Action(Color.RED, ActionType.ROLL, None),
        Action(Color.RED, ActionType.END_TURN, None)
    ]
    
    # Call the format method
    formatted_text = player._format_game_state_for_llm(game, game.state, actions)
    
    # Check that the formatted text includes all key sections
    assert "TURN 5" in formatted_text
    assert "PLAY_TURN" in formatted_text
    assert "RED" in formatted_text  # Current player
    
    # Check for main sections
    assert "GAME STATUS:" in formatted_text
    assert "CATAN RESOURCE & NODE GRID:" in formatted_text  # New grid representation
    assert "BUILDINGS:" in formatted_text
    assert "ROADS:" in formatted_text
    assert "PLAYERS:" in formatted_text
    assert "AVAILABLE ACTIONS:" in formatted_text
    
    # Check that actions are included
    assert "Roll the dice" in formatted_text
    assert "End your turn" in formatted_text


def test_format_board_state():
    """Test that board state is formatted correctly as a string"""
    player = LLMPlayer(Color.RED)
    game = Game([SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)])
    
    # Create StringIO to capture the output
    output = io.StringIO()
    
    # Call the format method
    player._format_board_state(output, game.state)
    formatted_text = output.getvalue()
    
    # Check for basic board information in the formatted text
    assert "GAME STATUS:" in formatted_text
    assert "Longest Road:" in formatted_text
    assert "Largest Army:" in formatted_text
    assert "Robber Location:" in formatted_text
    assert "CATAN RESOURCE & NODE GRID:" in formatted_text  # New grid representation
    assert "BUILDINGS:" in formatted_text
    assert "ROADS:" in formatted_text
    assert "PORTS:" in formatted_text  # Updated from "TRADING PORTS"


def test_format_player_states():
    """Test that player states are formatted correctly as a string"""
    player = LLMPlayer(Color.RED)
    game = Game([SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)])
    
    # Setup some player state
    red_key = player_key(game.state, Color.RED)
    blue_key = player_key(game.state, Color.BLUE)
    
    # Add some resources
    game.state.player_state[f"{red_key}_WOOD_IN_HAND"] = 2
    game.state.player_state[f"{red_key}_BRICK_IN_HAND"] = 1
    
    # Add a dev card
    game.state.player_state[f"{red_key}_KNIGHT_IN_HAND"] = 1
    
    # Add settlements/cities info
    game.state.buildings_by_color[Color.RED][SETTLEMENT] = [0, 1]  # Two settlements
    game.state.buildings_by_color[Color.RED][CITY] = [2]  # One city
    
    # Ensure RED is the current player for the test
    game.state.current_player_index = game.state.color_to_index[Color.RED]
    
    # Create StringIO to capture the output
    output = io.StringIO()
    
    # Call the format method
    player._format_player_states(output, game.state)
    formatted_text = output.getvalue()
    
    # Check player info
    assert "RED Player:" in formatted_text
    assert "Victory Points:" in formatted_text
    
    # Check resources (RED is current player by default, so resources should be shown)
    assert "Resources:" in formatted_text
    assert "WOOD" in formatted_text
    assert "BRICK" in formatted_text
    
    # Check buildings
    assert "Buildings:" in formatted_text
    assert "Settlements:" in formatted_text
    assert "Cities:" in formatted_text
    
    # Check bank info
    assert "Bank:" in formatted_text
    assert "Development Cards:" in formatted_text


def test_format_available_actions():
    """Test that available actions are formatted correctly as a string"""
    player = LLMPlayer(Color.RED)
    
    # Create some sample actions
    actions = [
        Action(Color.RED, ActionType.ROLL, None),
        Action(Color.RED, ActionType.BUILD_SETTLEMENT, 3),
        Action(Color.RED, ActionType.END_TURN, None)
    ]
    
    # Create StringIO to capture the output
    output = io.StringIO()
    
    # Call the format method
    player._format_available_actions(output, actions)
    formatted_text = output.getvalue()
    
    # Check general format
    assert "AVAILABLE ACTIONS:" in formatted_text
    
    # Check action descriptions
    assert "Roll" in formatted_text
    assert "Build" in formatted_text
    assert "End" in formatted_text
    
    # Check action numbering
    assert "[0]" in formatted_text
    assert "[1]" in formatted_text
    assert "[2]" in formatted_text
    
    # Check prompt for action selection
    assert "Select action by number" in formatted_text


def test_select_action_fallback():
    """Test the fallback action selection strategy"""
    player = LLMPlayer(Color.RED)
    game = Game([player, SimplePlayer(Color.BLUE)])
    
    # Ensure API key is not set to force fallback
    player.api_key = None
    
    # Create some sample actions
    actions = [
        Action(Color.RED, ActionType.ROLL, None),
        Action(Color.RED, ActionType.END_TURN, None)
    ]
    
    # Test basic action selection with fallback strategy
    chosen_action = player._select_action(actions, game.state)
    assert chosen_action in actions

    # Test selection with specific action types
    # Set up state for PLAY_TURN prompt
    game.state.current_prompt = ActionPrompt.PLAY_TURN
    
    # Only ROLL actions
    roll_actions = [Action(Color.RED, ActionType.ROLL, None)]
    chosen_action = player._select_action(roll_actions, game.state)
    assert chosen_action.action_type == ActionType.ROLL
    
    # Test action priority in fallback strategy
    # Create actions of different priorities
    actions = [
        Action(Color.RED, ActionType.END_TURN, None),
        Action(Color.RED, ActionType.BUILD_CITY, 0),
        Action(Color.RED, ActionType.BUILD_SETTLEMENT, 1),
        Action(Color.RED, ActionType.BUILD_ROAD, (0, 1))
    ]
    
    chosen_action = player._select_action(actions, game.state)
    assert chosen_action.action_type == ActionType.BUILD_CITY  # Highest priority
    
    # Test with only low priority action
    end_turn_actions = [Action(Color.RED, ActionType.END_TURN, None)]
    chosen_action = player._select_action(end_turn_actions, game.state)
    assert chosen_action.action_type == ActionType.END_TURN


@patch('anthropic.Anthropic')
def test_get_claude_decision(mock_anthropic):
    """Test the Claude API integration"""
    # Set up the player with a mock API key
    player = LLMPlayer(Color.RED)
    player.api_key = "test_api_key"
    
    # Create a mock response object
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="3")]  # Claude's response with action number
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 10
    
    # Set up the mock client to return our mock response
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_client
    
    # Test with a simple game state and 5 actions
    game_state_text = "Game state text would go here"
    num_actions = 5
    
    # Call the method
    action_idx = player._get_claude_decision(game_state_text, num_actions)
    
    # Verify the result
    assert action_idx == 3  # Should match the mock response
    
    # Verify API call was made with correct parameters
    mock_client.messages.create.assert_called_once()
    call_args = mock_client.messages.create.call_args[1]
    assert call_args['model'] == player.model
    assert call_args['max_tokens'] == 1500
    assert call_args['temperature'] == 0.1
    assert len(call_args['messages']) == 1
    assert call_args['messages'][0]['role'] == 'user'
    assert game_state_text in call_args['messages'][0]['content']
    
    # Verify stats were tracked
    assert player.api_calls == 1
    assert player.api_tokens_used == 110  # 100 input + 10 output


@patch('anthropic.Anthropic')
def test_get_claude_decision_with_invalid_response(mock_anthropic):
    """Test handling of invalid responses from Claude API"""
    # Set up the player with a mock API key
    player = LLMPlayer(Color.RED)
    player.api_key = "test_api_key"
    
    # Test with various invalid responses
    test_cases = [
        {"response": "I think you should pick option 10", "expected": None},  # Out of range
        {"response": "Let me think about it...", "expected": None},  # No number
        {"response": "Error", "expected": None}  # Generic error
    ]
    
    for i, test_case in enumerate(test_cases):
        # Create a mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=test_case["response"])]
        
        # Set up the mock client
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        # Call the method with 5 actions
        action_idx = player._get_claude_decision("Game state", 5)
        
        # Verify that invalid responses are handled correctly
        assert action_idx == test_case["expected"]
        
        # Verify API call was made
        assert mock_client.messages.create.called


@patch('anthropic.Anthropic')
def test_decide_with_claude_api(mock_anthropic):
    """Test the decide method with Claude API integration"""
    # Set up the player with a mock API key
    player = LLMPlayer(Color.RED)
    player.api_key = "test_api_key"
    
    # Create a game
    game = Game([player, SimplePlayer(Color.BLUE)])
    
    # Create some sample actions
    actions = [
        Action(Color.RED, ActionType.ROLL, None),
        Action(Color.RED, ActionType.END_TURN, None)
    ]
    
    # Set up the mock response to choose action 0
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="0")]
    mock_response.usage.input_tokens = 200
    mock_response.usage.output_tokens = 20
    
    # Set up the mock client
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_client
    
    # Call the decide method
    chosen_action = player.decide(game, actions)
    
    # Verify the result
    assert chosen_action == actions[0]  # Should choose the first action
    
    # Verify API call was made
    assert mock_client.messages.create.called
    
    # Verify stats were tracked
    assert player.api_calls == 1
    assert player.api_tokens_used == 220  # 200 input + 20 output
    assert len(player.decision_times) == 1

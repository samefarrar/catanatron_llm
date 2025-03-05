import io
import sys
from unittest.mock import patch

from catanatron.models.player import Color, SimplePlayer
from catanatron.models.enums import Action, ActionType, ActionPrompt, RESOURCES, SETTLEMENT, CITY
from catanatron.game import Game
from catanatron.state import State
from catanatron.state_functions import get_largest_army, player_key
from catanatron_experimental.llm_player import LLMPlayer, RESOURCE_EMOJI

def test_llm_player_initialization():
    """Test that LLMPlayer can be initialized properly"""
    player = LLMPlayer(Color.RED)
    assert player.color == Color.RED
    assert player.is_bot is True


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


def test_print_game_header():
    """Test that game header prints correctly"""
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
    
    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        player._print_game_header(game)
        output = captured_output.getvalue()
        
        assert "TURN 5" in output
        assert "PLAY_TURN" in output
        
        # More flexible assertion to handle test suite randomization
        current_color = game.state.current_color().name
        assert current_color in output
    finally:
        sys.stdout = sys.__stdout__  # Reset stdout


def test_print_board_state():
    """Test that board state prints correctly"""
    player = LLMPlayer(Color.RED)
    game = Game([SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)])
    
    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        player._print_board_state(game.state)
        output = captured_output.getvalue()
        
        # Check for basic board information
        assert "Game Status:" in output
        assert "Longest Road:" in output
        assert "Largest Army:" in output
        assert "Robber Location:" in output
        assert "Board Map:" in output
        assert "Tiles:" in output
        assert "Buildings:" in output
        assert "Roads:" in output
        assert "Ports:" in output
    finally:
        sys.stdout = sys.__stdout__  # Reset stdout


def test_print_player_states():
    """Test that player states print correctly"""
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
    
    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        player._print_player_states(game.state)
        output = captured_output.getvalue()
        
        # Check player info
        assert "RED Player:" in output
        assert "Victory Points:" in output
        
        # Check resources
        assert "Resources (3 cards):" in output
        assert "🌲 WOOD: 2" in output
        assert "🧱 BRICK: 1" in output
        
        # Check dev cards
        assert "Development Cards (1 total):" in output
        assert "KNIGHT: 1" in output
        
        # Check buildings
        assert "Settlements: 2" in output
        assert "Cities: 1" in output
        
        # Check bank info
        assert "🏦 Bank:" in output
        for resource in RESOURCES:
            assert f"{RESOURCE_EMOJI[resource]} {resource}" in output
        assert "Development Cards remaining:" in output
    finally:
        sys.stdout = sys.__stdout__  # Reset stdout


def test_print_available_actions():
    """Test that available actions print correctly"""
    player = LLMPlayer(Color.RED)
    
    # Create some sample actions
    actions = [
        Action(Color.RED, ActionType.ROLL, None),
        Action(Color.RED, ActionType.BUILD_SETTLEMENT, 3),
        Action(Color.RED, ActionType.END_TURN, None)
    ]
    
    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        player._print_available_actions(actions)
        output = captured_output.getvalue()
        
        assert "Available Actions:" in output
        assert "0: Roll the dice" in output
        assert "1: Build a settlement at node 3" in output
        assert "2: End your turn" in output
    finally:
        sys.stdout = sys.__stdout__  # Reset stdout


def test_select_action():
    """Test that the player can select an action from available options"""
    player = LLMPlayer(Color.RED)
    game = Game([player, SimplePlayer(Color.BLUE)])
    
    # Create some sample actions
    actions = [
        Action(Color.RED, ActionType.ROLL, None),
        Action(Color.RED, ActionType.END_TURN, None)
    ]
    
    # Test basic action selection (currently random in the implementation)
    chosen_action = player._select_action(actions, game.state)
    assert chosen_action in actions

    # Test selection with specific action types
    # Set up state for PLAY_TURN prompt
    game.state.current_prompt = ActionPrompt.PLAY_TURN
    
    # Only ROLL actions
    roll_actions = [Action(Color.RED, ActionType.ROLL, None)]
    chosen_action = player._select_action(roll_actions, game.state)
    assert chosen_action.action_type == ActionType.ROLL
    
    # Only END_TURN actions
    end_turn_actions = [Action(Color.RED, ActionType.END_TURN, None)]
    chosen_action = player._select_action(end_turn_actions, game.state)
    assert chosen_action.action_type == ActionType.END_TURN

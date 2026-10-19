import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dgisim.mode import DefaultMode
from dgisim.state.player_state import PlayerState
random.seed(11)
ps = PlayerState.example_player(DefaultMode())
print(json.dumps({
    "chars": [type(c).__name__ for c in ps.characters],
    "cards": [c.__name__ for c in ps.deck_cards.cards],
}))

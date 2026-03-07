#!/usr/bin/env python3
"""
Gemini baseline for Wordle (Gym/Gymnasium-style env).

Runs N games, reports:
- win rate
- win distribution (# guesses 1..max_steps, plus fails)
- prints a few sample games with emoji boards

Defaults to Vertex AI Gemini 3.1 Pro Preview model id:
  gemini-3.1-pro-preview
Alt:
  gemini-3-flash-preview

Refs:
- Gemini 3.1 Pro Preview model id: `gemini-3.1-pro-preview`  (Vertex AI)  :contentReference[oaicite:1]{index=1}
- Gemini 3 Flash Preview model id: `gemini-3-flash-preview`   (Vertex AI)  :contentReference[oaicite:2]{index=2}
"""

from __future__ import annotations

import argparse
import os
import random
import re
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import gymnasium as gymnasium  # type: ignore
except ImportError:
    gymnasium = None  # type: ignore

try:
    import gym as gym_classic  # type: ignore
except ImportError:
    gym_classic = None  # type: ignore


# Vertex AI SDK
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig
except ImportError as e:
    raise SystemExit(
        "Missing Vertex AI SDK. Install:\n"
        "  pip install google-cloud-aiplatform\n"
        "and ensure you have Application Default Credentials set up."
    ) from e


WORD_RE = re.compile(r"^[a-zA-Z]{5}$")


def make_wordle_env(env_id: str) -> Any:
    """
    Build env robustly across gymnasium/gym + gym_wordle registration styles.
    """
    errors: List[str] = []
    candidates = [env_id]
    if env_id == "Wordle":
        candidates.append("Wordle-v0")

    # Try gymnasium first if available.
    if gymnasium is not None:
        for candidate in candidates:
            try:
                return gymnasium.make(candidate)
            except Exception as e:
                errors.append(f"gymnasium.make({candidate!r}): {e}")

    # Fallback to classic gym (used by gym_wordle 0.1.x).
    if gym_classic is not None:
        try:
            import gym_wordle  # noqa: F401
        except Exception as e:
            errors.append(f"import gym_wordle: {e}")
        for candidate in candidates:
            try:
                return gym_classic.make(candidate)
            except Exception as e:
                errors.append(f"gym.make({candidate!r}): {e}")

    msg = "Failed to create Wordle env.\n" + "\n".join(f"- {x}" for x in errors)
    raise RuntimeError(msg)


@dataclass
class Turn:
    guess: str
    feedback: Sequence[int]  # expected 5 ints (0/1/2)


@dataclass
class Episode:
    secret: Optional[str]
    turns: List[Turn]
    won: bool


def feedback_to_emoji(feedback: Sequence[int]) -> str:
    # 2=green, 1=yellow, 0=gray
    m = {2: "🟩", 1: "🟨", 0: "⬛"}
    return "".join(m.get(int(x), "⬛") for x in feedback)


def normalize_guess(text: str) -> Optional[str]:
    text = text.strip()
    # Some models wrap in quotes or codeblocks; strip common wrappers.
    text = text.strip("`").strip().strip('"').strip("'").strip()
    # Take first token if extra text accidentally appears
    text = text.split()[0] if text else ""
    if WORD_RE.match(text):
        return text.lower()
    return None


def get_allowed_words(env: Any, obs: Any, info: Dict[str, Any]) -> Optional[List[str]]:
    """
    Try to pull an allowed word list from common places.
    If not available, return None and we will just validate by regex + let env reject.
    """
    # Common patterns people use:
    for key in ("allowed_words", "valid_words", "word_list", "dictionary"):
        if key in info and isinstance(info[key], (list, tuple)) and info[key]:
            return [str(w).lower() for w in info[key]]
    if hasattr(env, "allowed_words"):
        aw = getattr(env, "allowed_words")
        if isinstance(aw, (list, tuple)) and aw:
            return [str(w).lower() for w in aw]
    if isinstance(obs, dict):
        for key in ("allowed_words", "valid_words"):
            if key in obs and isinstance(obs[key], (list, tuple)) and obs[key]:
                return [str(w).lower() for w in obs[key]]
    return None


def extract_feedback_from_info(info: Dict[str, Any]) -> Optional[Sequence[int]]:
    """
    Try to extract Wordle feedback (0/1/2 per letter) from env info.
    Supports multiple common conventions:
      - info["feedback"] as list[int]
      - info["result"]   as list[int]
      - info["pattern"]  as "GYBBY" etc.
    """
    for key in ("feedback", "result", "coloring", "colors"):
        if key in info and isinstance(info[key], (list, tuple)) and len(info[key]) == 5:
            return [int(x) for x in info[key]]
    if (
        "pattern" in info
        and isinstance(info["pattern"], str)
        and len(info["pattern"]) == 5
    ):
        s = info["pattern"].upper()
        m = {"G": 2, "Y": 1, "B": 0, "X": 0, "_": 0, ".": 0}
        return [m.get(ch, 0) for ch in s]
    return None


def reset_env(env: Any) -> Tuple[Any, Dict[str, Any]]:
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1] if isinstance(out[1], dict) else {}
    return out, {}


def extract_feedback_from_obs(obs: Any) -> Optional[Sequence[int]]:
    """
    Parse gym_wordle-style board observation (6x10): [chars(5), flags(5)] rows.
    """
    try:
        if obs is None or len(obs) == 0:
            return None
        last_row = None
        for row in obs:
            if int(row[0]) == 0:
                break
            last_row = row
        if last_row is None:
            return None
        flags = last_row[5:10]
        # gym_wordle flags: 1=green, 2=yellow, 3=gray -> convert to 2/1/0
        mapped = []
        for flag in flags:
            value = int(flag)
            if value == 1:
                mapped.append(2)
            elif value == 2:
                mapped.append(1)
            else:
                mapped.append(0)
        return mapped
    except Exception:
        return None


def encode_action(env: Any, guess: str) -> Any:
    """
    Convert guess word to env action where needed.
    gym_wordle expects an integer index into action_space.
    """
    action_space = getattr(env.unwrapped, "action_space", None)
    if action_space is not None and hasattr(action_space, "index_of"):
        try:
            from gym_wordle.utils import to_array  # type: ignore

            idx = action_space.index_of(to_array(guess))
            if idx != -1:
                return int(idx)
        except Exception:
            try:
                idx = action_space.index_of(guess)
                if idx != -1:
                    return int(idx)
            except Exception:
                pass
    return guess


def build_prompt(
    history: List[Turn], max_steps: int, allowed_words: Optional[List[str]]
) -> str:
    """
    Keep the prompt compact to reduce tokens, but still grounded.
    We avoid dumping huge dictionaries. If you want to, you can pass a truncated list.
    """
    lines = []
    lines.append("You are playing Wordle.")
    lines.append(f"Goal: guess the hidden 5-letter word in <= {max_steps} guesses.")
    lines.append(
        "Feedback per letter uses digits: 2=green (correct place), 1=yellow (wrong place), 0=gray (not in word)."
    )
    lines.append("")
    if history:
        lines.append("Previous guesses and feedback:")
        for t in history:
            lines.append(f"- {t.guess} : {''.join(str(int(x)) for x in t.feedback)}")
    else:
        lines.append("No previous guesses yet.")
    lines.append("")
    if allowed_words is not None:
        # Don't dump the entire list; just tell it we will validate.
        lines.append(
            "Return a single 5-letter English word (letters only). I will validate it against the game's allowed list."
        )
    else:
        lines.append("Return a single 5-letter English word (letters only).")
    lines.append("")
    lines.append(
        "IMPORTANT: Output ONLY the next guess (exactly 5 letters). No extra text."
    )

    return "\n".join(lines)


def choose_fallback_word(allowed_words: Optional[List[str]], used: set) -> str:
    """
    If the model outputs invalid text, pick a random unseen allowed word if available,
    else a simple hardcoded starter.
    """
    if allowed_words:
        candidates = [w for w in allowed_words if w not in used]
        if candidates:
            return random.choice(candidates)
        return random.choice(allowed_words)
    # Reasonable starter if no allowed list is accessible:
    for w in ("slate", "crane", "adieu", "roast", "trace"):
        if w not in used:
            return w
    return "slate"


def extract_response_text(resp: Any) -> str:
    """
    Robustly extract text from Vertex response without raising on empty candidates.
    """
    try:
        text = getattr(resp, "text", "")
        if text:
            return str(text)
    except Exception:
        pass

    try:
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []
            chunks: List[str] = []
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    chunks.append(part_text)
            if chunks:
                return "\n".join(chunks)
    except Exception:
        pass

    return ""


def llm_next_guess(
    model: GenerativeModel,
    history: List[Turn],
    max_steps: int,
    allowed_words: Optional[List[str]],
    used: set,
) -> str:
    prompt = build_prompt(history, max_steps=max_steps, allowed_words=allowed_words)
    resp = model.generate_content(
        prompt,
        generation_config=GenerationConfig(
            temperature=0.2,
            max_output_tokens=32,
            candidate_count=1,
        ),
    )
    text = extract_response_text(resp)
    guess = normalize_guess(text)

    if guess is None:
        return choose_fallback_word(allowed_words, used)

    if allowed_words is not None and guess not in set(allowed_words):
        # If model guessed a real word but not in game's dictionary, fallback.
        return choose_fallback_word(allowed_words, used)

    if guess in used:
        return choose_fallback_word(allowed_words, used)

    return guess


def run_episode(env: Any, model: GenerativeModel, max_steps: int) -> Episode:
    obs, info = reset_env(env)
    allowed_words = get_allowed_words(env, obs, info)

    history: List[Turn] = []
    used: set = set()

    secret = None
    # If your env exposes the answer for debugging:
    for k in ("answer", "secret", "target"):
        if k in info and isinstance(info[k], str):
            secret = info[k].lower()

    for _step in range(max_steps):
        guess = llm_next_guess(model, history, max_steps, allowed_words, used)
        used.add(guess)

        # Most Wordle envs take the guess as the action (string).
        # If yours uses an integer action, adapt here.
        action = encode_action(env, guess)
        out = env.step(action)

        # Gymnasium: (obs, reward, terminated, truncated, info)
        # Gym:       (obs, reward, done, info)
        if len(out) == 5:
            obs2, reward, terminated, truncated, info2 = out
            done = bool(terminated or truncated)
        else:
            obs2, reward, done, info2 = out

        fb = extract_feedback_from_info(info2)
        if fb is None:
            fb = extract_feedback_from_obs(obs2)
        if fb is None:
            # If your env doesn't return feedback in info, you should add it.
            # For now, best effort: treat as all-gray.
            fb = [0, 0, 0, 0, 0]

        history.append(Turn(guess=guess, feedback=fb))

        # Winning signal: many envs set done=True and/or reward>0. We also check all-green.
        all_green = all(int(x) == 2 for x in fb)
        if all_green:
            return Episode(secret=secret, turns=history, won=True)

        if done:
            return Episode(secret=secret, turns=history, won=False)

    return Episode(secret=secret, turns=history, won=False)


def summarize(episodes: List[Episode], max_steps: int) -> None:
    n = len(episodes)
    wins = [ep for ep in episodes if ep.won]
    win_rate = len(wins) / n if n else 0.0

    dist = {i: 0 for i in range(1, max_steps + 1)}
    dist["fail"] = 0  # type: ignore[index]
    for ep in episodes:
        if ep.won:
            dist[len(ep.turns)] += 1
        else:
            dist["fail"] += 1  # type: ignore[index]

    lengths = [len(ep.turns) for ep in wins]
    avg_win_len = statistics.mean(lengths) if lengths else float("nan")

    print("\n=== Summary ===")
    print(f"Games: {n}")
    print(f"Wins: {len(wins)}  |  Win rate: {win_rate:.3f}")
    if lengths:
        print(f"Avg guesses (wins only): {avg_win_len:.2f}")

    print("\nWin distribution:")
    for i in range(1, max_steps + 1):
        print(f"  {i}: {dist[i]}")
    print(f"  fail: {dist['fail']}")  # type: ignore[index]


def print_episode(ep: Episode, idx: int) -> None:
    print(f"\n=== Example game #{idx} ===")
    if ep.secret:
        print(f"(secret: {ep.secret})")
    for t in ep.turns:
        emoji = feedback_to_emoji(t.feedback)
        print(f"{t.guess.upper()}  {emoji}")
    print("Result:", "WIN ✅" if ep.won else "LOSE ❌")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id", type=str, default=os.getenv("WORDLE_ENV_ID", "Wordle-v0")
    )
    parser.add_argument(
        "--games", type=int, default=int(os.getenv("WORDLE_N_GAMES", "100"))
    )
    parser.add_argument(
        "--max-steps", type=int, default=int(os.getenv("WORDLE_MAX_STEPS", "6"))
    )
    parser.add_argument(
        "--examples", type=int, default=int(os.getenv("WORDLE_N_EXAMPLES", "3"))
    )

    parser.add_argument(
        "--project", type=str, default=os.getenv("GOOGLE_CLOUD_PROJECT", "")
    )
    parser.add_argument(
        "--location", type=str, default=os.getenv("VERTEX_LOCATION", "global")
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
        help="Vertex model id, e.g. gemini-2.5-flash",
    )
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "0")))
    args = parser.parse_args()

    random.seed(args.seed)

    if not args.project:
        raise SystemExit(
            "Missing GCP project. Set GOOGLE_CLOUD_PROJECT or pass --project.\n"
            "Example:\n"
            "  export GOOGLE_CLOUD_PROJECT=your-project-id\n"
        )

    print("Initializing Vertex AI...")
    vertexai.init(project=args.project, location=args.location)
    model = GenerativeModel(args.model)

    print(f"Making env: {args.env_id}")
    env = make_wordle_env(args.env_id)

    episodes: List[Episode] = []
    for i in range(args.games):
        ep = run_episode(env, model=model, max_steps=args.max_steps)
        episodes.append(ep)
        if (i + 1) % 10 == 0:
            wins_so_far = sum(1 for e in episodes if e.won)
            print(f"[{i+1}/{args.games}] win_rate={wins_so_far/(i+1):.3f}")

    summarize(episodes, max_steps=args.max_steps)

    # Print a few sample games “visually”
    k = min(args.examples, len(episodes))
    for j in range(k):
        print_episode(episodes[j], idx=j + 1)

    env.close()


if __name__ == "__main__":
    main()

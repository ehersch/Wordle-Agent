import numpy as np
import torch
import entropy_ppo
from gym_wordle.utils import to_english, get_words

# Change this list as needed
MODEL_PATHS = [
    "entropy_ppo_best.pt",
    "results_entropy_rewards/entropy_ppo_shaped_best.pt",
]
TOP_K = 25

solution_words = get_words("solution")
matrix = entropy_ppo.build_pattern_matrix(solution_words)
ecalc = entropy_ppo.EntropyCalculator(matrix)
env = entropy_ppo.EntropyWordleWrapper(ecalc)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

crane_idx = next(
    (i for i, w in enumerate(solution_words) if to_english(w) == "crane"), None
)

for model_path in MODEL_PATHS:
    print("\n" + "=" * 80)
    print("MODEL:", model_path)

    model = entropy_ppo.ActorCritic(env.state_dim, env.word_features).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    state = env.reset()
    with torch.no_grad():
        s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        e_t = torch.FloatTensor(env.entropy_scores).unsqueeze(0).to(device)
        logits, _ = model(s_t, e_t)
        scores = logits.squeeze(0).cpu().numpy()

    order = np.argsort(scores)[::-1]

    print(f"Top {TOP_K} first guesses:")
    for rank, idx in enumerate(order[:TOP_K], start=1):
        print(
            f"{rank:>2}. {to_english(solution_words[int(idx)])}   score={scores[int(idx)]:.6f}"
        )

    if crane_idx is not None:
        crane_rank = int(np.where(order == crane_idx)[0][0]) + 1
        print(f"\nCRANE rank: {crane_rank}   score={scores[crane_idx]:.6f}")

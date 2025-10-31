def visualize_explanation(explanation):
    # ANSI color codes
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    # Extract feature names and contributions
    features = [f['feature'] for f in explanation['explanation']]
    contributions = [f['contribution'] for f in explanation['explanation']]
    features = ['bias'] + features
    contributions = [explanation['bias']] + contributions

    max_len = max(len(f) for f in features)
    max_contrib = max(abs(c) for c in contributions)

    print("\nPrediction Breakdown")
    print(f"Prediction Value: {explanation['prediction']:.4f}\n")
    print("Feature".ljust(max_len + 2) + " | Contribution")
    print("-" * (max_len + 20))

    for feature, contrib in zip(features, contributions):
        bar_len = int((abs(contrib) / max_contrib) * 20)
        bar = "█" * bar_len

        if feature == "bias":
            color = YELLOW
        elif contrib >= 0:
            color = GREEN
        else:
            color = RED

        sign = "+" if contrib >= 0 else "-"
        print(f"{feature.ljust(max_len + 2)} | {color}{sign}{bar} {contrib:.4f}{RESET}")
def formula_view(model, feature_names):
    """
    Prints the linear model formula in a readable way.
    """
    terms = [f"{w:.3f}*{f}" for w, f in zip(model.weights, feature_names)]
    formula = " + ".join(terms)
    print("\nModel Formula View:")
    print(f"Prediction = {model.bias:.3f} + {formula}\n")


def terminal_prediction_view(explanation):
    """
    Shows per-prediction contributions in terminal with colored bars.
    """
    GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"

    features = [f['feature'] for f in explanation['explanation']]
    contributions = [f['contribution'] for f in explanation['explanation']]
    features = ['bias'] + features
    contributions = [explanation['bias']] + contributions

    max_len = max(len(f) for f in features)
    max_contrib = max(abs(c) for c in contributions)

    print("\nPrediction Breakdown (Terminal View)")
    print(f"Prediction Value: {explanation['prediction']:.4f}\n")
    print("Feature".ljust(max_len + 2) + " | Contribution")
    print("-" * (max_len + 20))

    for feature, contrib in zip(features, contributions):
        bar_len = int((abs(contrib) / max_contrib) * 20)
        bar = "█" * bar_len
        color = YELLOW if feature == "bias" else (GREEN if contrib >= 0 else RED)
        sign = "+" if contrib >= 0 else "-"
        print(f"{feature.ljust(max_len + 2)} | {color}{sign}{bar} {contrib:.4f}{RESET}")


def aggregated_view(explanations):
    """
    Shows average contributions across multiple predictions.
    """
    from collections import defaultdict
    GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"

    agg = defaultdict(float)
    n = len(explanations)

    for exp in explanations:
        agg['bias'] += exp['bias']
        for f in exp['explanation']:
            agg[f['feature']] += f['contribution']

    print("\nAggregated Feature Contributions (Average)")
    max_len = max(len(f) for f in agg)
    max_val = max(abs(val/n) for val in agg.values())

    for feature, total in agg.items():
        avg = total / n
        bar_len = int((abs(avg) / max_val) * 20)
        bar = "█" * bar_len
        color = YELLOW if feature == "bias" else (GREEN if avg >= 0 else RED)
        sign = "+" if avg >= 0 else "-"
        print(f"{feature.ljust(max_len + 2)} | {color}{sign}{bar} {avg:.4f}{RESET}")

def terminal_prediction_view(explanation):
    GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
    features = [f['feature'] for f in explanation['explanation']]
    contributions = [f['contribution'] for f in explanation['explanation']]
    features = ['bias'] + features
    contributions = [explanation['bias']] + contributions

    max_len = max(len(f) for f in features)
    max_contrib = max(abs(c) for c in contributions) if contributions else 1

    print("\nPrediction Breakdown (Terminal View)")
    print(f"Prediction Value: {explanation['prediction']:.4f}\n")
    print("Feature".ljust(max_len + 2) + " | Contribution")
    print("-" * (max_len + 20))
    for feature, contrib in zip(features, contributions):
        bar_len = int((abs(contrib) / max_contrib) * 20)
        bar = "█" * bar_len
        color = YELLOW if feature == "bias" else (GREEN if contrib >= 0 else RED)
        sign = "+" if contrib >= 0 else "-"
        print(f"{feature.ljust(max_len + 2)} | {color}{sign}{bar} {contrib:.4f}{RESET}")

def aggregated_view(explanations):
    from collections import defaultdict
    GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
    agg = defaultdict(float)
    n = len(explanations)
    for exp in explanations:
        agg['bias'] += exp['bias']
        for f in exp['explanation']:
            agg[f['feature']] += f['contribution']
    print("\nAggregated Feature Contributions (Average)")
    max_len = max(len(f) for f in agg)
    max_val = max(abs(val/n) for val in agg.values()) if n>0 else 1
    for feature, total in agg.items():
        avg = total / n
        bar_len = int((abs(avg) / max_val) * 20)
        bar = "█" * bar_len
        color = YELLOW if feature == "bias" else (GREEN if avg >= 0 else RED)
        sign = "+" if avg >= 0 else "-"
        print(f"{feature.ljust(max_len + 2)} | {color}{sign}{bar} {avg:.4f}{RESET}")

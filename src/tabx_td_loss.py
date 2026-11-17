"""Terminal TD weighting changes the regression objective, not rewards or targets."""


def terminal_td_objective(masked_td_error, mask, terminated, weight, unweighted_loss):
    # Termination comes from the actual trajectory, irrespective of win/loss/source.
    factors = 1. + (weight - 1.) * terminated
    weighted_loss = unweighted_loss if weight == 1. else (
        (masked_td_error.square() * factors).sum() / (mask * factors).sum())
    valid = mask[:, :, :1]
    terminal_steps = float((valid * terminated).detach().sum())
    steps = float(valid.detach().sum())
    weight_sum = steps + (weight - 1.) * terminal_steps
    metrics = dict(weighted_td_loss=float(weighted_loss.detach()), td_terminal_weight=float(weight),
        td_terminal_transitions=int(terminal_steps), td_weight_sum=weight_sum,
        td_terminal_weight_fraction=terminal_steps * weight / weight_sum if weight_sum else 0.)
    return weighted_loss, metrics

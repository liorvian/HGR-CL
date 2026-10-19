"""Explicit, opt-in SRL routing for the P1-E Class-IL ablation."""
MODES = ('none', 'edge', 'memory', 'switch')


def choose_edge(mode, initial_average_degree, threshold):
    if mode not in MODES:
        raise ValueError('Unknown P1-E mode: ' + str(mode))
    if mode == 'none':
        return None
    if mode == 'edge':
        return True
    if mode == 'memory':
        return False
    return initial_average_degree >= threshold

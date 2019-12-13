# 1 in atomic units = _ATOMIC_TO_SI_MAP[dimension] in SI
# TODO use fundamental constants wherever possible
_ATOMIC_TO_SI_MAP = {
    'energy':4.359745e-18,
    'mass': 9.109384e-31,
    'length': 5.291772e-11,
    'charge': 1.602177e-19
}

_ATOMIC_TO_EV = 27.21138


def atomic_to_SI(value, dimension):
    dimension = dimension.lower()
    try:
        return value * _ATOMIC_TO_SI_MAP[dimension]
    except KeyError:
        raise ValueError(f"dimension {dimension} not implemented for unit mappings")
    
def SI_to_atomic(value, dimension):
    dimension = dimension.lower()
    try:
        return value / _ATOMIC_TO_SI_MAP[dimension]
    except KeyError:
        raise ValueError(f"dimension {dimension} not implemented for unit mappings")
        
def atomic_to_ev(value):
    return value * _ATOMIC_TO_EV

def ev_to_atomic(value):
    return value / _ATOMIC_TO_EV

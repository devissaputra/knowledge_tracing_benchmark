from dataclasses import dataclass

@dataclass(frozen=True)
class BKTParams:
    p_init: float = 0.2
    p_learn: float = 0.15
    p_guess: float = 0.2
    p_slip: float = 0.1

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

def predict_correct_probability(p_mastery: float, params: BKTParams | None=None) -> float:
    if not 0.0 <= p_mastery <= 1.0:
        raise ValueError("p_mastery must be between 0 and 1")
    params=params or BKTParams()
    return p_mastery*(1-params.p_slip)+(1-p_mastery)*params.p_guess

def update_mastery(p_mastery: float, correct: bool, params: BKTParams | None=None) -> float:
    if not 0.0 <= p_mastery <= 1.0:
        raise ValueError("p_mastery must be between 0 and 1")
    params=params or BKTParams()
    if correct:
        numerator=p_mastery*(1-params.p_slip)
        denominator=numerator+(1-p_mastery)*params.p_guess
    else:
        numerator=p_mastery*params.p_slip
        denominator=numerator+(1-p_mastery)*(1-params.p_guess)
    posterior=numerator/denominator if denominator else p_mastery
    return posterior+(1-posterior)*params.p_learn

def trace(sequence, params: BKTParams | None=None) -> list[float]:
    params=params or BKTParams()
    mastery=params.p_init
    output=[]
    for response in sequence:
        if response not in (0,1,False,True):
            raise ValueError("sequence must contain binary responses")
        mastery=update_mastery(mastery,bool(response),params)
        output.append(mastery)
    return output

def predictive_trace(sequence, params: BKTParams | None=None) -> list[float]:
    """Return P(correct) immediately before each observed response."""
    params=params or BKTParams()
    mastery=params.p_init
    probabilities=[]
    for response in sequence:
        if response not in (0,1,False,True):
            raise ValueError("sequence must contain binary responses")
        probabilities.append(predict_correct_probability(mastery,params))
        mastery=update_mastery(mastery,bool(response),params)
    return probabilities

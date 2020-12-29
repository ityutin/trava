import pytest  # noqa

from trava.fit_predictor import FitPredictorSteps


def test_addition(mocker):
    def gen_steps(n: int):
        return [mocker.Mock() for _ in range(n)]

    first_raw_model_steps = gen_steps(n=2)
    second_raw_model_steps = gen_steps(n=1)

    first_config_steps = gen_steps(n=5)
    second_config_steps = gen_steps(n=0)

    first_final_steps = gen_steps(n=1)
    second_final_steps = gen_steps(n=1)

    first = FitPredictorSteps(
        raw_model_steps=first_raw_model_steps, config_steps=first_config_steps, final_steps=first_final_steps
    )
    second = FitPredictorSteps(
        raw_model_steps=second_raw_model_steps, config_steps=second_config_steps, final_steps=second_final_steps
    )

    result = first + second

    assert result.raw_model_steps == (first_raw_model_steps + second_raw_model_steps)
    assert result.config_steps == (first_config_steps + second_config_steps)
    assert result.final_steps == (first_final_steps + second_final_steps)

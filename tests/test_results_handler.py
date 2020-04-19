import pytest

from tests.objects_for_tests import TestResultsHandler


def test_scorers_copy(mocker):
    scorers = [mocker.Mock(), mocker.Mock()]
    initial_scorers = scorers.copy()
    handler = TestResultsHandler(scorers=scorers)

    scorers.append(mocker.Mock())

    assert handler.scorers == initial_scorers

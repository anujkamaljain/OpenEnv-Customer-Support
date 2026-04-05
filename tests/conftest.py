import pytest

from env.environment import CustomerSupportEnv
from tasks.ticket_bank import TicketBank


@pytest.fixture
def bank() -> TicketBank:
    return TicketBank()


@pytest.fixture
def env(bank: TicketBank) -> CustomerSupportEnv:
    return CustomerSupportEnv(ticket_bank=bank)

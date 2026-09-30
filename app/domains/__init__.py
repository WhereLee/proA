from .base import Domain
from .data_analysis import DATA_ANALYSIS
from .research import RESEARCH
from .tickets import TICKETS

DOMAINS: dict[str, Domain] = {
    domain.name: domain for domain in (TICKETS, DATA_ANALYSIS, RESEARCH)
}


def get_domain(name: str) -> Domain:
    if name not in DOMAINS:
        raise KeyError(f"unknown domain: {name}")
    return DOMAINS[name]


__all__ = ["Domain", "TICKETS", "DATA_ANALYSIS", "RESEARCH", "DOMAINS", "get_domain"]

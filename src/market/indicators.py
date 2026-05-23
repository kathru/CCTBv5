"""
Indicadores técnicos compartilhados — fonte única de verdade.

Antes desta refatoração, _atr(), _bollinger() e _pearson() estavam
duplicados em 7+ arquivos (volatility_state.py, meta_regime.py,
advanced_risk.py, feature_analysis.py, recalibrate.py, walk_forward.py,
stress_backtest.py). Qualquer mudança precisava ser feita em múltiplos
lugares, garantindo divergências futuras.

Agora todos importam daqui. Funções puras, sem estado, sem I/O.
"""

import math


def atr(highs: list[float], lows: list[float], closes: list[float],
        period: int = 14) -> float:
    """Average True Range sobre `period` períodos. closes[0] = mais recente."""
    n = min(period, len(highs) - 1)
    if n <= 0:
        return (highs[0] - lows[0]) if highs else 0.0
    trs = [
        max(highs[i] - lows[i],
            abs(highs[i] - closes[i + 1]),
            abs(lows[i]  - closes[i + 1]))
        for i in range(n)
    ]
    return sum(trs) / len(trs) if trs else 0.0


def bollinger(closes: list[float],
              period: int = 20) -> tuple[float, float, float]:
    """
    Bollinger Bands (upper, middle, lower) com 2σ.
    closes[0] = mais recente. Retorna (upper, mid, lower).
    """
    n = min(period, len(closes))
    if n < 2:
        c = closes[0]
        return c, c, c
    window = closes[:n]
    mid = sum(window) / n
    std = math.sqrt(sum((x - mid) ** 2 for x in window) / n)
    return mid + 2 * std, mid, mid - 2 * std


def pearson(xs: list[float], ys: list[float]) -> float:
    """
    Correlação de Pearson entre duas séries de igual comprimento.
    Usa min(len(xs), len(ys)) pontos. Retorna 0.0 se dados insuficientes.
    """
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    sx  = sum(xs[:n])
    sy  = sum(ys[:n])
    sxy = sum(xs[i] * ys[i] for i in range(n))
    sx2 = sum(x * x for x in xs[:n])
    sy2 = sum(y * y for y in ys[:n])
    num = n * sxy - sx * sy
    den = math.sqrt(max(0.0, (n * sx2 - sx * sx) * (n * sy2 - sy * sy)))
    return num / den if den > 0 else 0.0


def returns_1h(closes: list[float], horizon: int = 24) -> list[float]:
    """
    Série de retornos simples 1H (mais recente primeiro).
    closes[0] = mais recente. Retorna até `horizon` retornos.
    """
    n = min(horizon, len(closes) - 1)
    return [
        (closes[i] - closes[i + 1]) / closes[i + 1]
        for i in range(n)
        if closes[i + 1] > 0
    ]


def euclidean(a: list[float], b: list[float]) -> float:
    """Distância Euclidiana entre dois vetores de mesmo comprimento."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=False)))


def sigmoid(x: float, k: float = 20.0) -> float:
    """Sigmoid centrada em 0 mapeando R → (0, 1)."""
    return 1.0 / (1.0 + math.exp(-k * x))

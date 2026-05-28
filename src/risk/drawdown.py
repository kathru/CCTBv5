"""
Drawdown control — daily, peak e trend de equity.

Três camadas:
  1. Daily drawdown  — reseta todo dia (MAX_DAILY_DD = 3%)
  2. Peak drawdown   — do all-time high
  3. Equity trend    — slope linear sobre janela 30d de equity diário
                       detecta aceleração real em vez de 10 amostras a 15s

Equity diário: persistido como lista de (date, value) para sobreviver a restarts.
Acceleration: slope normalizado da regressão linear sobre últimos N dias.
  slope negativo > threshold → drawdown acelerando
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

logger = logging.getLogger(__name__)

EQUITY_WINDOW_DAYS  = 30    # janela para slope de equity
MIN_DAYS_FOR_SLOPE  = 5     # mínimo de dias para calcular slope


def _today() -> date:
    return datetime.now(UTC).date()


def _linear_slope(values: list[float]) -> float:
    """
    Slope da regressão linear simples (normalizado pelo valor médio).
    Positivo = equity subindo. Negativo = equity caindo.
    Retorna slope diário como fração do valor médio.
    """
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    if y_mean <= 0:
        return 0.0
    num = sum((xs[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    if den <= 0:
        return 0.0
    return (num / den) / y_mean   # slope diário normalizado


@dataclass
class DrawdownState:
    # Daily tracking
    day_start_value: float = 0.0
    day_date: date = field(default_factory=_today)

    # Peak tracking
    peak_value: float = 0.0

    # Janela de equity diário — (date_isoformat, value)
    daily_equity: list[tuple[str, float]] = field(default_factory=list)

    # Último valor visto (para cálculos intra-day)
    last_value: float = 0.0


class DrawdownEngine:
    """
    Tracks daily, peak e trend de equity.
    Emite sinais de bloqueio quando limites são violados.
    """

    def __init__(
        self,
        max_daily_dd: float = 0.03,         # 3% daily limit
        max_peak_dd: float = 0.10,          # 10% from peak
        acceleration_threshold: float = -0.005,  # slope < -0.5%/dia = aceleração
    ) -> None:
        self._max_daily_dd     = max_daily_dd
        self._max_peak_dd      = max_peak_dd
        self._accel_threshold  = acceleration_threshold
        self._state            = DrawdownState()

    def update(self, current_value: float) -> None:
        """Chamado a cada ciclo com o valor atual do portfolio."""
        today = _today()
        state = self._state

        # Inicialização
        if state.day_start_value == 0.0:
            state.day_start_value = current_value
        if state.peak_value == 0.0:
            state.peak_value = current_value

        # Reset diário
        if state.day_date != today:
            self._record_daily_equity(state.day_date, state.last_value or current_value)
            state.day_start_value = current_value
            state.day_date = today

        # Atualiza peak
        if current_value > state.peak_value:
            state.peak_value = current_value

        state.last_value = current_value

    def _record_daily_equity(self, d: date, value: float) -> None:
        """Persiste snapshot de equity no fim do dia (janela 30d)."""
        if value <= 0:
            return
        state = self._state
        state.daily_equity.append((d.isoformat(), value))
        # Mantém apenas os últimos EQUITY_WINDOW_DAYS dias
        if len(state.daily_equity) > EQUITY_WINDOW_DAYS:
            state.daily_equity = state.daily_equity[-EQUITY_WINDOW_DAYS:]

    def restore_daily_equity(self, snapshots: list[tuple[str, float]]) -> None:
        """
        Restaura histórico de equity de um armazenamento externo (Redis/DB).
        Chamado no boot para reconstruir a janela 30d.
        """
        self._state.daily_equity = list(snapshots)[-EQUITY_WINDOW_DAYS:]
        if self._state.daily_equity:
            last_val = self._state.daily_equity[-1][1]
            if self._state.peak_value < last_val:
                self._state.peak_value = last_val

    @property
    def daily_drawdown(self) -> float:
        """Drawdown diário atual como fração positiva (0.03 = 3% queda)."""
        state = self._state
        if state.day_start_value <= 0 or state.last_value <= 0:
            return 0.0
        dd = (state.day_start_value - state.last_value) / state.day_start_value
        return max(0.0, dd)

    @property
    def peak_drawdown(self) -> float:
        """Drawdown do all-time peak como fração positiva."""
        state = self._state
        if state.peak_value <= 0 or state.last_value <= 0:
            return 0.0
        dd = (state.peak_value - state.last_value) / state.peak_value
        return max(0.0, dd)

    @property
    def equity_slope(self) -> float:
        """
        Slope diário da equity normalizado pelo valor médio.
        Calculado via regressão linear sobre janela 30d.
        Positivo = crescendo. Negativo = caindo.
        """
        daily = self._state.daily_equity
        if len(daily) < MIN_DAYS_FOR_SLOPE:
            return 0.0
        values = [v for _, v in daily]
        return _linear_slope(values)

    @property
    def acceleration(self) -> float:
        """
        Aceleração do drawdown: equity_slope negativo = acelerando negativamente.
        Para compatibilidade com código existente: positivo = piorando.
        """
        return -self.equity_slope   # invertido: positivo = queda

    def is_daily_limit_breached(self) -> tuple[bool, str]:
        dd = self.daily_drawdown
        if dd >= self._max_daily_dd:
            return True, (
                f"Daily drawdown limit breached: {dd:.2%} >= {self._max_daily_dd:.2%}"
            )
        return False, ""

    def is_peak_limit_breached(self) -> tuple[bool, str]:
        dd = self.peak_drawdown
        if dd >= self._max_peak_dd:
            return True, (
                f"Peak drawdown limit breached: {dd:.2%} >= {self._max_peak_dd:.2%}"
            )
        return False, ""

    def is_accelerating(self) -> tuple[bool, str]:
        slope = self.equity_slope
        if slope <= self._accel_threshold:
            return True, (
                f"Equity declining: slope={slope:.3%}/day "
                f"(threshold={self._accel_threshold:.3%})"
            )
        return False, ""

    def get_diagnostics(self) -> dict:
        """Retorna breakdown para debugging e dashboard."""
        return {
            "daily_dd":      round(self.daily_drawdown, 4),
            "peak_dd":       round(self.peak_drawdown, 4),
            "equity_slope":  round(self.equity_slope, 5),
            "acceleration":  round(self.acceleration, 5),
            "n_daily_obs":   len(self._state.daily_equity),
            "peak_value":    round(self._state.peak_value, 2),
            "day_start":     round(self._state.day_start_value, 2),
            "last_value":    round(self._state.last_value, 2),
        }

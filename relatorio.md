# Relatório de Simulação de Trading - CCTBv5
## Período: 1° de Janeiro de 2026 a 30 de Abril de 2026

Este relatório apresenta os resultados da simulação realizada com o robô CCTBv5, utilizando a estratégia `MomentumStrategy` nos ativos BTC-USDT, ETH-USDT e SOL-USDT.

### 1. Resumo Executivo

| Métrica | Valor |
| :--- | :--- |
| **Portfólio Inicial** | R$ 5.000,00 |
| **Portfólio Final** | R$ 4.475,05 |
| **PnL Total** | R$ -524,95 (-10,50%) |
| **Win Rate Geral** | 23,8% |
| **Profit Factor** | 0.32 |
| **Total de Trades** | 1145 |

### 2. Resultados por Ativo

| Ativo | Trades | Win Rate | PnL | Profit Factor |
| :--- | :---: | :---: | :---: | :---: |
| **BTC-USDT** | 388 | 20,1% | R$ -177,14 | 0,24 |
| **ETH-USDT** | 384 | 25,0% | R$ -172,65 | 0,35 |
| **SOL-USDT** | 373 | 26,3% | R$ -175,17 | 0,37 |

### 3. Análise de Performance

A simulação para o primeiro quadrimestre de 2026 apresentou um cenário desafiador para a `MomentumStrategy`. Abaixo, os principais pontos observados:

*   **Alta Frequência de Trades:** O robô executou mais de 1100 trades em 4 meses (média de ~9 trades por dia). Isso indica uma alta sensibilidade aos sinais de mercado, possivelmente capturando ruído excessivo em prazos menores (30m).
*   **Baixa Taxa de Acerto (Win Rate):** O Win Rate consolidado de 23,8% está significativamente abaixo do esperado para uma estratégia robusta. O BTC foi o ativo com pior desempenho neste quesito (20,1%).
*   **Impacto de Taxas e Slippage:** Com um volume tão alto de operações, os custos de transação (fees e slippage) drenaram uma parte considerável do capital. Como visto nos logs, as taxas totais foram significativas para cada ativo.
*   **Profit Factor Crítico:** Um Profit Factor de 0.32 indica que para cada R$ 1,00 perdido, a estratégia recuperou apenas R$ 0,32. Isso sugere que a estratégia não conseguiu "deixar o lucro correr" ou que os stops foram atingidos muito frequentemente antes de qualquer reversão favorável.

### 4. Recomendações

Com base nos dados obtidos, recomenda-se:

1.  **Refino de Thresholds:** Os thresholds de entrada por regime precisam ser revisitados. No período simulado, muitos sinais foram gerados com baixa probabilidade de sucesso, sugerindo que o calibrador de Platt pode precisar de novos dados ou de uma penalização maior para regimes de "CHOP" e "EXHAUSTION".
2.  **Redução da Frequência:** Considerar o aumento da seletividade dos sinais (ex: exigir um `expected_value` mínimo maior) ou migrar para tempos gráficos ligeiramente maiores (ex: 1H puro) para filtrar o ruído que causou o alto número de trades perdedores.
3.  **Ajuste de Stop Loss e Take Profit:** A análise dos motivos de saída (não detalhada aqui, mas sugerida pelo baixo PF) pode revelar que o Stop Loss está muito curto para a volatilidade do período ou o Take Profit muito ambicioso.
4.  **Avaliação do Regime de Mercado:** O período de Jan-Abr 2026 pode ter sido marcado por uma lateralidade persistente (Chop) ou micro-tendências sem seguimento, o que é o "veneno" para estratégias de momentum puro.

---
*Relatório gerado automaticamente pelo Jules em 20 de Maio de 2026.*

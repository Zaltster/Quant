// frontend/components/BotDashboard.tsx
"use client";

import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import styles from './BotDashboard.module.css';

interface Position {
    id: string;
    ticker: string;
    entryPrice: number;
    quantity: number;
    leverage: number;
    entryStep: number;
}

interface Trade {
    id: string;
    step: number;
    ticker: string;
    action: 'BUY' | 'SELL';
    quantity: number;
    price: number;
    leverage: number;
    realizedPnl?: number;
}

interface BotDashboardProps {
    botCash: number;
    botPositions: Position[];
    botPortfolioValue: number;
    botTradeHistory: Trade[];
    userPortfolioValue: number;
    currentStep: number;
    initialCash: number;
    isGameOver: boolean;
    selectedTicker: string;
}

export default function BotDashboard({
    botCash,
    botPositions,
    botPortfolioValue,
    botTradeHistory,
    userPortfolioValue,
    currentStep,
    initialCash,
    isGameOver,
    selectedTicker
}: BotDashboardProps) {
    // Calculate bot's holdings summary
    const botHoldingsSummary: Record<string, number> = {};
    botPositions.forEach(pos => {
        botHoldingsSummary[pos.ticker] = (botHoldingsSummary[pos.ticker] || 0) + pos.quantity;
    });

    // Create data for performance comparison chart
    const performanceData = Array(currentStep + 1).fill(null).map((_, index) => {
        const botTradesUpToStep = botTradeHistory.filter(t => t.step <= index);

        // Find the most recent portfolio values for this step (if we have historical data)
        // In a real implementation, you'd track historical values at each step
        const botValue = index === currentStep ? botPortfolioValue : initialCash;
        const userValue = index === currentStep ? userPortfolioValue : initialCash;

        return {
            step: index + 1,
            botValue,
            userValue
        };
    });

    // Calculate ROI percentages
    const botROI = ((botPortfolioValue - initialCash) / initialCash * 100).toFixed(2);
    const userROI = ((userPortfolioValue - initialCash) / initialCash * 100).toFixed(2);

    // Get bot's recent trades for the selected ticker
    const recentBotTrades = botTradeHistory
        .filter(trade => trade.ticker === selectedTicker)
        .slice(-5); // Show only the 5 most recent trades

    return (
        <div className={styles.botDashboard}>
            <div className={styles.header}>
                <h3>Trading Bot Dashboard</h3>
                {isGameOver && <span className={styles.gameOver}>GAME OVER</span>}
            </div>

            <div className={styles.performanceSection}>
                <h4>Performance Comparison</h4>
                <div className={styles.metrics}>
                    <div className={styles.metric}>
                        <span>Bot Portfolio:</span>
                        <span className={botPortfolioValue >= initialCash ? styles.positive : styles.negative}>
                            ${botPortfolioValue.toFixed(2)}
                        </span>
                    </div>
                    <div className={styles.metric}>
                        <span>Your Portfolio:</span>
                        <span className={userPortfolioValue >= initialCash ? styles.positive : styles.negative}>
                            ${userPortfolioValue.toFixed(2)}
                        </span>
                    </div>
                    <div className={styles.metric}>
                        <span>Bot ROI:</span>
                        <span className={parseFloat(botROI) >= 0 ? styles.positive : styles.negative}>
                            {botROI}%
                        </span>
                    </div>
                    <div className={styles.metric}>
                        <span>Your ROI:</span>
                        <span className={parseFloat(userROI) >= 0 ? styles.positive : styles.negative}>
                            {userROI}%
                        </span>
                    </div>
                </div>

                <div className={styles.chartContainer}>
                    <ResponsiveContainer width="100%" height={250}>
                        <LineChart data={performanceData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="step" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Line
                                type="monotone"
                                dataKey="botValue"
                                stroke="#8884d8"
                                name="Bot Portfolio"
                                dot={false}
                                activeDot={{ r: 8 }}
                            />
                            <Line
                                type="monotone"
                                dataKey="userValue"
                                stroke="#82ca9d"
                                name="Your Portfolio"
                                dot={false}
                                activeDot={{ r: 8 }}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <div className={styles.botPortfolioSection}>
                <h4>Bot Portfolio</h4>
                <div className={styles.portfolioDetails}>
                    <div className={styles.portfolioMetric}>
                        <span>Cash:</span>
                        <span>${botCash.toFixed(2)}</span>
                    </div>
                    <div className={styles.portfolioMetric}>
                        <span>Equity:</span>
                        <span>${botPortfolioValue.toFixed(2)}</span>
                    </div>
                </div>

                <h5>Bot Holdings</h5>
                {Object.keys(botHoldingsSummary).length > 0 ? (
                    <div className={styles.holdingsTable}>
                        {Object.entries(botHoldingsSummary).map(([ticker, quantity]) => (
                            <div key={ticker} className={styles.holdingRow}>
                                <span>{ticker}:</span>
                                <span>{quantity.toFixed(4)} shares</span>
                            </div>
                        ))}
                    </div>
                ) : (
                    <p className={styles.noHoldings}>No current holdings</p>
                )}
            </div>

            <div className={styles.tradeHistorySection}>
                <h4>Recent Bot Trades for {selectedTicker}</h4>
                {recentBotTrades.length > 0 ? (
                    <div className={styles.tradesTable}>
                        {recentBotTrades.map((trade) => (
                            <div key={trade.id} className={`${styles.tradeRow} ${trade.action === 'BUY' ? styles.buyTrade : styles.sellTrade}`}>
                                <div className={styles.tradeInfo}>
                                    <span className={styles.tradeStep}>Step {trade.step + 1}</span>
                                    <span className={styles.tradeAction}>{trade.action}</span>
                                </div>
                                <div className={styles.tradeDetails}>
                                    <span>{trade.quantity.toFixed(4)} shares @ ${trade.price.toFixed(2)}</span>
                                    {trade.action === 'SELL' && trade.realizedPnl !== undefined && (
                                        <span className={trade.realizedPnl >= 0 ? styles.profit : styles.loss}>
                                            P&L: ${trade.realizedPnl.toFixed(2)}
                                        </span>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <p className={styles.noTrades}>No trades for {selectedTicker} yet</p>
                )}
            </div>

            <div className={styles.botStrategySection}>
                <h4>Bot Strategy</h4>
                <p>
                    This trading bot uses a Deep Q-Network (DQN) with LSTM and
                    Transformer components to make trading decisions based on
                    price patterns and market conditions.
                </p>
            </div>
        </div>
    );
}
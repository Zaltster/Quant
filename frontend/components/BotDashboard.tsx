// frontend/components/BotDashboard.tsx
"use client";

import React, { useState, useEffect } from 'react';
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
    // State to track which view of trades to show
    const [showAllTrades, setShowAllTrades] = useState(true);

    // State to store historical portfolio values
    const [portfolioHistory, setPortfolioHistory] = useState<{ step: number, botValue: number, userValue: number }[]>([]);

    // Calculate bot's holdings summary
    const botHoldingsSummary: Record<string, number> = {};
    botPositions.forEach(pos => {
        botHoldingsSummary[pos.ticker] = (botHoldingsSummary[pos.ticker] || 0) + pos.quantity;
    });

    // Update portfolio history when currentStep changes
    useEffect(() => {
        setPortfolioHistory(prev => {
            // Create a copy of the previous history
            const newHistory = [...prev];

            // Add the current step's values if this step doesn't exist yet
            if (!newHistory.find(item => item.step === currentStep)) {
                newHistory.push({
                    step: currentStep,
                    botValue: botPortfolioValue,
                    userValue: userPortfolioValue
                });
            }

            // Sort by step
            return newHistory.sort((a, b) => a.step - b.step);
        });
    }, [currentStep, botPortfolioValue, userPortfolioValue]);

    // Create data for performance comparison chart
    const performanceData = portfolioHistory.length > 0
        ? portfolioHistory.map(item => ({
            step: item.step + 1, // Add 1 for display (0-indexed to 1-indexed)
            botValue: item.botValue,
            userValue: item.userValue
        }))
        : Array(currentStep + 1).fill(null).map((_, index) => ({
            step: index + 1,
            botValue: index === currentStep ? botPortfolioValue : initialCash,
            userValue: index === currentStep ? userPortfolioValue : initialCash
        }));

    // Calculate ROI percentages
    const botROI = ((botPortfolioValue - initialCash) / initialCash * 100).toFixed(2);
    const userROI = ((userPortfolioValue - initialCash) / initialCash * 100).toFixed(2);

    // Get bot's recent trades
    const tickerTrades = botTradeHistory
        .filter(trade => trade.ticker === selectedTicker)
        .sort((a, b) => b.step - a.step) // Sort by most recent first
        .slice(0, 5); // Show only the 5 most recent trades

    // Get all recent trades across all tickers
    const allRecentTrades = botTradeHistory
        .sort((a, b) => b.step - a.step) // Sort by most recent first
        .slice(0, 10); // Show only the 10 most recent trades

    // Total trading activity stats
    const totalBuyTrades = botTradeHistory.filter(t => t.action === 'BUY').length;
    const totalSellTrades = botTradeHistory.filter(t => t.action === 'SELL').length;
    const totalTrades = botTradeHistory.length;

    // Realized P&L from all completed trades
    const totalRealizedPnl = botTradeHistory
        .filter(t => t.action === 'SELL' && t.realizedPnl !== undefined)
        .reduce((sum, trade) => sum + (trade.realizedPnl || 0), 0);

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
                    <div className={styles.portfolioMetric}>
                        <span>Total Trades:</span>
                        <span>{totalTrades} ({totalBuyTrades} buys, {totalSellTrades} sells)</span>
                    </div>
                    <div className={styles.portfolioMetric}>
                        <span>Realized P&L:</span>
                        <span className={totalRealizedPnl >= 0 ? styles.positive : styles.negative}>
                            ${totalRealizedPnl.toFixed(2)}
                        </span>
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
                <div className={styles.tradeHistoryHeader}>
                    <h4>Bot Trade History</h4>
                    <div className={styles.tradeViewToggle}>
                        <button
                            className={`${styles.toggleButton} ${showAllTrades ? styles.activeToggle : ''}`}
                            onClick={() => setShowAllTrades(true)}
                        >
                            All Tickers
                        </button>
                        <button
                            className={`${styles.toggleButton} ${!showAllTrades ? styles.activeToggle : ''}`}
                            onClick={() => setShowAllTrades(false)}
                        >
                            {selectedTicker} Only
                        </button>
                    </div>
                </div>

                {(showAllTrades ? allRecentTrades : tickerTrades).length > 0 ? (
                    <div className={styles.tradesTable}>
                        {(showAllTrades ? allRecentTrades : tickerTrades).map((trade) => (
                            <div key={trade.id} className={`${styles.tradeRow} ${trade.action === 'BUY' ? styles.buyTrade : styles.sellTrade}`}>
                                <div className={styles.tradeInfo}>
                                    <span className={styles.tradeStep}>Step {trade.step + 1}</span>
                                    <span className={styles.tradeTicker}>{trade.ticker}</span>
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
                    <p className={styles.noTrades}>
                        {showAllTrades ? 'No trades recorded yet' : `No trades for ${selectedTicker} yet`}
                    </p>
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
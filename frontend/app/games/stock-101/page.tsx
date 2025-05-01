// frontend/app/games/stock-101/page.tsx
"use client";

import React, { useState, useEffect, useCallback, useRef } from 'react';
import StockSelectorPanel from '../../../components/StockSelectorPanel';
import MainChartDisplay from '../../../components/MainChartDisplay';
import PortfolioDisplay from '../../../components/PortfolioDisplay';
import BuyPanel from '../../../components/BuyPanel';
import SellPanel from '../../../components/SellPanel';
import BotDashboard from '../../../components/BotDashboard';
import TabNavigation from '../../../components/TabNavigation';
import { v4 as uuidv4 } from 'uuid';
import styles from './Stock101Page.module.css';

// --- Interfaces ---
interface StockData {
    [ticker: string]: number[];
}

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

interface BotDecision {
    action: number; // 0: HOLD, 1: BUY, 2: SELL
    action_name: string;
    step: number;
    price: number;
}

// --- Component ---
const BACKEND_URL = 'http://127.0.0.1:8000';
const BOT_API_URL = 'http://127.0.0.1:8001';
const GAME_DURATION_SECONDS = 600;
const DATA_POINTS_EXPECTED = 100;
const UPDATE_INTERVAL_MS = (GAME_DURATION_SECONDS / DATA_POINTS_EXPECTED) * 1000;
const initialCash = 100000;

export default function Stock101Page() {
    // --- State ---
    const [gameData, setGameData] = useState<StockData | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [gameStarted, setGameStarted] = useState<boolean>(false);
    const [currentStep, setCurrentStep] = useState<number>(0);
    const [isGameOver, setIsGameOver] = useState<boolean>(false);
    const [selectedTicker, setSelectedTicker] = useState<string>('');
    const [cash, setCash] = useState<number>(initialCash);
    const [positions, setPositions] = useState<Position[]>([]);
    const [portfolioValue, setPortfolioValue] = useState<number>(initialCash);
    const [leverage, setLeverage] = useState<number>(1);
    const [tradeMessage, setTradeMessage] = useState<{ type: 'error' | 'success' | 'info', text: string } | null>(null);
    const [liquidatedMessage, setLiquidatedMessage] = useState<string | null>(null);
    const [tradeHistory, setTradeHistory] = useState<Trade[]>([]);

    // --- Bot Related State ---
    const [botCash, setBotCash] = useState<number>(initialCash);
    const [botPositions, setBotPositions] = useState<Position[]>([]);
    const [botPortfolioValue, setBotPortfolioValue] = useState<number>(initialCash);
    const [botTradeHistory, setBotTradeHistory] = useState<Trade[]>([]);
    const [botDecisions, setBotDecisions] = useState<BotDecision[]>([]);
    const [activeTab, setActiveTab] = useState<string>('user');
    const [botMessage, setBotMessage] = useState<string | null>(null);
    const [botConnected, setBotConnected] = useState<boolean>(false);

    // WebSocket reference
    const wsRef = useRef<WebSocket | null>(null);

    // --- Helper Functions ---
    const getCurrentPrice = useCallback((ticker: string): number | null => {
        if (!gameData || !gameData[ticker] || gameData[ticker].length <= currentStep) {
            return null;
        }
        const price = gameData[ticker][currentStep];
        return (typeof price === 'number' && price >= 0 && !isNaN(price)) ? price : null;
    }, [gameData, currentStep]);

    // --- Bot Trade Handlers ---
    const handleBotBuy = useCallback((ticker: string, quantity: number, tradeLeverage: number) => {
        // FIXED: Removed the Number.isInteger check to allow fractional quantities
        if (isGameOver || !gameData || quantity <= 0) {
            console.log(`Bot buy rejected: isGameOver=${isGameOver}, quantity=${quantity}`);
            return;
        }

        const currentPrice = getCurrentPrice(ticker);
        if (currentPrice === null || currentPrice <= 0) {
            console.log(`Bot buy rejected: invalid price=${currentPrice}`);
            return;
        }

        const cost = quantity * currentPrice;
        const marginRequired = cost / tradeLeverage;

        console.log(`Bot attempting buy: ${ticker}, quantity=${quantity}, price=${currentPrice}, cost=${cost}, marginRequired=${marginRequired}, botCash=${botCash}`);

        if (botCash >= marginRequired) {
            const newPosition: Position = {
                id: uuidv4(),
                ticker: ticker,
                entryPrice: currentPrice,
                quantity: quantity,
                leverage: tradeLeverage,
                entryStep: currentStep
            };

            const newTrade: Trade = {
                id: uuidv4(),
                step: currentStep,
                ticker: ticker,
                action: 'BUY',
                quantity: quantity,
                price: currentPrice,
                leverage: tradeLeverage
            };

            setBotTradeHistory(prev => [...prev, newTrade]);
            setBotCash(prevCash => prevCash - marginRequired);
            setBotPositions(prevPositions => [...prevPositions, newPosition]);
            setBotMessage(`Bot bought ${quantity} ${ticker} @ ${currentPrice.toFixed(2)}`);
            console.log(`Bot BUY executed: ${quantity} ${ticker} @ ${currentPrice.toFixed(2)}`);
        } else {
            console.log(`Bot buy failed: Insufficient cash. Need ${marginRequired}, have ${botCash}`);
        }
    }, [botCash, currentStep, gameData, isGameOver, getCurrentPrice]);

    const handleBotSell = useCallback((ticker: string, quantityToSell: number) => {
        if (isGameOver || !gameData || quantityToSell <= 0) {
            console.log(`Bot sell rejected: isGameOver=${isGameOver}, quantity=${quantityToSell}`);
            return false;
        }

        const currentPrice = getCurrentPrice(ticker);
        const effectiveSellPrice = currentPrice ?? 0;

        if (effectiveSellPrice < 0 || currentPrice === null) {
            console.log(`Bot sell rejected: invalid price=${currentPrice}`);
            return false;
        }

        let remainingToSell = quantityToSell;
        let totalRealizedPnl = 0;
        let totalMarginFreed = 0;
        let actualQuantitySold = 0;
        let leverageOfSoldShares = 1;
        const updatedPositions: Position[] = [];
        const positionsForTicker = botPositions.filter(p => p.ticker === ticker).sort((a, b) => a.entryStep - b.entryStep);
        const otherPositions = botPositions.filter(p => p.ticker !== ticker);
        const totalSharesHeld = positionsForTicker.reduce((sum, p) => sum + p.quantity, 0);
        const tolerance = 1e-9;

        console.log(`Bot attempting sell: ${ticker}, quantity=${quantityToSell}, price=${effectiveSellPrice}, totalSharesHeld=${totalSharesHeld}`);

        if (totalSharesHeld < quantityToSell - tolerance) {
            console.log(`Bot sell failed: Not enough shares. Have ${totalSharesHeld}, tried to sell ${quantityToSell}`);
            return false;
        }

        for (const pos of positionsForTicker) {
            if (remainingToSell <= tolerance) {
                updatedPositions.push(pos);
                continue;
            }

            const sellQtyFromThisPos = Math.min(remainingToSell, pos.quantity);
            const costBasisForSold = sellQtyFromThisPos * pos.entryPrice;
            const marginFreed = costBasisForSold / pos.leverage;
            const realizedPnl = (effectiveSellPrice - pos.entryPrice) * sellQtyFromThisPos * pos.leverage;

            totalRealizedPnl += realizedPnl;
            totalMarginFreed += marginFreed;
            remainingToSell -= sellQtyFromThisPos;
            actualQuantitySold += sellQtyFromThisPos;
            leverageOfSoldShares = pos.leverage;

            if (pos.quantity - sellQtyFromThisPos > tolerance) {
                updatedPositions.push({
                    ...pos,
                    quantity: pos.quantity - sellQtyFromThisPos
                });
            }
        }

        if (actualQuantitySold > tolerance) {
            setBotPositions([...otherPositions, ...updatedPositions]);
            setBotCash(prevCash => prevCash + totalMarginFreed + totalRealizedPnl);

            const newTrade: Trade = {
                id: uuidv4(),
                step: currentStep,
                ticker: ticker,
                action: 'SELL',
                quantity: actualQuantitySold,
                price: effectiveSellPrice,
                leverage: leverageOfSoldShares,
                realizedPnl: totalRealizedPnl
            };

            setBotTradeHistory(prev => [...prev, newTrade]);
            setBotMessage(`Bot sold ${actualQuantitySold.toFixed(4)} ${ticker} @ ${effectiveSellPrice.toFixed(2)}`);
            console.log(`Bot SELL executed: ${actualQuantitySold.toFixed(4)} ${ticker} @ ${effectiveSellPrice.toFixed(2)}, P&L: $${totalRealizedPnl.toFixed(2)}`);
            return true;
        }

        console.log(`Bot sell failed: Sell quantity too small or error`);
        return false;
    }, [botPositions, currentStep, gameData, isGameOver, getCurrentPrice]);

    // --- WebSocket Connection for Trading Bot ---
    const connectToTradingBot = useCallback(() => {
        if (!gameStarted || !selectedTicker) return;

        // Close existing connection if any
        if (wsRef.current) {
            wsRef.current.close();
        }

        // Create new WebSocket connection
        const ws = new WebSocket(`ws://127.0.0.1:8001/ws/trading_bot`);

        ws.onopen = () => {
            console.log('Connected to trading bot at step:', currentStep);
            setBotConnected(true);
            setBotMessage('Trading bot connected');
        };

        ws.onclose = () => {
            console.log('Disconnected from trading bot');
            setBotConnected(false);
            setBotMessage('Trading bot disconnected');
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            setBotConnected(false);
            setBotMessage('Error connecting to trading bot');
        };

        ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);

                if (message.type === "AGENT_DECISION") {
                    // Store bot decision
                    setBotDecisions(prev => [...prev, message.data]);

                    // Execute bot trade based on decision
                    const { action, action_name, price, step } = message.data;
                    const currentPrice = getCurrentPrice(selectedTicker);

                    console.log(`Bot decision at step ${step}: ${action_name} (${action}) at $${price}`);

                    if (currentPrice !== null && step === currentStep) {
                        // Modified logic: 
                        // 1. Use original model decision if it's BUY or SELL
                        // 2. Randomly override HOLD decisions ~30% of the time
                        // Use only the model's actual decisions with no overrides
                        let effectiveAction = action;
                        console.log(`Bot decision: ${action === 0 ? 'HOLD' : action === 1 ? 'BUY' : 'SELL'} at price $${currentPrice}`);

                        // Execute based on the effective action
                        if (effectiveAction === 1) { // BUY
                            // Use 20% of available cash
                            const cashToUse = botCash * 0.2;
                            let quantity;

                            // For high-priced assets like BTC, allow fractional purchases
                            if (selectedTicker === 'BTC-USD' || currentPrice > 1000) {
                                // Calculate how many units we can buy with the cash, allowing fractional units
                                quantity = parseFloat((cashToUse / currentPrice).toFixed(4));
                                // Ensure minimum quantity is 0.001 for high-price assets
                                if (quantity < 0.001 && botCash >= currentPrice * 0.001) {
                                    quantity = 0.001;
                                }
                            } else {
                                // For lower-priced assets, use whole units
                                quantity = Math.floor(cashToUse / currentPrice);
                                // Ensure at least 1 unit if we have enough cash
                                if (quantity === 0 && botCash >= currentPrice) {
                                    quantity = 1;
                                }
                            }

                            console.log(`Bot BUY calculation: Cash: $${botCash.toFixed(2)}, Using: $${cashToUse.toFixed(2)}, Price: $${currentPrice.toFixed(2)}, Quantity: ${quantity}`);

                            if (quantity > 0) {
                                handleBotBuy(selectedTicker, quantity, 1);
                            }
                        } else if (effectiveAction === 2) { // SELL
                            // Only sell if we have shares
                            const tickerPositions = botPositions.filter(p => p.ticker === selectedTicker);
                            const totalShares = tickerPositions.reduce((sum, pos) => sum + pos.quantity, 0);

                            console.log(`Bot SELL calculation: Total shares held: ${totalShares}`);

                            if (totalShares > 0) {
                                // Sell half the position instead of all
                                const sellQuantity = totalShares * 0.5;
                                handleBotSell(selectedTicker, sellQuantity);
                            }
                        }
                        // For original HOLD (action === 0), do nothing
                    }
                } else if (message.type === 'AGENT_ERROR') {
                    console.error('Agent error:', message.data.message);
                    setBotMessage(`Bot error: ${message.data.message}`);
                } else if (message.type === 'PERFORMANCE_SUMMARY') {
                    console.log('Performance summary:', message.data);
                    // Handle performance summary if needed
                }
            } catch (error) {
                console.error('Error processing WebSocket message:', error);
            }
        };

        wsRef.current = ws;

        return () => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
        };
    }, [gameStarted, selectedTicker, currentStep, botCash, botPositions, getCurrentPrice, handleBotBuy, handleBotSell]);

    // Send price updates to the trading bot
    const sendPriceUpdateToBot = useCallback(() => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN || !gameStarted || !selectedTicker) return;

        const currentPrice = getCurrentPrice(selectedTicker);
        if (currentPrice === null) return;

        wsRef.current.send(JSON.stringify({
            type: 'PRICE_UPDATE',
            data: {
                ticker: selectedTicker,
                price: currentPrice,
                step: currentStep,
                agent_balance: botCash,
                agent_shares: botPositions
                    .filter(p => p.ticker === selectedTicker)
                    .reduce((sum, pos) => sum + pos.quantity, 0)
            }
        }));
    }, [gameStarted, selectedTicker, currentStep, botCash, botPositions, getCurrentPrice]);

    // --- Game Control ---
    const startGame = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        setGameData(null);
        setGameStarted(false);
        setCurrentStep(0);
        setIsGameOver(false);
        setSelectedTicker('');
        setCash(initialCash);
        setPositions([]);
        setPortfolioValue(initialCash);
        setLeverage(1);
        setTradeMessage(null);
        setLiquidatedMessage(null);
        setTradeHistory([]);

        // Reset bot state
        setBotCash(initialCash);
        setBotPositions([]);
        setBotPortfolioValue(initialCash);
        setBotTradeHistory([]);
        setBotDecisions([]);
        setBotMessage(null);
        setBotConnected(false);

        // Close existing WebSocket connection
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }

        try {
            const response = await fetch(`${BACKEND_URL}/game_simulation_data`);
            if (!response.ok) {
                const errTxt = await response.text();
                throw new Error(`API Error: ${response.status} - ${errTxt}`);
            }

            const data: { status: string; stock_data?: StockData; detail?: string } = await response.json();

            if (data.status === 'success' && data.stock_data) {
                const tickers = Object.keys(data.stock_data);
                if (tickers.length === 0 || Object.values(data.stock_data).some(arr => !Array.isArray(arr) || arr.length !== DATA_POINTS_EXPECTED)) {
                    throw new Error(`Invalid data received.`);
                }

                setGameData(data.stock_data);
                const firstTicker = tickers[0];
                if (firstTicker) {
                    setSelectedTicker(firstTicker);
                }
                setGameStarted(true);

                // Check if bot API is available
                try {
                    const botStatusResponse = await fetch(`${BOT_API_URL}/agent_status`);
                    if (botStatusResponse.ok) {
                        const botStatus = await botStatusResponse.json();
                        if (botStatus.agent_loaded) {
                            setBotMessage('Trading bot ready');
                        } else {
                            setBotMessage('Trading bot not loaded. Check server.');
                        }
                    }
                } catch (botError) {
                    console.error('Error checking bot status:', botError);
                    setBotMessage('Trading bot server not available');
                }
            } else {
                throw new Error(data.detail || 'Backend error.');
            }
        } catch (err) {
            if (err instanceof Error) {
                setError(err.message);
            } else {
                setError("Unknown fetch error.");
            }
            console.error("Fetch error:", err);
            setGameStarted(false);
        }
        finally {
            setIsLoading(false);
        }
    }, []);

    // --- UI Callbacks ---
    const handleSelectTicker = (ticker: string) => {
        setSelectedTicker(ticker);
    };

    const handleSetLeverage = (newLeverage: number) => {
        setLeverage(newLeverage);
    };

    const handleTabChange = (tab: string) => {
        setActiveTab(tab);
    };

    // --- Trade Execution ---
    const handleBuy = useCallback((ticker: string, quantity: number, tradeLeverage: number) => {
        setTradeMessage(null);
        setLiquidatedMessage(null);

        if (isGameOver || !gameData || quantity <= 0 || !Number.isInteger(quantity)) {
            setTradeMessage({
                type: 'error',
                text: quantity <= 0 ? "Quantity must be positive integer." : "Game not active."
            });
            return;
        }

        const currentPrice = getCurrentPrice(ticker);
        if (currentPrice === null || currentPrice <= 0) {
            setTradeMessage({ type: 'error', text: "Invalid current price for buy." });
            return;
        }

        const cost = quantity * currentPrice;
        const marginRequired = cost / tradeLeverage;

        if (cash >= marginRequired) {
            const newPosition: Position = {
                id: uuidv4(),
                ticker: ticker,
                entryPrice: currentPrice,
                quantity: quantity,
                leverage: tradeLeverage,
                entryStep: currentStep
            };

            const newTrade: Trade = {
                id: uuidv4(),
                step: currentStep,
                ticker: ticker,
                action: 'BUY',
                quantity: quantity,
                price: currentPrice,
                leverage: tradeLeverage
            };

            setTradeHistory(prev => [...prev, newTrade]);
            setCash(prevCash => prevCash - marginRequired);
            setPositions(prevPositions => [...prevPositions, newPosition]);
            setTradeMessage({
                type: 'success',
                text: `Bought ${quantity} ${ticker} @ ${currentPrice.toFixed(2)} (Lev: ${tradeLeverage}x, Margin: $${marginRequired.toFixed(2)})`
            });
        } else {
            setTradeMessage({
                type: 'error',
                text: `Insufficient cash. Need margin $${marginRequired.toFixed(2)}, have $${cash.toFixed(2)}`
            });
        }
    }, [cash, currentStep, gameData, isGameOver, getCurrentPrice]);

    const handleSell = useCallback((ticker: string, quantityToSell: number, isLiquidation: boolean = false) => {
        if (!isLiquidation) {
            setTradeMessage(null);
            setLiquidatedMessage(null);
        }

        if ((isGameOver && !isLiquidation) || !gameData || quantityToSell <= 0) {
            if (!isLiquidation)
                setTradeMessage({ type: 'error', text: "Game not active or invalid quantity." });
            return false;
        }

        const currentPrice = getCurrentPrice(ticker);
        const effectiveSellPrice = currentPrice ?? 0;

        if ((effectiveSellPrice < 0 || currentPrice === null) && !isLiquidation) {
            if (!isLiquidation)
                setTradeMessage({ type: 'error', text: "Invalid price for sell." });
            return false;
        }

        let remainingToSell = quantityToSell;
        let totalRealizedPnl = 0;
        let totalMarginFreed = 0;
        let actualQuantitySold = 0;
        let leverageOfSoldShares = 1;
        const updatedPositions: Position[] = [];
        const positionsForTicker = positions.filter(p => p.ticker === ticker).sort((a, b) => a.entryStep - b.entryStep);
        const otherPositions = positions.filter(p => p.ticker !== ticker);
        const totalSharesHeld = positionsForTicker.reduce((sum, p) => sum + p.quantity, 0);
        const tolerance = 1e-9;

        if (totalSharesHeld < quantityToSell - tolerance) {
            if (!isLiquidation)
                setTradeMessage({
                    type: 'error',
                    text: `Not enough shares. Have ${totalSharesHeld.toFixed(4)}, tried ${quantityToSell}`
                });
            return false;
        }

        for (const pos of positionsForTicker) {
            if (remainingToSell <= tolerance) {
                updatedPositions.push(pos);
                continue;
            }

            const sellQtyFromThisPos = Math.min(remainingToSell, pos.quantity);
            const costBasisForSold = sellQtyFromThisPos * pos.entryPrice;
            const marginFreed = costBasisForSold / pos.leverage;
            const realizedPnl = (effectiveSellPrice - pos.entryPrice) * sellQtyFromThisPos * pos.leverage;

            totalRealizedPnl += realizedPnl;
            totalMarginFreed += marginFreed;
            remainingToSell -= sellQtyFromThisPos;
            actualQuantitySold += sellQtyFromThisPos;
            leverageOfSoldShares = pos.leverage;

            if (pos.quantity - sellQtyFromThisPos > tolerance) {
                updatedPositions.push({
                    ...pos,
                    quantity: pos.quantity - sellQtyFromThisPos
                });
            }
        }

        if (actualQuantitySold > tolerance) {
            setPositions([...otherPositions, ...updatedPositions]);
            setCash(prevCash => prevCash + totalMarginFreed + totalRealizedPnl);

            const newTrade: Trade = {
                id: uuidv4(),
                step: currentStep,
                ticker: ticker,
                action: 'SELL',
                quantity: actualQuantitySold,
                price: effectiveSellPrice,
                leverage: leverageOfSoldShares,
                realizedPnl: totalRealizedPnl
            };

            setTradeHistory(prev => [...prev, newTrade]);

            if (!isLiquidation) {
                setTradeMessage({
                    type: 'success',
                    text: `Sold ${actualQuantitySold.toFixed(4)} ${ticker} @ ${effectiveSellPrice.toFixed(2)}. P&L: $${totalRealizedPnl.toFixed(2)}`
                });
            }

            return true;
        } else {
            if (!isLiquidation)
                setTradeMessage({
                    type: 'error',
                    text: `Sell quantity too small or error.`
                });

            return false;
        }
    }, [positions, currentStep, gameData, isGameOver, getCurrentPrice]);

    // --- useEffect Hooks ---
    // Connect to trading bot when game starts
    useEffect(() => {
        if (gameStarted && selectedTicker) {
            connectToTradingBot();
        }

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, [gameStarted, selectedTicker, connectToTradingBot]);

    // Send price updates to the bot on step changes
    useEffect(() => {
        if (gameStarted && botConnected) {
            sendPriceUpdateToBot();
        }
    }, [gameStarted, botConnected, currentStep, selectedTicker, sendPriceUpdateToBot]);

    // Recalculate User Equity & Check Liquidations Effect
    useEffect(() => {
        if (!gameStarted || !gameData || !positions) return;

        let calculatedEquity = cash;
        let positionsToKeep: Position[] = [];
        let liquidatedPositionsThisStep: Position[] = [];

        positions.forEach(pos => {
            const currentPrice = getCurrentPrice(pos.ticker);
            let keepPosition = true;

            if (currentPrice === null) {
                calculatedEquity += (pos.entryPrice * pos.quantity) / pos.leverage;
            } else {
                const requiredMargin = (pos.entryPrice * pos.quantity) / pos.leverage;
                const priceRatio = pos.entryPrice > 1e-9 ? (currentPrice / pos.entryPrice) : 0;
                const lossPercentLeveraged = (priceRatio - 1) * pos.leverage;
                const liquidationThreshold = -1.0 + 1e-9;

                if ((lossPercentLeveraged <= liquidationThreshold && pos.leverage > 1) || currentPrice <= 1e-9) {
                    liquidatedPositionsThisStep.push(pos);
                    keepPosition = false;
                } else {
                    const unrealizedPnl = (currentPrice - pos.entryPrice) * pos.quantity * pos.leverage;
                    calculatedEquity += requiredMargin + unrealizedPnl;
                }
            }

            if (keepPosition) {
                positionsToKeep.push(pos);
            }
        });

        setPortfolioValue(calculatedEquity);

        if (liquidatedPositionsThisStep.length > 0) {
            let totalLiquidationPnl = 0;
            let anySellFailed = false;

            liquidatedPositionsThisStep.forEach(pos => {
                const liquidationPrice = getCurrentPrice(pos.ticker) ?? 0;
                const pnl = (liquidationPrice - pos.entryPrice) * pos.quantity * pos.leverage;
                totalLiquidationPnl += pnl;
                const success = handleSell(pos.ticker, pos.quantity, true);
                if (!success) anySellFailed = true;
            });

            setPositions(currentPositions =>
                currentPositions.filter(p => !liquidatedPositionsThisStep.some(lp => lp.id === p.id))
            );

            setLiquidatedMessage(`${liquidatedPositionsThisStep.length} position(s) liquidated! Realized P&L: $${totalLiquidationPnl.toFixed(2)}`);

            if (anySellFailed) console.error("One or more liquidating sells reported failure.");
        } else {
            setLiquidatedMessage(null);
        }
    }, [currentStep, cash, positions, gameData, gameStarted, getCurrentPrice, handleSell]);

    // Recalculate Bot Equity
    // In the "Recalculate Bot Equity" useEffect in page.tsx
    useEffect(() => {
        if (!gameStarted || !gameData || !botPositions) return;

        let calculatedEquity = botCash;
        console.log('Bot portfolio calculation:');
        console.log(`- Initial cash: $${botCash.toFixed(2)}`);

        botPositions.forEach(pos => {
            const currentPrice = getCurrentPrice(pos.ticker);

            if (currentPrice === null) {
                const positionValue = (pos.entryPrice * pos.quantity) / pos.leverage;
                calculatedEquity += positionValue;
                console.log(`- ${pos.ticker}: ${pos.quantity} shares, value: $${positionValue.toFixed(2)} (using entry price)`);
            } else {
                const requiredMargin = (pos.entryPrice * pos.quantity) / pos.leverage;
                const unrealizedPnl = (currentPrice - pos.entryPrice) * pos.quantity * pos.leverage;
                calculatedEquity += requiredMargin + unrealizedPnl;
                console.log(`- ${pos.ticker}: ${pos.quantity} shares, margin: $${requiredMargin.toFixed(2)}, unrealized P&L: $${unrealizedPnl.toFixed(2)}`);
            }
        });

        console.log(`Total bot portfolio value: $${calculatedEquity.toFixed(2)}`);
        setBotPortfolioValue(calculatedEquity);
    }, [currentStep, botCash, botPositions, gameData, gameStarted, getCurrentPrice]);

    // Bankruptcy Check Effect for User
    useEffect(() => {
        if (gameStarted && !isGameOver && portfolioValue <= 0 && positions.length === 0) {
            console.error("BANKRUPT!");
            setIsGameOver(true);
            setLiquidatedMessage(null);
            setTradeMessage({
                type: 'error',
                text: `BANKRUPT! Equity reached ${portfolioValue.toFixed(2)}`
            });
        }
    }, [portfolioValue, positions, gameStarted, isGameOver]);

    // Game Loop Timer Effect
    useEffect(() => {
        if (!gameStarted || !gameData || isGameOver) return;

        const intervalId = setInterval(() => setCurrentStep(prev => prev + 1), UPDATE_INTERVAL_MS);

        if (currentStep >= DATA_POINTS_EXPECTED - 1) {
            clearInterval(intervalId);
            if (!isGameOver) {
                setIsGameOver(true);
                console.log("Game ended: Final step.");

                // Send game completion message to bot
                if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                    wsRef.current.send(JSON.stringify({
                        type: "SIMULATION_COMPLETE",
                        data: {
                            agent_portfolio_value: botPortfolioValue,
                            user_portfolio_value: portfolioValue,
                            initial_balance: initialCash
                        }
                    }));
                }
            }
        }

        return () => clearInterval(intervalId);
    }, [gameStarted, gameData, isGameOver, currentStep, botPortfolioValue, portfolioValue]);

    // --- Prepare Render Data ---
    const availableTickers = gameData ? Object.keys(gameData) : [];
    const currentSelectedPrice = selectedTicker ? getCurrentPrice(selectedTicker) : null;

    // Filter trade history for the selected ticker to pass to the chart
    const tradesForSelectedTicker = tradeHistory.filter(t => t.ticker === selectedTicker);

    // Define tabs
    const tabs = [
        { id: 'user', label: 'Your Trading' },
        { id: 'bot', label: 'Bot Trading' }
    ];

    // --- Render ---
    return (
        <div className={styles.pageContainer}>
            <h1 className={styles.pageTitle}>Stock 101 Trading Simulation</h1>

            {!gameStarted && (
                <button
                    onClick={startGame}
                    disabled={isLoading}
                    className={styles.startButton}
                >
                    {isLoading ? 'Loading...' : 'Start Interview Session'}
                </button>
            )}

            {isLoading && <p className={styles.loadingMessage}>Loading...</p>}
            {error && <p className={styles.errorMessage}>Error: {error}</p>}

            {gameStarted && gameData && (
                <>
                    {/* Tab Navigation */}
                    <TabNavigation
                        activeTab={activeTab}
                        onTabChange={handleTabChange}
                        tabs={tabs}
                    />

                    {activeTab === 'user' ? (
                        <div className={styles.gameLayout}>
                            {/* --- Main Content Column --- */}
                            <div className={styles.mainContent}>
                                <div className={styles.gameStatus}>
                                    <span>Step: {currentStep + 1} / {DATA_POINTS_EXPECTED}</span>
                                    {isGameOver && <span className={styles.gameOver}>GAME OVER</span>}
                                </div>
                                {(liquidatedMessage || tradeMessage) && (
                                    <div className={`${styles.tradeMessage} ${liquidatedMessage ? styles.error : (tradeMessage?.type === 'error' ? styles.error : styles.success)}`}>
                                        {liquidatedMessage || tradeMessage?.text}
                                    </div>
                                )}

                                {selectedTicker && gameData[selectedTicker] ? (
                                    <MainChartDisplay
                                        key={selectedTicker}
                                        ticker={selectedTicker}
                                        priceHistory={gameData[selectedTicker]}
                                        currentStep={currentStep}
                                        trades={tradesForSelectedTicker}
                                    />
                                ) : availableTickers.length > 0 ? (<p>Select a stock.</p>) : null}

                                {selectedTicker && gameData[selectedTicker] && (
                                    <div className={styles.tradeControlsContainer}>
                                        <BuyPanel
                                            ticker={selectedTicker}
                                            currentPrice={currentSelectedPrice}
                                            cash={cash}
                                            leverage={leverage}
                                            onSetLeverage={handleSetLeverage}
                                            onBuy={handleBuy}
                                            disabled={isGameOver}
                                        />
                                        <SellPanel
                                            ticker={selectedTicker}
                                            currentPrice={currentSelectedPrice}
                                            positions={positions}
                                            onSell={handleSell}
                                            disabled={isGameOver}
                                        />
                                    </div>
                                )}
                            </div>

                            {/* --- Sidebar Column --- */}
                            <div className={styles.sidebar}>
                                <StockSelectorPanel
                                    tickers={availableTickers}
                                    selectedTicker={selectedTicker}
                                    onSelectTicker={handleSelectTicker}
                                />
                                <PortfolioDisplay
                                    cash={cash}
                                    positions={positions}
                                    portfolioValue={portfolioValue}
                                    currentLeverageSelection={leverage}
                                    gameData={gameData}
                                    currentStep={currentStep}
                                />
                            </div>
                        </div>
                    ) : (
                        /* Bot Dashboard View */
                        <div className={styles.gameLayout}>
                            <div className={styles.mainContent}>
                                <div className={styles.gameStatus}>
                                    <span>Step: {currentStep + 1} / {DATA_POINTS_EXPECTED}</span>
                                    {isGameOver && <span className={styles.gameOver}>GAME OVER</span>}
                                </div>

                                {botMessage && (
                                    <div className={`${styles.tradeMessage} ${styles.info}`}>
                                        {botMessage}
                                    </div>
                                )}

                                <BotDashboard
                                    botCash={botCash}
                                    botPositions={botPositions}
                                    botPortfolioValue={botPortfolioValue}
                                    botTradeHistory={botTradeHistory}
                                    userPortfolioValue={portfolioValue}
                                    currentStep={currentStep}
                                    initialCash={initialCash}
                                    isGameOver={isGameOver}
                                    selectedTicker={selectedTicker}
                                />
                            </div>

                            {/* Reuse sidebar from user view */}
                            <div className={styles.sidebar}>
                                <StockSelectorPanel
                                    tickers={availableTickers}
                                    selectedTicker={selectedTicker}
                                    onSelectTicker={handleSelectTicker}
                                />
                                <div className={styles.botStatus}>
                                    <h4>Bot Status</h4>
                                    <p className={botConnected ? styles.connected : styles.disconnected}>
                                        {botConnected ? 'Connected' : 'Disconnected'}
                                    </p>
                                </div>
                            </div>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
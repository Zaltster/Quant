// frontend/app/games/stock-101/page.tsx
"use client";

import React, { useState, useEffect, useCallback } from 'react';
import StockSelectorPanel from '../../../components/StockSelectorPanel';
import MainChartDisplay from '../../../components/MainChartDisplay';
import PortfolioDisplay from '../../../components/PortfolioDisplay';
import BuyPanel from '../../../components/BuyPanel';
import SellPanel from '../../../components/SellPanel';
import { v4 as uuidv4 } from 'uuid'; // Ensure installed: npm install uuid @types/uuid

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

// Interface for Trade History
interface Trade {
    id: string;
    step: number; // Step number (0-based index) when trade occurred
    ticker: string;
    action: 'BUY' | 'SELL';
    quantity: number;
    price: number;
    leverage: number;
    realizedPnl?: number; // Optional: only for SELL trades
}

// --- Component ---
const BACKEND_URL = 'http://127.0.0.1:8000';
const GAME_DURATION_SECONDS = 600;
const DATA_POINTS_EXPECTED = 100;
const UPDATE_INTERVAL_MS = (GAME_DURATION_SECONDS / DATA_POINTS_EXPECTED) * 1000;
const initialCash = 100000;

export default function Stock101Page() {
    // State variables
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
    const [tradeHistory, setTradeHistory] = useState<Trade[]>([]); // State for trade history

    const getCurrentPrice = useCallback((ticker: string): number | null => {
        return gameData?.[ticker]?.[currentStep] ?? null;
    }, [gameData, currentStep]);

    const startGame = useCallback(async () => {
        setIsLoading(true);
        setError(null); setGameData(null); setGameStarted(false); setCurrentStep(0);
        setIsGameOver(false); setSelectedTicker(''); setCash(initialCash);
        setPositions([]); setPortfolioValue(initialCash); setLeverage(1);
        setTradeMessage(null); setTradeHistory([]); // Reset trade history

        try {
            const response = await fetch(`${BACKEND_URL}/game_simulation_data`);
            if (!response.ok) { const errTxt = await response.text(); throw new Error(`API Error: ${response.status} - ${errTxt}`); }
            const data: { status: string; stock_data?: StockData; detail?: string } = await response.json();
            if (data.status === 'success' && data.stock_data) {
                if (Object.keys(data.stock_data).length === 0 || Object.values(data.stock_data).some(arr => !Array.isArray(arr) || arr.length !== DATA_POINTS_EXPECTED)) { throw new Error(`Invalid data structure/length.`); }
                setGameData(data.stock_data);
                const firstTicker = Object.keys(data.stock_data)[0];
                if (firstTicker) { setSelectedTicker(firstTicker); }
                setGameStarted(true); console.log("Game data received.");
            } else { throw new Error(data.detail || 'Backend error.'); }
        } catch (err) { if (err instanceof Error) { setError(err.message); } else { setError("Unknown fetch error."); } console.error("Fetch error:", err); setGameStarted(false); }
        finally { setIsLoading(false); }
    }, []);

    const handleSelectTicker = (ticker: string) => { setSelectedTicker(ticker); };
    const handleSetLeverage = (newLeverage: number) => { setLeverage(newLeverage); };

    const handleBuy = useCallback((ticker: string, quantity: number, tradeLeverage: number) => {
        setTradeMessage(null);
        if (isGameOver || !gameData || quantity <= 0 || !Number.isInteger(quantity)) { /*...*/ return; }
        const currentPrice = getCurrentPrice(ticker);
        if (currentPrice === null || currentPrice <= 0) { /*...*/ return; }
        const cost = quantity * currentPrice;
        const marginRequired = cost / tradeLeverage;

        if (cash >= marginRequired) {
            const newPosition: Position = { id: uuidv4(), ticker: ticker, entryPrice: currentPrice, quantity: quantity, leverage: tradeLeverage, entryStep: currentStep };
            const newTrade: Trade = { id: uuidv4(), step: currentStep, ticker: ticker, action: 'BUY', quantity: quantity, price: currentPrice, leverage: tradeLeverage };
            setTradeHistory(prev => [...prev, newTrade]); // Add to trade history
            setCash(prevCash => prevCash - marginRequired);
            setPositions(prevPositions => [...prevPositions, newPosition]);
            setTradeMessage({ type: 'success', text: `Bought ${quantity} ${ticker} @ ${currentPrice.toFixed(2)} (Lev: ${tradeLeverage}x, Margin: $${marginRequired.toFixed(2)})` });
        } else { /* Insufficient cash error */ }
    }, [cash, currentStep, gameData, isGameOver, getCurrentPrice]);

    const handleSell = useCallback((ticker: string, quantityToSell: number, isLiquidation: boolean = false) => {
        if (!isLiquidation) setTradeMessage(null);
        if ((isGameOver && !isLiquidation) || !gameData || quantityToSell <= 0) return false;
        const currentPrice = getCurrentPrice(ticker);
        const effectiveSellPrice = currentPrice ?? 0;
        if ((effectiveSellPrice < 0 || currentPrice === null) && !isLiquidation) { /*...*/ return false; }

        let remainingToSell = quantityToSell;
        let totalRealizedPnl = 0;
        let totalMarginFreed = 0;
        let actualQuantitySold = 0;
        let leverageOfSoldShares = 1; // Default if no position found (shouldn't happen)
        const updatedPositions: Position[] = [];
        const positionsForTicker = positions.filter(p => p.ticker === ticker).sort((a, b) => a.entryStep - b.entryStep);
        const otherPositions = positions.filter(p => p.ticker !== ticker);
        const totalSharesHeld = positionsForTicker.reduce((sum, p) => sum + p.quantity, 0);
        const tolerance = 1e-9;

        if (totalSharesHeld < quantityToSell - tolerance) { if (!isLiquidation) setTradeMessage({ type: 'error', text: `Not enough shares...` }); return false; }

        for (const pos of positionsForTicker) {
            if (remainingToSell <= tolerance) { updatedPositions.push(pos); continue; }
            const sellQtyFromThisPos = Math.min(remainingToSell, pos.quantity);
            const costBasisForSold = sellQtyFromThisPos * pos.entryPrice;
            const marginFreed = costBasisForSold / pos.leverage;
            const realizedPnl = (effectiveSellPrice - pos.entryPrice) * sellQtyFromThisPos * pos.leverage;

            totalRealizedPnl += realizedPnl;
            totalMarginFreed += marginFreed;
            remainingToSell -= sellQtyFromThisPos;
            actualQuantitySold += sellQtyFromThisPos;
            leverageOfSoldShares = pos.leverage; // Store leverage for the trade record

            if (pos.quantity - sellQtyFromThisPos > tolerance) {
                updatedPositions.push({ ...pos, quantity: pos.quantity - sellQtyFromThisPos });
            }
        }

        if (actualQuantitySold > tolerance) {
            setPositions([...otherPositions, ...updatedPositions]);
            setCash(prevCash => prevCash + totalMarginFreed + totalRealizedPnl);
            // Record Sell Trade
            const newTrade: Trade = {
                id: uuidv4(), step: currentStep, ticker: ticker, action: 'SELL',
                quantity: actualQuantitySold, price: effectiveSellPrice,
                leverage: leverageOfSoldShares, // Use leverage from sold position
                realizedPnl: totalRealizedPnl
            };
            setTradeHistory(prev => [...prev, newTrade]);

            if (!isLiquidation) { setTradeMessage({ type: 'success', text: `Sold ${actualQuantitySold.toFixed(4)} ${ticker} @ ${effectiveSellPrice.toFixed(2)}. P&L: $${totalRealizedPnl.toFixed(2)}` }); }
            return true;
        } else {
            if (!isLiquidation) setTradeMessage({ type: 'error', text: `Sell quantity too small or error.` });
            return false;
        }
    }, [positions, currentStep, gameData, isGameOver, getCurrentPrice]); // Removed setTradeHistory from deps

    // --- Equity Calculation & Liquidation Check Effect ---
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
                const priceDropPercent = pos.entryPrice > 1e-9 ? (currentPrice / pos.entryPrice) - 1 : -1.0;
                const lossPercentLeveraged = priceDropPercent * pos.leverage;
                const liquidationThreshold = -1.0 + 1e-9;
                if ((lossPercentLeveraged <= liquidationThreshold && pos.leverage > 1) || currentPrice <= 1e-9) {
                    liquidatedPositionsThisStep.push(pos); keepPosition = false;
                    console.warn(`LIQUIDATION Check Triggered: ${pos.ticker}`);
                } else {
                    const unrealizedPnl = (currentPrice - pos.entryPrice) * pos.quantity * pos.leverage;
                    calculatedEquity += requiredMargin + unrealizedPnl;
                }
            }
            if (keepPosition) { positionsToKeep.push(pos); }
        });

        setPortfolioValue(calculatedEquity); // Update equity first

        if (liquidatedPositionsThisStep.length > 0) {
            console.log(`Processing ${liquidatedPositionsThisStep.length} liquidations...`);
            let totalLiquidationPnl = 0;
            // Create a list of sell actions to perform after iterating
            const sellActions: { ticker: string; quantity: number }[] = [];
            liquidatedPositionsThisStep.forEach(pos => {
                // PnL calculation for message (might differ slightly from handleSell if price is null)
                const liquidationPrice = getCurrentPrice(pos.ticker) ?? 0;
                const pnl = (liquidationPrice - pos.entryPrice) * pos.quantity * pos.leverage;
                totalLiquidationPnl += pnl;
                sellActions.push({ ticker: pos.ticker, quantity: pos.quantity });
            });

            // Perform sells after calculating equity based on pre-liquidation state
            let allSellsSucceeded = true;
            sellActions.forEach(action => {
                const sellSuccess = handleSell(action.ticker, action.quantity, true); // isLiquidation = true
                if (!sellSuccess) allSellsSucceeded = false;
            });

            if (allSellsSucceeded) {
                setTradeMessage({ type: 'error', text: `${liquidatedPositionsThisStep.length} position(s) liquidated! Realized P&L: $${totalLiquidationPnl.toFixed(2)}` });
            } else {
                setTradeMessage({ type: 'error', text: `Error occurred during liquidation process.` });
            }
            // Note: handleSell updates cash and positions state
        }
    }, [currentStep, cash, positions, gameData, gameStarted, getCurrentPrice, handleSell]); // handleSell added

    // --- Bankruptcy Check Effect ---
    useEffect(() => {
        if (gameStarted && !isGameOver && portfolioValue <= 0 && positions.length === 0) {
            console.error("BANKRUPT!"); setIsGameOver(true);
            setTradeMessage({ type: 'error', text: `BANKRUPT! Equity <= 0` });
        }
    }, [portfolioValue, positions, gameStarted, isGameOver]);

    // --- Game Loop Timer useEffect ---
    useEffect(() => {
        if (!gameStarted || !gameData || isGameOver) return;
        const intervalId = setInterval(() => setCurrentStep(prev => prev + 1), UPDATE_INTERVAL_MS);
        if (currentStep >= DATA_POINTS_EXPECTED - 1) {
            clearInterval(intervalId);
            if (!isGameOver) { setIsGameOver(true); console.log("Game ended."); }
        }
        return () => clearInterval(intervalId);
    }, [gameStarted, gameData, isGameOver, currentStep]);

    // --- Render Logic ---
    const availableTickers = gameData ? Object.keys(gameData) : [];
    const currentSelectedPrice = selectedTicker ? getCurrentPrice(selectedTicker) : null;
    // Filter trade history for the selected ticker
    const tradesForSelectedTicker = tradeHistory.filter(t => t.ticker === selectedTicker);

    return (
        <div>
            <h1>Stock 101 Trading Simulation</h1>
            {!gameStarted && (<button onClick={startGame} disabled={isLoading}> {isLoading ? 'Loading...' : 'Start Game'} </button>)}
            {isLoading && <p>Loading...</p>}
            {error && <p style={{ color: 'red' }}>Error: {error}</p>}
            {gameStarted && gameData && (
                <div className="game-layout">
                    <div className="main-content">
                        <p>Current Day/Step: {currentStep + 1} / {DATA_POINTS_EXPECTED}</p>
                        {isGameOver && <p style={{ color: 'red', fontWeight: 'bold' }}>GAME OVER</p>}
                        {tradeMessage && <p style={{ color: tradeMessage.type === 'error' ? 'orange' : 'green', fontWeight: 'bold', marginTop: '10px', minHeight: '1.2em' }}>{tradeMessage.text}</p>}
                        {selectedTicker && gameData[selectedTicker] ? (
                            <MainChartDisplay
                                key={selectedTicker}
                                ticker={selectedTicker}
                                priceHistory={gameData[selectedTicker]}
                                currentStep={currentStep}
                                trades={tradesForSelectedTicker} // Pass filtered trades
                            />
                        ) : availableTickers.length > 0 ? (<p>Select a stock.</p>) : null}
                        {selectedTicker && gameData[selectedTicker] && (
                            <div className="trade-panels">
                                <BuyPanel ticker={selectedTicker} currentPrice={currentSelectedPrice} cash={cash} leverage={leverage} onSetLeverage={handleSetLeverage} onBuy={handleBuy} disabled={isGameOver} />
                                <SellPanel ticker={selectedTicker} currentPrice={currentSelectedPrice} positions={positions} onSell={handleSell} disabled={isGameOver} />
                            </div>
                        )}
                    </div>
                    <div className="sidebar">
                        <StockSelectorPanel tickers={availableTickers} selectedTicker={selectedTicker} onSelectTicker={handleSelectTicker} />
                        <PortfolioDisplay cash={cash} positions={positions} portfolioValue={portfolioValue} currentLeverageSelection={leverage} gameData={gameData} currentStep={currentStep} />
                    </div>
                </div>
            )}
            <style jsx>{`
        .game-layout { display: flex; flex-direction: row; gap: 30px; margin-top: 20px; align-items: flex-start; }
        .main-content { flex-grow: 1; display: flex; flex-direction: column; }
        .trade-panels { display: flex; flex-direction: row; gap: 20px; margin-top: 20px; justify-content: space-around; flex-wrap: wrap; }
        .sidebar { flex-shrink: 0; width: 240px; }
        button:disabled { cursor: not-allowed; opacity: 0.6; }
        .tradeMessage { margin: 10px 0; padding: 8px; border-radius: 4px; text-align: center; min-height: 1.5em; font-size: 0.9em;}
        .tradeMessage.error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .tradeMessage.success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
         p { margin: 0.5em 0;}
      `}</style>
        </div>
    );
}
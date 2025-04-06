// app/games/stock-101/page.tsx
"use client"; // Still a Client Component

import React, { useState, useEffect, useCallback } from 'react';
import StockSelectorPanel from '../../../components/StockSelectorPanel'; // Import sidebar
import MainChartDisplay from '../../../components/MainChartDisplay'; // Import main display

const BACKEND_URL = 'http://127.0.0.1:8000';
const GAME_DURATION_SECONDS = 600;
const DATA_POINTS_EXPECTED = 100;
const UPDATE_INTERVAL_MS = (GAME_DURATION_SECONDS / DATA_POINTS_EXPECTED) * 1000; // ~6 seconds

interface StockData {
    [ticker: string]: number[];
}

export default function Stock101Page() {
    // State variables
    const [gameData, setGameData] = useState<StockData | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [gameStarted, setGameStarted] = useState<boolean>(false);
    const [currentStep, setCurrentStep] = useState<number>(0);
    const [isGameOver, setIsGameOver] = useState<boolean>(false);
    const [selectedTicker, setSelectedTicker] = useState<string>(''); // State for selected stock

    // Function to fetch data and start the game
    const startGame = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        setGameData(null);
        setGameStarted(false);
        setCurrentStep(0);
        setIsGameOver(false);
        setSelectedTicker(''); // Reset selected ticker

        try {
            console.log("Attempting to fetch game data...");
            const response = await fetch(`${BACKEND_URL}/game_simulation_data`);
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`API Error: ${response.status} - ${errorText}`);
            }
            const data: { status: string; stock_data?: StockData; detail?: string } = await response.json();

            if (data.status === 'success' && data.stock_data) {
                if (Object.keys(data.stock_data).length === 0 || Object.values(data.stock_data).some(arr => arr.length !== DATA_POINTS_EXPECTED)) {
                    throw new Error(`Received invalid data structure or length from backend.`);
                }

                setGameData(data.stock_data);
                // Set the first ticker as initially selected
                const firstTicker = Object.keys(data.stock_data)[0];
                if (firstTicker) {
                    setSelectedTicker(firstTicker);
                }
                setGameStarted(true);
                console.log("Game data received, game starting:", data.stock_data);
            } else {
                throw new Error(data.detail || 'Backend failed to generate data.');
            }
        } catch (err) {
            if (err instanceof Error) { setError(err.message); console.error("Fetch error:", err); }
            else { setError("An unknown error occurred during fetch."); console.error("Fetch error:", err); }
            setGameStarted(false);
        } finally {
            setIsLoading(false);
        }
    }, []);

    // Callback function for the sidebar
    const handleSelectTicker = (ticker: string) => {
        setSelectedTicker(ticker);
    };

    // --- Game Loop Logic (useEffect remains the same) ---
    useEffect(() => {
        if (!gameStarted || !gameData || isGameOver) return;
        console.log(`Starting game timer. Interval: ${UPDATE_INTERVAL_MS}ms`);
        const intervalId = setInterval(() => {
            setCurrentStep(prevStep => {
                const nextStep = prevStep + 1;
                if (nextStep >= DATA_POINTS_EXPECTED) {
                    clearInterval(intervalId);
                    setIsGameOver(true);
                    console.log("Game simulation finished.");
                    return prevStep;
                }
                return nextStep;
            });
        }, UPDATE_INTERVAL_MS);
        return () => { console.log("Clearing game timer interval."); clearInterval(intervalId); }
    }, [gameStarted, gameData, isGameOver]);

    // Get tickers only when gameData is available
    const availableTickers = gameData ? Object.keys(gameData) : [];

    return (
        <div>
            <h1>Stock 101 Trading Simulation</h1>

            {!gameStarted && (
                <button onClick={startGame} disabled={isLoading}>
                    {isLoading ? 'Loading Data...' : 'Start Game'}
                </button>
            )}

            {isLoading && <p>Loading simulation data from backend...</p>}
            {error && <p style={{ color: 'red' }}>Error loading data: {error}</p>}

            {gameStarted && gameData && selectedTicker && ( // Only render if game started AND data loaded AND ticker selected
                <div className="game-layout">
                    {/* Left Side: Main Chart and Controls */}
                    <div className="main-content">
                        <p>Current Day/Step: {currentStep + 1} / {DATA_POINTS_EXPECTED}</p>
                        {isGameOver && <p style={{ color: 'red', fontWeight: 'bold' }}>GAME OVER</p>}

                        <MainChartDisplay
                            key={selectedTicker} // Add key to force re-render on ticker change if needed
                            ticker={selectedTicker}
                            priceHistory={gameData[selectedTicker]} // Pass data for the selected ticker
                            currentStep={currentStep}
                        />
                    </div>

                    {/* Right Side: Stock Selector Panel */}
                    <div className="sidebar">
                        <StockSelectorPanel
                            tickers={availableTickers}
                            selectedTicker={selectedTicker}
                            onSelectTicker={handleSelectTicker}
                        />
                        {/* You could add portfolio summary here later */}
                    </div>
                </div>
            )}

            {/* Basic Styling for Layout */}
            <style jsx>{`
        .game-layout {
          display: flex;
          flex-direction: row;
          gap: 20px; /* Space between main content and sidebar */
          margin-top: 20px;
        }
        .main-content {
          flex-grow: 1; /* Allows chart area to take up available space */
        }
        .sidebar {
          flex-shrink: 0; /* Prevent sidebar from shrinking */
        }
        button:disabled { cursor: not-allowed; opacity: 0.6; }
      `}</style>
        </div>
    );
}
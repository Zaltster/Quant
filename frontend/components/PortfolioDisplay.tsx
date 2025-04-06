// frontend/components/PortfolioDisplay.tsx
import React from 'react';

interface Position {
    id: string;
    ticker: string;
    entryPrice: number;
    quantity: number;
    leverage: number;
    entryStep: number;
}

interface PortfolioDisplayProps {
    cash: number;
    positions: Position[];
    portfolioValue: number; // Represents Equity
    currentLeverageSelection: number;
    gameData: { [ticker: string]: number[] } | null;
    currentStep: number;
}

export default function PortfolioDisplay({
    cash,
    positions,
    portfolioValue,
    currentLeverageSelection,
    gameData,
    currentStep
}: PortfolioDisplayProps) {

    const holdingsSummary: Record<string, number> = {};
    positions.forEach(pos => {
        holdingsSummary[pos.ticker] = (holdingsSummary[pos.ticker] || 0) + pos.quantity;
    });

    const currentHoldings = Object.entries(holdingsSummary)
        .filter(([ticker, quantity]) => quantity > 0.0001); // Use tolerance

    return (
        <div className="portfolio-panel" style={{ marginTop: '20px', border: '1px solid #ccc', padding: '10px 15px', borderRadius: '8px', backgroundColor: '#f8f9fa' }}>
            <h4>Portfolio</h4>
            <p>Cash: <strong>${cash.toFixed(2)}</strong></p>
            <p>Equity: <strong style={{ color: portfolioValue >= initialCash ? 'green' : 'red' }}>${portfolioValue.toFixed(2)}</strong></p>
            <p style={{ fontSize: '0.9em' }}>Next Trade Leverage: {currentLeverageSelection}x</p>
            <h5 style={{ marginTop: '15px', marginBottom: '5px', borderTop: '1px solid #eee', paddingTop: '10px' }}>Holdings:</h5>
            {currentHoldings.length > 0 ? (
                <ul style={{ listStyle: 'none', paddingLeft: '5px', fontSize: '0.9em', margin: 0 }}>
                    {currentHoldings.map(([ticker, quantity]) => (
                        <li key={ticker}>{ticker}: {quantity.toFixed(4)} shares</li>
                    ))}
                </ul>
            ) : (
                <p style={{ fontSize: '0.9em', color: '#666', margin: '5px 0' }}>No current holdings</p>
            )}
        </div>
    );
}

// Define initialCash here or import if defined globally
const initialCash = 100000;
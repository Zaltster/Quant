// frontend/components/SellPanel.tsx
"use client";

import React, { useState, useEffect } from 'react';
import styles from './TradePanel.module.css';

interface Position {
    id: string;
    ticker: string;
    entryPrice: number;
    quantity: number;
    leverage: number;
    entryStep: number;
}

interface SellPanelProps {
    ticker: string;
    currentPrice: number | null;
    positions: Position[]; // All open positions
    onSell: (ticker: string, quantity: number) => void; // Quantity based sell
    disabled: boolean;
}

export default function SellPanel({
    ticker, currentPrice, positions, onSell, disabled
}: SellPanelProps) {
    const [quantity, setQuantity] = useState<string>('1'); // Keep as string for input control

    const sharesHeld = positions
        .filter(p => p.ticker === ticker)
        .reduce((sum, p) => sum + p.quantity, 0);

    useEffect(() => {
        setQuantity(sharesHeld > 1 ? '1' : sharesHeld.toFixed(4)); // Default to 1 or max if less than 1
    }, [ticker, sharesHeld]);

    const handleQuantityChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setQuantity(event.target.value);
    };

    const executeSell = () => {
        const numericQuantity = parseFloat(quantity);
        // Use tolerance for floating point comparison
        if (!disabled && currentPrice !== null && numericQuantity > 1e-9 && numericQuantity <= sharesHeld + 1e-9) {
            onSell(ticker, numericQuantity);
        } else {
            console.error("Sell validation failed", { disabled, currentPrice, numericQuantity, sharesHeld });
        }
    };

    const sellMax = () => {
        if (!disabled && currentPrice !== null && sharesHeld > 1e-9) { // Use tolerance
            onSell(ticker, sharesHeld);
        }
    }

    // Validate quantity for button disabling
    const numericQuantity = parseFloat(quantity);
    const isSellQuantityValid = numericQuantity > 1e-9 && numericQuantity <= sharesHeld + 1e-9;


    return (
        <div className={`${styles.tradePanel} ${styles.sellPanel}`}>
            <h4>Sell {ticker}</h4>
            <p className={styles.info}>Shares Held: {sharesHeld.toFixed(4)}</p>
            <div className={styles.inputGroup}>
                <label htmlFor={`quantity-sell-${ticker}`}>Quantity: </label>
                <input
                    type="number"
                    id={`quantity-sell-${ticker}`}
                    value={quantity}
                    onChange={handleQuantityChange}
                    min="0"
                    max={sharesHeld}
                    step="any" // Allow selling fractional parts
                    disabled={disabled || sharesHeld <= 0}
                    className={styles.inputField}
                />
            </div>
            <div className={styles.tradeButtons}>
                <button
                    onClick={executeSell}
                    disabled={disabled || currentPrice === null || !isSellQuantityValid}
                    className={`${styles.tradeButton} ${styles.sellButton}`}
                >
                    Sell {quantity || 0} {ticker}
                </button>
                <button
                    onClick={sellMax}
                    disabled={disabled || currentPrice === null || sharesHeld <= 1e-9} // Use tolerance
                    className={`${styles.tradeButton} ${styles.sellMaxButton}`}
                >
                    Sell Max ({sharesHeld.toFixed(2)})
                </button>
            </div>
        </div>
    );
}
// frontend/components/BuyPanel.tsx
"use client";

import React, { useState, useEffect } from 'react';
import styles from './TradePanel.module.css'; // Reuse shared CSS

interface BuyPanelProps {
    ticker: string;
    currentPrice: number | null;
    cash: number;
    leverage: number;
    onSetLeverage: (newLeverage: number) => void;
    onBuy: (ticker: string, quantity: number, leverage: number) => void; // Expect quantity
    disabled: boolean;
}

export default function BuyPanel({
    ticker, currentPrice, cash, leverage, onSetLeverage, onBuy, disabled
}: BuyPanelProps) {
    const [quantity, setQuantity] = useState<string>('1'); // State for quantity input

    // Reset quantity when ticker changes
    useEffect(() => {
        setQuantity('1');
    }, [ticker]);


    const handleQuantityChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setQuantity(event.target.value); // Keep as string for input flexibility
    };

    const handleLeverageChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        onSetLeverage(parseInt(event.target.value, 10));
    };

    const executeBuy = () => {
        const numericQuantity = parseInt(quantity, 10); // Parse to integer for validation/trading
        if (!disabled && currentPrice !== null && numericQuantity > 0 && Number.isInteger(numericQuantity)) {
            onBuy(ticker, numericQuantity, leverage);
        } else {
            console.error("Buy validation failed", { disabled, currentPrice, numericQuantity });
            // Optionally set a local error state to display in this panel
        }
    };

    // Calculate approx margin needed for the entered quantity
    const numericQuantity = parseInt(quantity, 10);
    const approxMarginNeeded = (currentPrice !== null && numericQuantity > 0)
        ? (numericQuantity * currentPrice / leverage)
        : 0;
    const canAfford = cash >= approxMarginNeeded;

    return (
        <div className={styles.tradePanel}>
            <h4>Buy {ticker}</h4>
            <div className={styles.inputGroup}>
                <label htmlFor={`quantity-buy-${ticker}`}>Quantity: </label>
                <input
                    type="number"
                    id={`quantity-buy-${ticker}`}
                    value={quantity}
                    onChange={handleQuantityChange}
                    min="1"
                    step="1" // Step by whole shares
                    disabled={disabled}
                    className={styles.inputField}
                />
            </div>
            <div className={styles.inputGroup}>
                <label htmlFor={`leverage-buy-${ticker}`}>Leverage: </label>
                <select
                    id={`leverage-buy-${ticker}`}
                    value={leverage}
                    onChange={handleLeverageChange}
                    disabled={disabled}
                    className={styles.selectField}
                >
                    <option value={1}>1x</option>
                    <option value={2}>2x</option>
                    <option value={3}>3x</option>
                    <option value={5}>5x</option>
                </select>
            </div>
            <p className={styles.info}>Approx Margin Needed: ${approxMarginNeeded.toFixed(2)}</p>
            <button
                onClick={executeBuy}
                disabled={disabled || currentPrice === null || !canAfford || parseInt(quantity, 10) <= 0 || !Number.isInteger(parseInt(quantity, 10))}
                className={`${styles.tradeButton} ${styles.buyButton}`}
            >
                Buy {quantity} {ticker}
            </button>
        </div>
    );
}
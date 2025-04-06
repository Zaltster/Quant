// frontend/components/MainChartDisplay.tsx
"use client";

import React, { useMemo } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer // Keep Circle import for clarity if used explicitly below
} from 'recharts';
import styles from './MainChartDisplay.module.css'; // Ensure this CSS module file exists

// Define Trade interface locally or import from shared types
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

interface MainChartDisplayProps {
    ticker: string;
    priceHistory: number[];
    currentStep: number;
    trades: Trade[];
}

// --- Custom Tooltip Component ---
const CustomTooltip = ({ active, payload, label }: any) => {
    // This component remains the same as the previous version
    if (active && payload && payload.length) {
        const data = payload[0].payload;
        return (
            <div className={styles.customTooltip}>
                <p className={styles.tooltipLabel}>{`Step: ${label}`}</p>
                <p className={styles.tooltipPrice}>{`Price: ${data.price?.toFixed(2) ?? 'N/A'}`}</p>
                {data.tradeInfo && data.tradeInfo.length > 0 && (
                    <div className={styles.tooltipTradeDetails}>
                        {data.tradeInfo.map((trade: Trade) => (
                            <p key={trade.id} className={trade.action === 'BUY' ? styles.tooltipBuy : styles.tooltipSell}>
                                {trade.action} {trade.quantity.toFixed(4)} @ {trade.price.toFixed(2)}
                                {trade.action === 'SELL' && trade.realizedPnl !== undefined && ` (P&L: $${trade.realizedPnl.toFixed(2)})`}
                            </p>
                        ))}
                    </div>
                )}
            </div>
        );
    }
    return null;
};

// --- Custom Dot Rendering Logic (Function passed to dot prop) ---
// This function decides what kind of dot to render for EACH point on the line
const renderConditionalDot = (props: any) => {
    const { cx, cy, payload, key } = props; // payload is the chartData object: {step, price, tradeInfo?}

    // --- Default Dot Style (Blue, small) ---
    const defaultDotRadius = 2;
    const defaultDotFill = "#0d6efd"; // Match line color

    // --- Trade Marker Style (Larger, Green/Red) ---
    const tradeDotRadius = 5;
    const tradeStroke = "#fff";
    const tradeStrokeWidth = 1;

    // Check if trade(s) occurred at this step
    if (payload.tradeInfo && payload.tradeInfo.length > 0) {
        // Render LARGER, COLORED dot for TRADES
        const isSell = payload.tradeInfo.some((t: Trade) => t.action === 'SELL');
        const isBuy = payload.tradeInfo.some((t: Trade) => t.action === 'BUY');
        let fillColor = '#6c757d'; // Default grey if mixed trades
        if (isSell) fillColor = '#dc3545'; // Red for sell
        else if (isBuy) fillColor = '#28a745'; // Green for buy

        // Use lowercase <circle> for SVG element
        return <circle key={`trade-dot-${key}-${payload.step}`} cx={cx} cy={cy} r={tradeDotRadius} fill={fillColor} stroke={tradeStroke} strokeWidth={tradeStrokeWidth} />;
    } else {
        // Render SMALLER, DEFAULT dot for NON-TRADE steps
        return <circle key={`default-dot-${key}-${payload.step}`} cx={cx} cy={cy} r={defaultDotRadius} fill={defaultDotFill} />;
    }
};


// --- Main Chart Component ---
export default function MainChartDisplay({
    ticker,
    priceHistory,
    currentStep,
    trades
}: MainChartDisplayProps) {

    const currentPrice = (priceHistory && currentStep >= 0 && priceHistory.length > currentStep)
        ? priceHistory[currentStep]
        : null;

    // Memoize the calculation of trades mapped by step
    const tradesByStep = useMemo(() => {
        const map: Record<number, Trade[]> = {};
        trades?.forEach(trade => {
            const stepIndex = trade.step;
            if (stepIndex < 0) return;
            if (!map[stepIndex]) map[stepIndex] = [];
            map[stepIndex].push(trade);
        });
        return map;
    }, [trades]);

    // Prepare data for the chart, embedding trade info
    const chartData = priceHistory
        ?.slice(0, currentStep + 1)
        .map((price, index) => {
            const stepIndex = index;
            const stepNumber = index + 1;
            const tradesAtThisStep = tradesByStep[stepIndex] || [];
            return {
                step: stepNumber,
                price: price,
                tradeInfo: tradesAtThisStep.length > 0 ? tradesAtThisStep : undefined
            };
        }) || [];

    return (
        <div className={styles.chartArea}>
            <h3>{ticker}</h3>
            <p> Current Price (Step {currentStep + 1}):
                <span className={styles.priceValue}>
                    {currentPrice !== null ? currentPrice.toFixed(2) : 'Loading...'}
                </span>
            </p>
            <div className={styles.chartContainer}>
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 20 }} >
                        <CartesianGrid strokeDasharray="3 3" stroke="#ccc" />
                        <XAxis dataKey="step" label={{ value: "Step", position: "insideBottom", dy: 10 }} type="number" domain={[1, Math.max(currentStep + 1, 10)]} allowDecimals={false} />
                        <YAxis domain={['auto', 'auto']} tickFormatter={(tick) => typeof tick === 'number' ? tick.toFixed(0) : tick} label={{ value: "Price", angle: -90, position: 'insideLeft', offset: -5 }} width={70} />
                        <Tooltip content={<CustomTooltip />} />
                        <Line
                            type="linear"
                            dataKey="price"
                            stroke="#0d6efd"
                            strokeWidth={2}
                            // --- Pass the rendering FUNCTION to the dot prop ---
                            dot={renderConditionalDot}
                            // ---                                           ---
                            activeDot={{ r: 6 }} // Default active dot styling on hover
                            isAnimationActive={false}
                            name={ticker}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
            {/* Removed Trade controls placeholder */}
        </div>
    );
}
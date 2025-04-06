// frontend/components/MainChartDisplay.tsx
"use client"; // Mark as Client Component because Recharts uses client-side features/hooks

import React from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend, // Keep Legend import if you might use it later
    ResponsiveContainer
} from 'recharts';
import styles from './MainChartDisplay.module.css'; // Ensure this CSS module file exists

// Define types for the component's props
interface MainChartDisplayProps {
    ticker: string;
    priceHistory: number[];
    currentStep: number;
}

export default function MainChartDisplay({
    ticker,
    priceHistory,
    currentStep
}: MainChartDisplayProps) {

    // Ensure currentStep doesn't exceed bounds, default to null if no data yet
    const currentPrice = (priceHistory && currentStep >= 0 && priceHistory.length > currentStep)
        ? priceHistory[currentStep]
        : null; // Use null if data not ready

    // Prepare data for the chart: array of objects {step: number, price: number}
    // Only include data up to the current step
    const chartData = priceHistory
        ?.slice(0, currentStep + 1) // Use optional chaining ?. in case priceHistory is initially null/undefined
        .map((price, index) => ({
            step: index + 1, // Display step number starting from 1
            price: price,
        })) || []; // Provide empty array if data isn't ready

    // Add console log to verify data if needed (can remove later)
    // console.log(`Chart Data for ${ticker} at Step ${currentStep + 1}:`, chartData);

    return (
        <div className={styles.chartArea}>
            {/* Display Ticker and Current Price */}
            <h3>{ticker}</h3>
            <p>
                Current Price (Step {currentStep + 1}):
                <span className={styles.priceValue}>
                    {currentPrice !== null ? currentPrice.toFixed(2) : 'Loading...'}
                </span>
            </p>

            {/* Chart Container */}
            <div className={styles.chartContainer}>
                {/* ResponsiveContainer makes the chart adapt to parent div size */}
                <ResponsiveContainer width="100%" height="100%">
                    {/* LineChart is the SINGLE direct child of ResponsiveContainer */}
                    <LineChart
                        data={chartData}
                        margin={{ top: 5, right: 30, left: 20, bottom: 20 }} // Adjust margins
                    >
                        {/* Grid, Axes, Tooltip, Line go INSIDE LineChart */}
                        <CartesianGrid strokeDasharray="3 3" stroke="#ccc" />
                        <XAxis
                            dataKey="step"
                            label={{ value: "Step", position: "insideBottom", dy: 10 }} // Position label below axis
                            type="number" // Ensure step is treated as a number
                            domain={[1, 'dataMax']} // Start axis at step 1
                            allowDecimals={false} // Show integer steps
                        />
                        <YAxis
                            domain={['auto', 'auto']} // Auto-adjust Y-axis domain based on visible data
                            tickFormatter={(tick) => typeof tick === 'number' ? tick.toFixed(0) : tick} // Format Y-axis ticks as integers
                            label={{ value: "Price", angle: -90, position: 'insideLeft', offset: -5 }}
                            width={70} // Adjust width to prevent label overlap
                        />
                        <Tooltip formatter={(value: number | string | Array<string | number>, name: string) => {
                            // Tooltip shows the price formatted
                            if (typeof value === 'number') {
                                return [`${value.toFixed(2)}`, name]; // Show price formatted to 2 decimals
                            }
                            return [value, name];
                        }} />
                        {/* Optional Legend: <Legend /> */}
                        <Line
                            type="linear" // Using linear to connect points directly
                            dataKey="price" // Key in chartData objects referencing the price value
                            stroke="#0d6efd" // Line color
                            strokeWidth={2}
                            // --- MODIFIED THIS LINE ---
                            dot={{ r: 3, fill: '#0d6efd' }} // Render small filled dots at each point
                            // ---                       ---
                            isAnimationActive={false} // Disable animation for better performance on frequent updates
                            name={ticker} // Name shown in tooltip
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            {/* Placeholder for Buy/Sell buttons */}
            <div className={styles.tradeControlsPlaceholder}>Trade Controls for {ticker}</div>
        </div>
    );
}
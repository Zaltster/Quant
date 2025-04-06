// components/StockSelectorPanel.tsx
import React from 'react';
import styles from './StockSelectorPanel.module.css'; // Create this CSS file

interface StockSelectorPanelProps {
    tickers: string[];
    selectedTicker: string;
    onSelectTicker: (ticker: string) => void; // Callback function when a ticker is clicked
}

export default function StockSelectorPanel({
    tickers,
    selectedTicker,
    onSelectTicker,
}: StockSelectorPanelProps) {
    return (
        <div className={styles.panel}>
            <h4>Stocks</h4>
            <ul>
                {tickers.map((ticker) => (
                    <li
                        key={ticker}
                        className={ticker === selectedTicker ? styles.selected : styles.listItem}
                        onClick={() => onSelectTicker(ticker)}
                    >
                        {ticker}
                    </li>
                ))}
            </ul>
        </div>
    );
}

/* Create components/StockSelectorPanel.module.css
.panel {
  border: 1px solid #ccc;
  padding: 10px;
  border-radius: 5px;
  background-color: #f8f9fa;
  min-width: 150px; // Adjust as needed
  height: fit-content; // Adjust height as needed
}
.panel h4 {
  margin-top: 0;
  margin-bottom: 10px;
  text-align: center;
  border-bottom: 1px solid #eee;
  padding-bottom: 5px;
}
.panel ul {
  list-style: none;
  padding: 0;
  margin: 0;
}
.listItem, .selected {
  padding: 8px 10px;
  margin-bottom: 5px;
  cursor: pointer;
  border-radius: 4px;
  transition: background-color 0.2s ease;
}
.listItem:hover {
  background-color: #e9ecef;
}
.selected {
  background-color: #0d6efd; // Example selected color (Bootstrap blue)
  color: white;
  font-weight: bold;
}
*/
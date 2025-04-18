// frontend/components/TabNavigation.tsx
import React from 'react';
import styles from './TabNavigation.module.css';

interface TabNavigationProps {
    activeTab: string;
    onTabChange: (tab: string) => void;
    tabs: { id: string, label: string }[];
}

export default function TabNavigation({ activeTab, onTabChange, tabs }: TabNavigationProps) {
    return (
        <div className={styles.tabNavigation}>
            {tabs.map(tab => (
                <button
                    key={tab.id}
                    className={`${styles.tab} ${activeTab === tab.id ? styles.active : ''}`}
                    onClick={() => onTabChange(tab.id)}
                >
                    {tab.label}
                </button>
            ))}
        </div>
    );
}
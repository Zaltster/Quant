// components/NavBar.tsx
import Link from 'next/link';
import styles from './NavBar.module.css'; // Create this CSS file for basic styling

export default function NavBar() {
    return (
        <nav className={styles.navbar}>
            <Link href="/" className={styles.brand}>
                Quant Game Platform
            </Link>
            <div className={styles.gameLinks}>
                {/* Add more games here later */}
                <Link href="/games/stock-101" className={styles.gameLink}>
                    Stock 101
                </Link>
                {/* <Link href="/games/game-2" className={styles.gameLink}>Game 2</Link> */}
            </div>
        </nav>
    );
}

/* Optional: Create components/NavBar.module.css
.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  background-color: #333;
  color: white;
}
.brand {
  color: white;
  text-decoration: none;
  font-weight: bold;
}
.gameLinks a {
  color: #aaa;
  text-decoration: none;
  margin-left: 1rem;
}
.gameLinks a:hover {
  color: white;
}
*/
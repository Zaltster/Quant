// frontend/app/page.tsx
"use client"; // Mark as client component if using client-side routing hooks like useRouter, otherwise not needed for just Link

import Head from 'next/head';
import Link from 'next/link'; // Import Link for navigation
import styles from './page.module.css'; // Using default Next.js page module for basic styling (can customize)
// Or import a dedicated CSS module: import styles from './HomePage.module.css';

export default function Home() {
  // If you wanted programmatic navigation instead of Link:
  // import { useRouter } from 'next/navigation'; // Note: Use 'next/navigation' in App Router
  // const router = useRouter();
  // const startInterview = () => {
  //   router.push('/games/stock-101'); // Navigate to the first game
  // };

  return (
    <div className={styles.container}> {/* Use a container for centering/padding */}
      <Head>
        <title>Quant Interview Practice Platform</title>
        <meta name="description" content="Simulated quant interview games and challenges" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className={styles.main}>
        <header className={styles.hero}>
          <h1 className={styles.title}>
            Quant Interview Practice Platform
          </h1>

          <p className={styles.description}>
            Sharpen your skills with realistic quantitative trading simulations
            and challenges, presented in an interview-style format. Prepare for your next quant interview.
          </p>

          {/* Call to Action Button - Links to the first game */}
          <Link href="/games/stock-101" passHref legacyBehavior>
            <a className={styles.ctaButton}>Start Interview Session</a>
          </Link>
          {/* Or if using useRouter: */}
          {/* <button onClick={startInterview} className={styles.ctaButton}>
             Start Interview Session
          </button> */}

        </header>

        <section className={styles.features}>
          <h2>Interview Modules Included:</h2>
          <div className={styles.featureList}>
            {/* List games here - dynamically later if needed */}
            <div className={styles.featureCard}>
              <h3>Stock 101 Trading Simulation</h3>
              <p>Test your market intuition and risk management in a 10-minute simulated trading session across 5 diverse assets.</p>
            </div>
            {/* Add more cards when you have more games */}
            {/* <div className={styles.featureCard}>...</div> */}
          </div>
        </section>
      </main>

      {/* Basic Styling (You can put this in a global CSS, page module, or keep inline) */}
      <style jsx>{`
        .container {
          min-height: 80vh; /* Adjust height */
          padding: 0;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
        }
        .main {
          padding: 4rem 1rem; /* More padding */
          flex: 1;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          max-width: 800px; /* Limit content width */
          width: 100%;
        }
        .hero {
          text-align: center;
          margin-bottom: 4rem;
        }
        .title {
          margin: 0;
          line-height: 1.15;
          font-size: 3.5rem; /* Larger title */
          font-weight: bold;
        }
        .title,
        .description {
          text-align: center;
        }
        .description {
          line-height: 1.6;
          font-size: 1.2rem;
          color: #555;
          margin: 1.5rem 0 2.5rem 0;
        }
        .ctaButton {
          display: inline-block;
          padding: 1rem 2rem;
          font-size: 1.1rem;
          font-weight: bold;
          color: #fff;
          background-color: #0070f3; /* Example button color */
          border: none;
          border-radius: 8px;
          text-decoration: none;
          cursor: pointer;
          transition: background-color 0.2s ease;
        }
        .ctaButton:hover,
        .ctaButton:focus {
          background-color: #005bb5;
        }

        .features {
          margin-top: 3rem;
          width: 100%;
          text-align: center;
        }
        .features h2 {
            margin-bottom: 2rem;
            font-size: 2rem;
        }
        .featureList {
            display: flex;
            justify-content: center; /* Center cards */
            flex-wrap: wrap;
            gap: 1.5rem;
        }
        .featureCard {
            border: 1px solid #eaeaea;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: left;
            max-width: 350px; /* Max width for cards */
            background-color: #fafafa;
        }
         .featureCard h3 {
            margin-top: 0;
            margin-bottom: 0.5rem;
            font-size: 1.2rem;
         }
         .featureCard p {
            font-size: 1rem;
            line-height: 1.5;
            color: #333;
         }
      `}</style>
    </div>
  );
}
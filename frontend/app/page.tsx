// app/page.tsx
import Image from "next/image"; // Can remove if not using images
import styles from "./page.module.css"; // Default styles module

export default function Home() {
  return (
    <div> {/* Replace default content */}
      <h1>Welcome to the Quant Game Platform!</h1>
      <p>Select a game from the navigation bar above to start practicing.</p>
    </div>
  );
}
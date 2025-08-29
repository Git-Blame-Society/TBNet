import Link from 'next/link';
import Navbar from './components/Navbar';
const Home = () => {
  return (
    <>
      <Navbar />
      <h1>Hello</h1>
      <Link href="/dashboard">Go to Dashboard</Link>
    </>
  );
}

export default Home;

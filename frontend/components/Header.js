import Link from 'next/link';
import { Leaf, Menu, X, Sun, Moon } from 'lucide-react';
import { useState, useEffect } from 'react';

export default function Header() {
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const [isDarkMode, setIsDarkMode] = useState(false);

    useEffect(() => {
        // Check local storage or system preference
        if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
            document.documentElement.classList.add('dark');
            setIsDarkMode(true);
        } else {
            document.documentElement.classList.remove('dark');
            setIsDarkMode(false);
        }
    }, []);

    const toggleDarkMode = () => {
        if (isDarkMode) {
            document.documentElement.classList.remove('dark');
            localStorage.theme = 'light';
            setIsDarkMode(false);
        } else {
            document.documentElement.classList.add('dark');
            localStorage.theme = 'dark';
            setIsDarkMode(true);
        }
    };

    return (
        <header className="bg-white dark:bg-agri-green-950 shadow-sm sticky top-0 z-50 transition-colors duration-200">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16 items-center">
                    <div className="flex items-center">
                        <Link href="/" className="flex items-center gap-2 group">
                            <div className="bg-agri-green-100 dark:bg-agri-green-900 p-2 rounded-lg group-hover:bg-agri-green-200 dark:group-hover:bg-agri-green-800 transition-colors">
                                <Leaf className="h-6 w-6 text-agri-green-600 dark:text-agri-green-400" />
                            </div>
                            <div>
                                <h1 className="text-xl font-bold text-gray-900 dark:text-white leading-none">AgriSense</h1>
                                <p className="text-xs text-agri-green-600 dark:text-agri-green-400 font-medium tracking-wide">Intelligent Crop Yield Prediction</p>
                            </div>
                        </Link>
                    </div>

                    {/* Desktop Navigation */}
                    <nav className="hidden md:flex items-center space-x-8">
                        <div className="flex space-x-6">
                            <Link href="/" className="text-gray-600 dark:text-gray-300 hover:text-agri-green-600 dark:hover:text-agri-green-400 font-medium transition-colors">Predict</Link>
                            <Link href="/dashboard" className="text-gray-600 dark:text-gray-300 hover:text-agri-green-600 dark:hover:text-agri-green-400 font-medium transition-colors">Dashboard</Link>
                            <Link href="/model-info" className="text-gray-600 dark:text-gray-300 hover:text-agri-green-600 dark:hover:text-agri-green-400 font-medium transition-colors">Model Info</Link>
                            <Link href="/about" className="text-gray-600 dark:text-gray-300 hover:text-agri-green-600 dark:hover:text-agri-green-400 font-medium transition-colors">About</Link>
                        </div>

                        <div className="pl-6 border-l border-gray-200 dark:border-gray-800">
                            <button
                                onClick={toggleDarkMode}
                                className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-500 dark:text-gray-400 transition-colors"
                                aria-label="Toggle dark mode"
                            >
                                {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
                            </button>
                        </div>
                    </nav>

                    {/* Mobile menu button */}
                    <div className="md:hidden flex items-center gap-4">
                        <button
                            onClick={toggleDarkMode}
                            className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-500 dark:text-gray-400 transition-colors"
                        >
                            {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
                        </button>
                        <button
                            onClick={() => setIsMenuOpen(!isMenuOpen)}
                            className="text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
                        >
                            {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
                        </button>
                    </div>
                </div>
            </div>

            {/* Mobile Navigation */}
            {isMenuOpen && (
                <div className="md:hidden bg-white dark:bg-agri-green-950 border-t border-gray-100 dark:border-gray-800">
                    <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
                        <Link href="/" className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 dark:text-gray-200 hover:text-agri-green-600 dark:hover:text-agri-green-400 hover:bg-gray-50 dark:hover:bg-gray-900/50">Dashboard</Link>
                        <Link href="/model-info" className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 dark:text-gray-200 hover:text-agri-green-600 dark:hover:text-agri-green-400 hover:bg-gray-50 dark:hover:bg-gray-900/50">Model Info</Link>
                        <Link href="/about" className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 dark:text-gray-200 hover:text-agri-green-600 dark:hover:text-agri-green-400 hover:bg-gray-50 dark:hover:bg-gray-900/50">About</Link>
                        <Link href="/faq" className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 dark:text-gray-200 hover:text-agri-green-600 dark:hover:text-agri-green-400 hover:bg-gray-50 dark:hover:bg-gray-900/50">FAQ</Link>
                    </div>
                </div>
            )}
        </header>
    );
}

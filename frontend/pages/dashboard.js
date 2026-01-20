
import { useState, useEffect } from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';
import { History, TrendingUp, Calendar, Droplets, Thermometer, FlaskConical } from 'lucide-react';

export default function Dashboard() {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch('http://localhost:5000/history')
            .then(res => res.json())
            .then(data => {
                setHistory(data);
                setLoading(false);
            })
            .catch(err => {
                console.error(err);
                setLoading(false);
            });
    }, []);

    return (
        <div className="min-h-screen flex flex-col bg-gray-50 dark:bg-black transition-colors duration-200">
            <Header />

            <main className="flex-grow max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 w-full">
                <div className="flex items-center gap-3 mb-8">
                    <TrendingUp className="h-8 w-8 text-agri-green-500" />
                    <h1 className="text-3xl font-extrabold text-gray-900 dark:text-white">Prediction Analytics</h1>
                </div>

                <div className="bg-white dark:bg-gray-900 rounded-3xl shadow-xl border border-gray-100 dark:border-gray-800 overflow-hidden">
                    <div className="px-6 py-5 border-b border-gray-100 dark:border-gray-800 flex items-center gap-2">
                        <History className="h-5 w-5 text-gray-400" />
                        <h2 className="text-lg font-bold text-gray-900 dark:text-white">Recent Prediction History</h2>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-800">
                            <thead className="bg-gray-50 dark:bg-gray-900/50">
                                <tr>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Date</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Crop</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Rain/Temp</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">N - P - K</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Yield</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Confidence</th>
                                </tr>
                            </thead>
                            <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
                                {loading ? (
                                    Array(5).fill(0).map((_, i) => (
                                        <tr key={i} className="animate-pulse">
                                            <td colSpan="6" className="px-6 py-4"><div className="h-4 bg-gray-100 dark:bg-gray-800 rounded w-full"></div></td>
                                        </tr>
                                    ))
                                ) : history.length === 0 ? (
                                    <tr>
                                        <td colSpan="6" className="px-6 py-12 text-center text-gray-500 dark:text-gray-400">No predictions found in history.</td>
                                    </tr>
                                ) : (
                                    history.map((pred) => (
                                        <tr key={pred.id} className="hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors">
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                                                {new Date(pred.created_at).toLocaleDateString()}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-agri-green-100 dark:bg-agri-green-900/30 text-agri-green-800 dark:text-agri-green-400">
                                                    {pred.crop_type}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                                                <div className="flex items-center gap-4">
                                                    <span className="flex items-center gap-1"><Droplets className="h-3 w-3 text-blue-400" /> {pred.rainfall}</span>
                                                    <span className="flex items-center gap-1"><Thermometer className="h-3 w-3 text-orange-400" /> {pred.temperature}</span>
                                                </div>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400 font-mono">
                                                {pred.nitrogen} - {pred.phosphorus} - {pred.potassium}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-agri-green-600 dark:text-agri-green-400">
                                                {pred.predicted_yield} <span className="text-[10px] font-normal text-gray-400">kg/ha</span>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <div className="flex items-center gap-2">
                                                    <div className="flex-grow bg-gray-200 dark:bg-gray-700 h-1.5 rounded-full w-16 overflow-hidden">
                                                        <div
                                                            className="bg-agri-green-500 h-full rounded-full transition-all duration-1000"
                                                            style={{ width: `${pred.confidence}%` }}
                                                        />
                                                    </div>
                                                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{pred.confidence}%</span>
                                                </div>
                                            </td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            </main>

            <Footer />
        </div>
    );
}

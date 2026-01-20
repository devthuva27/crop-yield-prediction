
import { useState, useEffect } from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';
import ModelPerformanceChart from '../components/ModelPerformanceChart';
import FeatureImportanceChart from '../components/FeatureImportanceChart';
import YieldDistributionChart from '../components/YieldDistributionChart';
import { Cpu, Database, Award, Info } from 'lucide-react';

export default function ModelInfo() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch('http://localhost:5000/analytics')
            .then(res => res.json())
            .then(data => {
                setData(data);
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

            <main className="flex-grow max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                <div className="flex items-center gap-3 mb-8">
                    <Cpu className="h-8 w-8 text-agri-green-500" />
                    <h1 className="text-3xl font-extrabold text-gray-900 dark:text-white">Model Methodology & Performance</h1>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
                    <ModelPerformanceChart data={data?.performance} />
                    <FeatureImportanceChart data={data?.feature_importance} />
                </div>

                <div className="mb-12">
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Historical Context</h2>
                    <YieldDistributionChart data={data?.distribution} />
                    <p className="mt-4 text-gray-600 dark:text-gray-400 max-w-3xl">
                        Our model is trained on a robust dataset spanning 20 districts and 7 major crop varieties in Sri Lanka.
                        The distribution above shows the historical variation in yields, where most farms achieve between 3,000 and 7,000 kg/ha.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="bg-white dark:bg-gray-900 p-6 rounded-2xl border border-gray-100 dark:border-gray-800">
                        <Database className="h-6 w-6 text-blue-500 mb-4" />
                        <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">Training Data</h3>
                        <p className="text-gray-600 dark:text-gray-400 text-sm">
                            Utilizes over 5,000 historical records from Department of Agriculture Sri Lanka, including weather patterns and soil metrics.
                        </p>
                    </div>
                    <div className="bg-white dark:bg-gray-900 p-6 rounded-2xl border border-gray-100 dark:border-gray-800">
                        <Info className="h-6 w-6 text-agri-green-500 mb-4" />
                        <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">XGBoost Architecture</h3>
                        <p className="text-gray-600 dark:text-gray-400 text-sm">
                            Employs Gradient Boosted Decision Trees for capturing non-linear relationships between climate variables and harvest success.
                        </p>
                    </div>
                    <div className="bg-white dark:bg-gray-900 p-6 rounded-2xl border border-gray-100 dark:border-gray-800">
                        <Award className="h-6 w-6 text-yellow-500 mb-4" />
                        <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">Cross-Validated</h3>
                        <p className="text-gray-600 dark:text-gray-400 text-sm">
                            Verified with 5-fold cross-validation ensuring a Mean Absolute Error of less than 4% across various regional scenarios.
                        </p>
                    </div>
                </div>
            </main>

            <Footer />
        </div>
    );
}

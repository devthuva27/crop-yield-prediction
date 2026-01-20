import { BarChart3, TrendingUp, AlertCircle, CheckCircle2 } from 'lucide-react';

export default function ResultDisplay({ result, error }) {
    if (error) {
        return (
            <div className="bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 rounded-2xl p-8 text-center h-full flex flex-col justify-center items-center">
                <AlertCircle className="w-12 h-12 text-red-500 mb-4" />
                <h3 className="text-xl font-bold text-red-700 dark:text-red-400 mb-2">Prediction Failed</h3>
                <p className="text-red-600 dark:text-red-300">{error}</p>
                <p className="text-sm text-red-500 dark:text-red-400 mt-4">For demo purposes, the mock API will return a result.</p>
            </div>
        );
    }

    if (!result) {
        return (
            <div className="bg-agri-green-50 dark:bg-agri-green-950/20 border-2 border-dashed border-agri-green-200 dark:border-agri-green-800/50 rounded-2xl p-8 text-center h-full flex flex-col justify-center items-center">
                <div className="bg-white dark:bg-agri-green-900/50 p-4 rounded-full mb-4 shadow-sm">
                    <BarChart3 className="w-8 h-8 text-agri-green-400/70" />
                </div>
                <h3 className="text-xl font-medium text-agri-green-800 dark:text-agri-green-300 mb-2">Ready to Predict</h3>
                <p className="text-agri-green-600 dark:text-agri-green-400/70 max-w-xs mx-auto">
                    Adjust the values on the left and click "Predict Yield" to see intelligent forecasts.
                </p>
            </div>
        );
    }

    const { predictedYield, confidence, lowerBound, upperBound, factors } = result;

    // Determine color based on yield
    let statusColor = 'text-yellow-500';
    let bgColor = 'bg-yellow-500';
    let borderColor = 'border-yellow-200';
    let message = 'Average Yield';

    if (predictedYield > 4500) {
        statusColor = 'text-agri-green-600';
        bgColor = 'bg-agri-green-600';
        borderColor = 'border-agri-green-200';
        message = 'High Yield Expected';
    } else if (predictedYield < 3000) {
        statusColor = 'text-red-500';
        bgColor = 'bg-red-500';
        borderColor = 'border-red-200';
        message = 'Low Yield Warning';
    }

    // Calculate percentage relative to max (e.g. 6000)
    const percentage = Math.min(100, Math.max(0, (predictedYield / 6000) * 100));

    return (
        <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-xl p-6 md:p-8 h-full transition-colors duration-200 flex flex-col">
            <div className="mb-6 pb-6 border-b border-gray-100 dark:border-gray-800">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2 mb-1">
                    <TrendingUp className="text-agri-green-600" />
                    Forecast Results
                </h2>
                <p className="text-sm text-gray-500 dark:text-gray-400">{message} based on current inputs</p>
            </div>

            <div className="flex flex-col items-center justify-center py-4 mb-8">
                <div className="relative w-48 h-48 flex items-center justify-center mb-4">
                    {/* Gauge Background */}
                    <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                        <path
                            className="text-gray-100 dark:text-gray-800"
                            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="3"
                        />
                        <path
                            className={`${statusColor} transition-all duration-1000 ease-out`}
                            strokeDasharray={`${percentage}, 100`}
                            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="3"
                        />
                    </svg>
                    <div className="absolute flex flex-col items-center">
                        <span className={`text-4xl font-bold ${statusColor}`}>{predictedYield}</span>
                        <span className="text-xs text-gray-500 dark:text-gray-400 uppercase font-bold tracking-wider">kg/ha</span>
                    </div>
                </div>

                <div className={`flex items-center gap-2 px-3 py-1 rounded-full ${bgColor} bg-opacity-10 dark:bg-opacity-20`}>
                    <CheckCircle2 className={`w-4 h-4 ${statusColor}`} />
                    <span className={`text-sm font-medium ${statusColor}`}>
                        {confidence}% Confidence (±{Math.round((upperBound - lowerBound) / 2)})
                    </span>
                </div>
            </div>

            <div className="mt-auto">
                <h3 className="text-sm font-semibold text-gray-900 dark:text-white uppercase tracking-wider mb-4">Top Influencing Factors</h3>
                <div className="space-y-3">
                    {factors.map((factor, index) => (
                        <div key={index} className="space-y-1">
                            <div className="flex justify-between text-sm">
                                <span className="text-gray-600 dark:text-gray-300">{factor.name}</span>
                                <span className="font-medium text-gray-900 dark:text-white">{factor.impact > 0 ? '+' : ''}{factor.impact}%</span>
                            </div>
                            <div className="h-2 w-full bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
                                <div
                                    className={`h-full rounded-full ${factor.impact > 0 ? 'bg-agri-green-500' : 'bg-red-500'}`}
                                    style={{ width: `${Math.abs(factor.impact)}%` }}
                                ></div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

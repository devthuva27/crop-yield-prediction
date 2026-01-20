import {
    Chart as ChartJS,
    LinearScale,
    PointElement,
    LineElement,
    Tooltip,
    Legend,
} from 'chart.js';
import { Scatter } from 'react-chartjs-2';

ChartJS.register(
    LinearScale,
    PointElement,
    LineElement,
    Tooltip,
    Legend
);

export default function ModelPerformanceChart({ data }) {
    if (!data) return <div className="animate-pulse bg-gray-200 dark:bg-gray-800 h-64 rounded-xl"></div>;

    const minVal = Math.min(...data.actual, ...data.predicted) * 0.9;
    const maxVal = Math.max(...data.actual, ...data.predicted) * 1.1;

    const chartData = {
        datasets: [
            {
                label: 'Predictions',
                data: data.actual.map((val, i) => ({ x: val, y: data.predicted[i] })),
                backgroundColor: 'rgba(59, 130, 246, 0.6)', // Blue points
                borderColor: '#3b82f6',
                pointRadius: 4,
                pointHoverRadius: 6,
            },
            {
                label: 'Perfect Fit',
                data: [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }],
                type: 'line',
                borderColor: '#f97316', // Orange
                borderDash: [5, 5],
                borderWidth: 2,
                pointRadius: 0,
                fill: false,
            }
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
                labels: { color: '#888', usePointStyle: true }
            },
            title: {
                display: true,
                text: 'Model Accuracy on Test Data',
                color: '#fff',
                font: { size: 16, weight: 'bold' }
            },
            tooltip: {
                callbacks: {
                    label: (context) => `Actual: ${context.parsed.x.toFixed(0)}, Predicted: ${context.parsed.y.toFixed(0)}`
                }
            }
        },
        scales: {
            x: {
                type: 'linear',
                title: { display: true, text: 'Actual Yield (kg/ha)', color: '#666' },
                grid: { color: 'rgba(128, 128, 128, 0.1)' },
                min: minVal,
                max: maxVal
            },
            y: {
                title: { display: true, text: 'Predicted Yield (kg/ha)', color: '#666' },
                grid: { color: 'rgba(128, 128, 128, 0.1)' },
                min: minVal,
                max: maxVal
            }
        }
    };

    return (
        <div className="bg-white dark:bg-gray-900 p-6 rounded-2xl shadow-xl border border-gray-100 dark:border-gray-800 h-[400px]">
            <Scatter options={options} data={chartData} />
        </div>
    );
}

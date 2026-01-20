
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
);

export default function FeatureImportanceChart({ data }) {
    if (!data) return <div className="animate-pulse bg-gray-200 dark:bg-gray-800 h-64 rounded-xl"></div>;

    const sorted = [...data].sort((a, b) => b.importance - a.importance);

    const chartData = {
        labels: sorted.map(d => d.name),
        datasets: [
            {
                label: 'Importance Score',
                data: sorted.map(d => d.importance),
                backgroundColor: sorted.map((_, i) => `hsla(${120 - i * 10}, 70%, 50%, 0.7)`),
                borderColor: sorted.map((_, i) => `hsla(${120 - i * 10}, 70%, 50%, 1)`),
                borderWidth: 1,
                borderRadius: 8,
            },
        ],
    };

    const options = {
        indexAxis: 'y', // Horizontal
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            title: {
                display: true,
                text: 'What Factors Affect Yield?',
                color: '#fff',
                font: { size: 16, weight: 'bold' }
            },
        },
        scales: {
            x: {
                title: { display: true, text: 'Importance (%)', color: '#666' },
                grid: { display: false }
            },
            y: {
                grid: { display: false }
            }
        }
    };

    return (
        <div className="bg-white dark:bg-gray-900 p-6 rounded-2xl shadow-xl border border-gray-100 dark:border-gray-800 h-[400px]">
            <Bar options={options} data={chartData} />
        </div>
    );
}

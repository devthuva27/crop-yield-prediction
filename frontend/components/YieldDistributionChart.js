
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

export default function YieldDistributionChart({ data }) {
    if (!data) return <div className="animate-pulse bg-gray-200 dark:bg-gray-800 h-64 rounded-xl"></div>;

    const chartData = {
        labels: data.bins.slice(0, -1).map((b, i) => `${Math.round(b)}`),
        datasets: [
            {
                label: 'Frequency',
                data: data.counts,
                backgroundColor: 'rgba(16, 185, 129, 0.6)',
                borderColor: '#10b981',
                borderWidth: 1,
                borderRadius: 4,
                barPercentage: 1,
                categoryPercentage: 1,
            },
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            title: {
                display: true,
                text: 'Yield Distribution (kg/ha)',
                color: '#fff',
                font: { size: 16, weight: 'bold' }
            },
            tooltip: {
                callbacks: {
                    title: (items) => {
                        const index = items[0].dataIndex;
                        return `Yield: ${Math.round(data.bins[index])} - ${Math.round(data.bins[index + 1])} kg/ha`;
                    }
                }
            }
        },
        scales: {
            x: {
                title: { display: true, text: 'Yield (kg/ha)', color: '#666' }
            },
            y: {
                title: { display: true, text: 'Frequency', color: '#666' }
            }
        }
    };

    return (
        <div className="bg-white dark:bg-gray-900 p-6 rounded-2xl shadow-xl border border-gray-100 dark:border-gray-800 h-[300px]">
            <Bar options={options} data={chartData} />
        </div>
    );
}

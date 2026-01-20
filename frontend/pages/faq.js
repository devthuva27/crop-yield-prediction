import Header from '../components/Header';
import Footer from '../components/Footer';

export default function FAQ() {
    const faqs = [
        {
            question: "How accurate are the predictions?",
            answer: "Our model has achieved an R² score of 0.96 during testing, indicating very high accuracy. However, actual results can vary due to unmeasured factors like pests, diseases, or extreme weather events not captured in the average data."
        },
        {
            question: "Can I use this for any location?",
            answer: "The model is trained on data primarily from tropical and sub-tropical regions. While it can provide general estimates elsewhere, it is most accurate for regions with similar climatic conditions to the training data."
        },
        {
            question: "What units should I use for input?",
            answer: "Rainfall should be in millimeters (mm), Temperature in degrees Celsius (°C), and nutrient contents (K, P, K) in kilograms per hectare (kg/ha)."
        },
        {
            question: "Is my data saved?",
            answer: "No, we do not store your specific query data. The calculations are performed and the result is returned to you. We respect your privacy and data ownership."
        },
        {
            question: "Why do I see a range of values?",
            answer: "We provide a confidence interval (± value) because biological systems have inherent variability. The 'Predicted Yield' is the most likely outcome, but the true value is likely to fall within the displayed range."
        }
    ];

    return (
        <div className="min-h-screen flex flex-col bg-gray-50 dark:bg-black transition-colors duration-200">
            <Header />

            <main className="flex-grow py-12 px-4 sm:px-6 lg:px-8">
                <div className="max-w-4xl mx-auto">
                    <h1 className="text-3xl font-extrabold text-gray-900 dark:text-white mb-8 text-center">Frequently Asked Questions</h1>

                    <div className="space-y-6">
                        {faqs.map((faq, index) => (
                            <div key={index} className="bg-white dark:bg-gray-900 shadow-md rounded-xl p-6 hover:shadow-lg transition-shadow">
                                <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">{faq.question}</h3>
                                <p className="text-gray-600 dark:text-gray-400 leading-relaxed">{faq.answer}</p>
                            </div>
                        ))}
                    </div>

                    <div className="mt-12 text-center">
                        <p className="text-gray-600 dark:text-gray-400">
                            Can't find what you're looking for? <a href="#" className="text-agri-green-600 font-medium hover:underline">Contact our support team</a>
                        </p>
                    </div>
                </div>
            </main>

            <Footer />
        </div>
    );
}

import Header from '../components/Header';
import Footer from '../components/Footer';

export default function About() {
    return (
        <div className="min-h-screen flex flex-col bg-gray-50 dark:bg-black transition-colors duration-200">
            <Header />

            <main className="flex-grow py-12 px-4 sm:px-6 lg:px-8">
                <div className="max-w-4xl mx-auto">
                    <h1 className="text-3xl font-extrabold text-gray-900 dark:text-white mb-8">About AgriSense</h1>

                    <div className="bg-white dark:bg-gray-900 shadow-lg rounded-2xl p-8 space-y-6 text-gray-700 dark:text-gray-300">
                        <section>
                            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-3">Our Mission</h2>
                            <p>
                                AgriSense aims to bridge the gap between traditional farming and modern technology.
                                We believe that every farmer deserves access to data-driven insights to maximize their harvest potential.
                            </p>
                        </section>

                        <section>
                            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-3">How It Works</h2>
                            <p className="mb-4">
                                Our system uses advanced machine learning algorithms trained on historical agricultural data.
                                By analyzing key factors such as:
                            </p>
                            <ul className="list-disc pl-5 space-y-2">
                                <li>Rainfall patterns and water availability</li>
                                <li>Temperature variants</li>
                                <li>Soil composition (Nitrogen, Phosphorus, Potassium)</li>
                                <li>Crop specific characteristics</li>
                            </ul>
                        </section>

                        <section>
                            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-3">Why Use Prediction?</h2>
                            <p>
                                Yield prediction helps farmers in better planning, resource allocation, and risk management.
                                Knowing the expected yield can assist in marketing decisions and supply chain optimization.
                            </p>
                        </section>
                    </div>
                </div>
            </main>

            <Footer />
        </div>
    );
}

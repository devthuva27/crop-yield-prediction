import { useState } from 'react';
import { Droplets, Thermometer, FlaskConical, Sprout, Wheat } from 'lucide-react';

export default function InputForm({ onPredict, isLoading }) {
    const [formData, setFormData] = useState({
        rainfall: 400,
        temperature: 25,
        nitrogen: 80,
        phosphorus: 40,
        potassium: 50,
        crop: 'Rice'
    });

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: name === 'crop' ? value : Number(value)
        }));
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        onPredict(formData);
    };

    const handleReset = () => {
        setFormData({
            rainfall: 400,
            temperature: 25,
            nitrogen: 80,
            phosphorus: 40,
            potassium: 50,
            crop: 'Rice'
        });
    };

    const SliderInput = ({ label, name, value, min, max, unit, icon: Icon, color }) => (
        <div className="mb-6">
            <div className="flex justify-between items-center mb-2">
                <label htmlFor={name} className="flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300">
                    <Icon className={`w-4 h-4 ${color}`} />
                    {label}
                </label>
                <span className="text-sm font-bold text-gray-900 dark:text-white bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">
                    {value} <span className="text-xs font-normal text-gray-500">{unit}</span>
                </span>
            </div>
            <input
                type="range"
                id={name}
                name={name}
                min={min}
                max={max}
                value={value}
                onChange={handleChange}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700 accent-agri-green-600"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>{min}</span>
                <span>{max}</span>
            </div>
        </div>
    );

    return (
        <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-xl p-6 md:p-8 transition-colors duration-200">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
                <Sprout className="text-agri-green-600" />
                Enter Farm Conditions
            </h2>

            <form onSubmit={handleSubmit}>
                <SliderInput
                    label="Rainfall"
                    name="rainfall"
                    value={formData.rainfall}
                    min={100}
                    max={1000}
                    unit="mm"
                    icon={Droplets}
                    color="text-blue-500"
                />

                <SliderInput
                    label="Temperature"
                    name="temperature"
                    value={formData.temperature}
                    min={10}
                    max={40}
                    unit="°C"
                    icon={Thermometer}
                    color="text-red-500"
                />

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <SliderInput
                        label="Nitrogen (N)"
                        name="nitrogen"
                        value={formData.nitrogen}
                        min={0}
                        max={200}
                        unit="kg/ha"
                        icon={FlaskConical}
                        color="text-yellow-500"
                    />
                    <SliderInput
                        label="Phosphorus (P)"
                        name="phosphorus"
                        value={formData.phosphorus}
                        min={0}
                        max={100}
                        unit="kg/ha"
                        icon={FlaskConical}
                        color="text-purple-500"
                    />
                    <SliderInput
                        label="Potassium (K)"
                        name="potassium"
                        value={formData.potassium}
                        min={0}
                        max={100}
                        unit="kg/ha"
                        icon={FlaskConical}
                        color="text-pink-500"
                    />
                </div>

                <div className="mb-8">
                    <label htmlFor="crop" className="flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        <Wheat className="w-4 h-4 text-amber-600" />
                        Crop
                    </label>
                    <div className="relative">
                        <select
                            id="crop"
                            name="crop"
                            value={formData.crop}
                            onChange={handleChange}
                            className="block w-full pl-3 pr-10 py-3 text-base border-gray-300 dark:border-gray-700 focus:outline-none focus:ring-agri-green-500 focus:border-agri-green-500 sm:text-sm rounded-lg dark:bg-gray-800 dark:text-white"
                        >
                            <option value="Tea">Tea</option>
                            <option value="Rice">Rice</option>
                            <option value="Rubber">Rubber</option>
                            <option value="Cinnamon">Cinnamon</option>
                            <option value="Sugarcane">Sugarcane</option>
                        </select>
                    </div>
                </div>

                <div className="flex gap-4 pt-4 border-t border-gray-100 dark:border-gray-800">
                    <button
                        type="button"
                        onClick={handleReset}
                        className="flex-1 px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg text-sm font-medium text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-agri-green-500 transition-colors"
                    >
                        Reset
                    </button>
                    <button
                        type="submit"
                        disabled={isLoading}
                        className="flex-1 flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-agri-green-600 hover:bg-agri-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-agri-green-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-[1.02]"
                    >
                        {isLoading ? (
                            <span className="flex items-center gap-2">
                                <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Predicting...
                            </span>
                        ) : 'Predict Yield'}
                    </button>
                </div>
            </form>
        </div>
    );
}

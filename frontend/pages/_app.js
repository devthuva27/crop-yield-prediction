import '../styles/globals.css'
import Head from 'next/head'

function MyApp({ Component, pageProps }) {
    return (
        <>
            <Head>
                <title>AgriSense - Intelligent Crop Yield Prediction</title>
                <meta name="description" content="Predict your harvest yield accurately with AI driven insights." />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <link rel="icon" href="/favicon.ico" />
            </Head>
            <Component {...pageProps} />
        </>
    )
}

export default MyApp

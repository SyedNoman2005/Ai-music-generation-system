document.addEventListener('DOMContentLoaded', () => {
    const engineInput = document.getElementById('engine');
    const tempInput = document.getElementById('temperature');
    const tempVal = document.getElementById('temp-val');
    
    const lengthInput = document.getElementById('length');
    const lengthVal = document.getElementById('length-val');
    
    const generateBtn = document.getElementById('generate-btn');
    const btnText = generateBtn.querySelector('.btn-text');
    const loader = document.getElementById('loader');
    
    const statusMsg = document.getElementById('status-message');
    const resultSection = document.getElementById('result-section');
    const downloadLink = document.getElementById('download-link');
    const pianoRollImg = document.getElementById('piano-roll-img');

    tempInput.addEventListener('input', (e) => {
        tempVal.textContent = parseFloat(e.target.value).toFixed(1);
    });

    lengthInput.addEventListener('input', (e) => {
        lengthVal.textContent = e.target.value;
    });

    generateBtn.addEventListener('click', async () => {
        statusMsg.classList.add('hidden');
        resultSection.classList.add('hidden');
        pianoRollImg.classList.add('hidden');
        
        generateBtn.disabled = true;
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');

        const engine = engineInput.value;
        const temperature = parseFloat(tempInput.value);
        const length = parseInt(lengthInput.value);

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ engine, temperature, length })
            });

            const data = await response.json();

            if (response.ok) {
                downloadLink.href = data.file_url;
                
                if (data.image_url) {
                    // Cache bust the image URL so it updates if generated multiple times
                    pianoRollImg.src = data.image_url + "?t=" + new Date().getTime();
                    pianoRollImg.classList.remove('hidden');
                }
                
                resultSection.classList.remove('hidden');
            } else {
                statusMsg.textContent = data.detail || 'An unknown error occurred.';
                statusMsg.classList.remove('hidden');
            }
        } catch (error) {
            statusMsg.textContent = 'Failed to connect to the server.';
            statusMsg.classList.remove('hidden');
        } finally {
            generateBtn.disabled = false;
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    });
});

const video = document.getElementById('camera');
const cameraSelect = document.getElementById('camera-select');
const startButton = document.getElementById('start-button');
const applyButton = document.getElementById('apply-button');
const predictionResult = document.getElementById('prediction-result');
const exportButton = document.getElementById('export-button');
const importModelInput = document.getElementById('import-model');
const chineseWordInput = document.getElementById('chinese-word');
const logCanvas = document.getElementById('log-canvas');
const ctx = logCanvas.getContext('2d');

// Store labels and data
let labels = [];
let images = [];

// Create and compile a new model from scratch
let model = tf.sequential();
model.add(tf.layers.conv2d({
    inputShape: [224, 224, 3],
    filters: 16,
    kernelSize: 3,
    activation: 'relu'
}));
model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({ units: 3, activation: 'softmax' })); // Adjust units based on labels
model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
});

// Get available cameras
async function getCameras() {
    try {
        // Request permission to access media devices, including video input devices
        await navigator.mediaDevices.getUserMedia({ video: true });

        // List all available video input devices
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');

        // Clear existing options
        cameraSelect.innerHTML = '';

        // Add video input devices to the select dropdown
        videoDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || `Camera ${index + 1}`;
            cameraSelect.appendChild(option);
        });

        // Automatically select the first available camera
        if (videoDevices.length > 0) {
            await setupCamera(videoDevices[0].deviceId);
        }
    } catch (error) {
        console.error('Error accessing media devices:', error);
    }
}

// Start the selected camera
async function setupCamera(deviceId) {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: { exact: deviceId } }
    });
    video.srcObject = stream;
}

// Capture a frame from the video
function captureFrame(video) {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return tf.browser.fromPixels(canvas);
}

// Display logs on canvas
function logToCanvas(message) {
    ctx.clearRect(0, 0, logCanvas.width, logCanvas.height); // Clear the canvas
    ctx.font = '16px Arial';
    ctx.fillStyle = 'black';
    ctx.fillText(message, 10, 50); // Display log message
}

// Apply learning with the inputted Chinese word
applyButton.addEventListener('click', async () => {
    const chineseWord = chineseWordInput.value.trim();
    if (chineseWord) {
        // Capture the frame
        const frame = captureFrame(video);
        const resizedFrame = tf.image.resizeBilinear(frame, [224, 224]).div(255).expandDims(0);
        
        // Convert word to one-hot encoded label
        if (!labels.includes(chineseWord)) {
            labels.push(chineseWord);
        }
        const labelIndex = labels.indexOf(chineseWord);
        const oneHotLabel = tf.oneHot([labelIndex], labels.length);

        // Store the data for training
        images.push({ image: resizedFrame, label: oneHotLabel });

        // Log learning
        logToCanvas(`Learning applied for: ${chineseWord}`);
    }
});

// Train the model
async function trainModel() {
    const xs = tf.concat(images.map(item => item.image));
    const ys = tf.concat(images.map(item => item.label));

    await model.fit(xs, ys, {
        epochs: 10,
        callbacks: { 
            onEpochEnd: (epoch, logs) => {
                logToCanvas(`Epoch ${epoch}: loss = ${logs.loss}`);
            }
        }
    });
}

// Continuous monitoring and prediction
function monitorAndPredict() {
    setInterval(async () => {
        const frame = captureFrame(video);
        const resizedFrame = tf.image.resizeBilinear(frame, [224, 224]).div(255).expandDims(0);

        const prediction = model.predict(resizedFrame);
        const predictedClass = prediction.argMax(-1).dataSync()[0];

        if (labels[predictedClass]) {
            predictionResult.innerText = `Prediction: ${labels[predictedClass]}`;
            logToCanvas(`Predicted word: ${labels[predictedClass]}`);
        }
        tf.dispose([frame, resizedFrame]);
    }, 2000); // Predict every 2 seconds
}

// Start continuous monitoring on button click
startButton.addEventListener('click', monitorAndPredict);

// Export the model
exportButton.addEventListener('click', async () => {
    await model.save('downloads://my-chinese-word-model');
    logToCanvas('Model exported.');
});

// Import the model
importModelInput.addEventListener('change', async (event) => {
    const files = event.target.files;
    if (files.length > 0) {
        model = await tf.loadLayersModel(tf.io.browserFiles(files));
        logToCanvas('Model imported.');
    }
});

// Initialize camera dropdown and set up the first camera
getCameras().then(() => {
    cameraSelect.addEventListener('change', () => {
        setupCamera(cameraSelect.value);
    });
});


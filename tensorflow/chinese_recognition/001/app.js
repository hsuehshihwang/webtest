const video = document.getElementById('camera');
const cameraSelect = document.getElementById('camera-select');
const startButton = document.getElementById('start-button');
const applyButton = document.getElementById('apply-button');
const predictionResult = document.getElementById('prediction-result');
const exportButton = document.getElementById('export-button');
const importModelInput = document.getElementById('import-model');
const chineseWordInput = document.getElementById('chinese-word');

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
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');
    videoDevices.forEach((device, index) => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.text = device.label || `Camera ${index + 1}`;
        cameraSelect.appendChild(option);
    });
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

        console.log(`Learning applied for word: ${chineseWord}`);
    }
});

// Train the model
async function trainModel() {
    const xs = tf.concat(images.map(item => item.image));
    const ys = tf.concat(images.map(item => item.label));

    await model.fit(xs, ys, {
        epochs: 10,  // Adjust epochs
        callbacks: { onEpochEnd: (epoch, logs) => console.log(`Epoch ${epoch}: loss = ${logs.loss}`) }
    });
}

// Run prediction
startButton.addEventListener('click', async () => {
    const frame = captureFrame(video);
    const resizedFrame = tf.image.resizeBilinear(frame, [224, 224]).div(255).expandDims(0);
    const prediction = model.predict(resizedFrame);
    const predictedClass = prediction.argMax(-1).dataSync()[0];
    predictionResult.innerText = `Prediction: ${labels[predictedClass]}`;
    tf.dispose([frame, resizedFrame]);
});

// Export the model
exportButton.addEventListener('click', async () => {
    await model.save('downloads://my-chinese-word-model');
    console.log('Model exported.');
});

// Import the model
importModelInput.addEventListener('change', async (event) => {
    const files = event.target.files;
    if (files.length > 0) {
        model = await tf.loadLayersModel(tf.io.browserFiles(files));
        console.log('Model imported.');
    }
});

// Initialize camera dropdown and set up the first camera
getCameras().then(() => {
    cameraSelect.addEventListener('change', () => {
        setupCamera(cameraSelect.value);
    });
    setupCamera(cameraSelect.value);  // Start the first camera by default
});


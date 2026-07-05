document.addEventListener('DOMContentLoaded', () => {
    const videoElement = document.getElementById('video-feed');
    const canvasElement = document.getElementById('output-canvas');
    const canvasCtx = canvasElement.getContext('2d');
    const loader = document.getElementById('loader');
    
    const typedTextElement = document.getElementById('typed-text');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const confidenceFill = document.getElementById('confidence-fill');
    
    const toggleDebug = document.getElementById('toggle-debug');
    
    let typedBuffer = "";
    let points = [];
    let isDrawing = false;
    let isGesturing = false;
    let lastDrawTime = Date.now();
    let lastGestureTime = 0;
    let predictTimeout = null;
    let spaceTimeout = null;
    let finalizeTimeout = null;
    
    const PAUSE_THRESHOLD_MS = 600; 
    const SPACE_THRESHOLD_MS = 1500;
    const FINALIZE_THRESHOLD_MS = 3000;
    const CONFIDENCE_THRESHOLD = 0.5;
    
    const emergencyKeywords = ["help", "pain", "doctor", "fire", "hurt", "emergency"];

    // --- Buttons ---
    document.getElementById('btn-space').addEventListener('click', () => {
        typedBuffer += " ";
        updateDisplay();
    });
    
    document.getElementById('btn-backspace').addEventListener('click', () => {
        if(typedBuffer.length > 0) {
            typedBuffer = typedBuffer.slice(0, -1);
            updateDisplay();
        }
    });
    
    document.getElementById('btn-clear').addEventListener('click', () => {
        typedBuffer = "";
        updateDisplay();
        confidenceFill.style.width = "0%";
    });

    function updateDisplay() {
        typedTextElement.textContent = typedBuffer;
    }
    
    function setStatus(state) {
        statusDot.className = "dot"; // reset
        if(state === 'wait') {
            statusText.textContent = "Waiting for hand...";
        } else if(state === 'detect') {
            statusDot.classList.add('active');
            statusText.textContent = "Hand detected";
        } else if(state === 'draw') {
            statusDot.classList.add('drawing');
            statusText.textContent = "Writing...";
        }
    }

    function speakText(text) {
        if (!text || text.trim() === "") return;
        
        const utterance = new SpeechSynthesisUtterance(text);
        
        // Context Classification
        const lowerText = text.toLowerCase();
        const isEmergency = emergencyKeywords.some(kw => lowerText.includes(kw));
        
        if (isEmergency) {
            utterance.rate = 1.2; // faster
            utterance.pitch = 1.5; // higher pitch
            utterance.volume = 1.0;
            document.querySelector('.glass-panel').style.boxShadow = "0 8px 32px rgba(239, 68, 68, 0.6)"; // Red glow
            setTimeout(() => {
                document.querySelector('.glass-panel').style.boxShadow = "0 8px 32px rgba(0, 0, 0, 0.3)";
            }, 3000);
        } else {
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
        }
        
        window.speechSynthesis.speak(utterance);
    }

    function checkTimeouts() {
        if (!isDrawing && points.length === 0 && typedBuffer.length > 0) {
            const timeSinceLastDraw = Date.now() - lastDrawTime;
            
            // Add space if 1.5s passed and last char isn't space
            if (timeSinceLastDraw > SPACE_THRESHOLD_MS && timeSinceLastDraw <= FINALIZE_THRESHOLD_MS) {
                if (!typedBuffer.endsWith(" ")) {
                    typedBuffer += " ";
                    updateDisplay();
                }
            }
            
            // Finalize and speak if 3s passed
            if (timeSinceLastDraw > FINALIZE_THRESHOLD_MS) {
                // Ensure we only speak once per sentence by checking if we already spoke it
                // We'll clear the buffer after speaking
                const textToSpeak = typedBuffer.trim();
                if (textToSpeak.length > 0) {
                    speakText(textToSpeak);
                    typedBuffer = ""; // clear for next sentence
                    updateDisplay();
                    confidenceFill.style.width = "0%";
                    lastDrawTime = Date.now(); // reset so it doesn't loop
                }
            }
        }
    }
    
    // Periodically check for spacing and finalization
    setInterval(checkTimeouts, 300);

    // --- Logic functions ---
    function isOnlyIndexUp(landmarks) {
        const indexUp = landmarks[8].y < landmarks[6].y;
        const middleUp = landmarks[12].y < landmarks[10].y;
        const ringUp = landmarks[16].y < landmarks[14].y;
        const pinkyUp = landmarks[20].y < landmarks[18].y;
        return indexUp && !(middleUp || ringUp || pinkyUp);
    }
    
    function isGestureMode(landmarks) {
        // Simple heuristic: if not air writing and not fully closed fist, maybe it's a gesture.
        // For simplicity, if >= 2 fingers are up, we consider it gesture mode.
        const indexUp = landmarks[8].y < landmarks[6].y;
        const middleUp = landmarks[12].y < landmarks[10].y;
        const ringUp = landmarks[16].y < landmarks[14].y;
        const pinkyUp = landmarks[20].y < landmarks[18].y;
        
        let count = 0;
        if(indexUp) count++;
        if(middleUp) count++;
        if(ringUp) count++;
        if(pinkyUp) count++;
        
        return count >= 2;
    }

    function extractGestureFeatures(landmarks) {
        const pts = landmarks.map(lm => ({x: lm.x, y: lm.y, z: lm.z}));
        const wrist = pts[0];
        const translated = pts.map(pt => ({
            x: pt.x - wrist.x,
            y: pt.y - wrist.y,
            z: pt.z - wrist.z
        }));
        
        const mcp = translated[9];
        let scale = Math.sqrt(mcp.x**2 + mcp.y**2 + mcp.z**2);
        if (scale < 1e-6) scale = 1.0;
        
        const features = [];
        translated.forEach(pt => {
            features.push(pt.x / scale);
            features.push(pt.y / scale);
            features.push(pt.z / scale);
        });
        return features;
    }

    async function sendForGesturePrediction(features) {
        try {
            const res = await fetch('/predict_gesture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features: features })
            });
            const data = await res.json();
            
            if (data.prediction && data.confidence > CONFIDENCE_THRESHOLD) {
                // If the word isn't already the last word we added, append it
                const words = typedBuffer.trim().split(" ");
                if (words[words.length - 1] !== data.prediction) {
                    if (typedBuffer.length > 0 && !typedBuffer.endsWith(" ")) {
                        typedBuffer += " ";
                    }
                    typedBuffer += data.prediction + " ";
                    updateDisplay();
                    confidenceFill.style.width = `${Math.round(data.confidence * 100)}%`;
                    lastDrawTime = Date.now(); // reset timers
                }
            }
        } catch(e) {
            console.error("Gesture prediction error:", e);
        }
    }

    async function sendForPrediction() {
        if(points.length < 5) {
            points = [];
            return;
        }
        
        const pts = [...points];
        points = [];
        
        try {
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ points: pts })
            });
            
            const data = await res.json();
            
            if(data.prediction && data.confidence > CONFIDENCE_THRESHOLD) {
                typedBuffer += data.prediction;
                updateDisplay();
                confidenceFill.style.width = `${Math.round(data.confidence * 100)}%`;
            }
        } catch(e) {
            console.error("Prediction error:", e);
        }
    }

    // --- MediaPipe setup ---
    const hands = new window.Hands({locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }});

    hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.6
    });

    hands.onResults((results) => {
        if(loader.style.display !== 'none') {
            loader.style.display = 'none';
        }
        
        // Match canvas to video size
        if (canvasElement.width !== videoElement.videoWidth) {
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;
        }

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        const w = canvasElement.width;
        const h = canvasElement.height;

        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const landmarks = results.multiHandLandmarks[0];
            
            if(toggleDebug.checked) {
                window.drawConnectors(canvasCtx, landmarks, window.HAND_CONNECTIONS, {color: '#ffffff', lineWidth: 2});
                window.drawLandmarks(canvasCtx, landmarks, {color: '#10b981', lineWidth: 1, radius: 3});
            }

            if (isOnlyIndexUp(landmarks)) {
                setStatus('draw');
                isDrawing = true;
                isGesturing = false;
                lastDrawTime = Date.now();
                
                // Add point (mirror x so that what the user draws is correctly oriented for the CNN)
                const px = (1 - landmarks[8].x) * w;
                const py = landmarks[8].y * h;
                points.push([px, py]);
                
                if (predictTimeout) clearTimeout(predictTimeout);
            } else if (isGestureMode(landmarks)) {
                setStatus('detect'); // Gesture mode
                isDrawing = false;
                isGesturing = true;
                
                // Only predict gesture once every 2 seconds to avoid spamming the same word
                if (Date.now() - lastGestureTime > 2000) {
                    lastGestureTime = Date.now();
                    const features = extractGestureFeatures(landmarks);
                    sendForGesturePrediction(features);
                }
            } else {
                setStatus('detect');
                isDrawing = false;
                isGesturing = false;
            }
        } else {
            setStatus('wait');
            isDrawing = false;
            isGesturing = false;
        }

        // Draw current stroke
        if (points.length > 1) {
            canvasCtx.beginPath();
            canvasCtx.moveTo(w - points[0][0], points[0][1]);
            for (let i = 1; i < points.length; i++) {
                canvasCtx.lineTo(w - points[i][0], points[i][1]);
            }
            canvasCtx.strokeStyle = '#3b82f6';
            canvasCtx.lineWidth = 4;
            canvasCtx.lineCap = 'round';
            canvasCtx.lineJoin = 'round';
            canvasCtx.stroke();
        }

        // Trigger prediction if stopped drawing
        if (!isDrawing && points.length > 5 && (Date.now() - lastDrawTime > PAUSE_THRESHOLD_MS)) {
            // We use a timeout to avoid spamming while hand is slightly jittering
            if(!predictTimeout) {
                predictTimeout = setTimeout(() => {
                    sendForPrediction();
                    predictTimeout = null;
                }, 100);
            }
        }

        canvasCtx.restore();
    });

    const camera = new window.Camera(videoElement, {
        onFrame: async () => {
            await hands.send({image: videoElement});
        },
        width: 640,
        height: 480
    });

    camera.start();
});

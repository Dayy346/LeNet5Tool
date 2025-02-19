import React, { useState } from "react";
import "./App.css";
import UploadDataset from "./UploadDataset";
import DownloadModel from "./DownloadModel";
import axios from "axios";

function App() {
  const [trainingStatus, setTrainingStatus] = useState("");
  const [trainingProgress, setTrainingProgress] = useState(0); // Progress bar percentage
  const [numEpochs, setNumEpochs] = useState(1); // Default epochs
  const [numClasses, setNumClasses] = useState(10); // Default to 10 classes
  const [useGrayscale, setUseGrayscale] = useState(false); // New state for Grayscale/RGB option
  const [testImageStatus, setTestImageStatus] = useState(""); // Status for image testing
  const [trainAccuracy, setTrainAccuracy] = useState(null); // State for train accuracy
  const [testAccuracy, setTestAccuracy] = useState(null);   // State for test accuracy
  
  const handleDatasetUpload = async (file) => {
    try {
      setTrainingStatus("Processing dataset...");
      setTrainingProgress(0);

      if (file) {
        // User uploaded a file
        const formData = new FormData();
        formData.append("file", file);
        formData.append("useGrayscale", useGrayscale); // Include grayscale setting in form data

        const uploadResponse = await axios.post(`${process.env.REACT_APP_API_BASE_URL}/upload-dataset`, formData);

        if (uploadResponse.data.message) {
          setTrainingStatus(uploadResponse.data.message);
          getFirstEpochTime();
        }
      } else {
        // No file uploaded, default to MNIST dataset
        setTrainingStatus("No dataset uploaded. Defaulting to MNIST dataset...");
        const mnistResponse = await axios.post(`${process.env.REACT_APP_API_BASE_URL}/upload-dataset`, { useGrayscale });


        if (mnistResponse.data.message) {
          setTrainingStatus(mnistResponse.data.message);
          getFirstEpochTime();
        }
      }
    } catch (error) {
      console.error("Error processing dataset:", error);
      setTrainingStatus("Failed to process dataset. Please try again.");
    }
  };

  const getFirstEpochTime = async () => {
    try {
      setTrainingStatus("Calculating first epoch duration...");
      const epochTimeResponse = await axios.post(`${process.env.REACT_APP_API_BASE_URL}/epoch-time`, { num_classes: numClasses });


      if (epochTimeResponse.data.epoch_time) {
        const epochTime = epochTimeResponse.data.epoch_time;
        setTrainingStatus(`First epoch took ${epochTime.toFixed(2)} seconds.`);
        startTraining(epochTime);
      }
    } catch (error) {
      console.error("Error calculating first epoch time:", error);
      setTrainingStatus(
        "Failed to calculate first epoch time. Please try again."
      );
    }
  };
  const startTraining = async (epochTime) => {
    try {
      setTrainingStatus("Training started...");
      let estimatedTotalTime = epochTime * numEpochs;

      // Simulate progress updates
      let progress = 0;
      const interval = setInterval(() => {
        progress += (100 / estimatedTotalTime) * epochTime;
        if (progress >= 99) {
          clearInterval(interval);
        } else {
          setTrainingProgress(progress);
        }
      }, epochTime * 1000);

      const trainResponse = await axios.post(`${process.env.REACT_APP_API_BASE_URL}/train`, {
        num_epochs: numEpochs,
        num_classes: numClasses, // Send number of classes to backend
      });

      if (trainResponse.data.message) {
        setTrainingStatus("Training completed!");
        setTrainingProgress(100); // Training completed
        clearInterval(interval); // Ensure interval is cleared

        // Set train and test accuracies
        setTrainAccuracy(trainResponse.data.train_accuracy);
        setTestAccuracy(trainResponse.data.test_accuracy);
      }
    } catch (error) {
      console.error("Error during training:", error);
      setTrainingStatus("Training failed. Please try again.");
      setTrainingProgress(0);
    }
  };

  const handleImageTest = async (file) => {
    try {
      if (!file) {
        alert("Please upload an image for testing.");
        return;
      }
  
      const formData = new FormData();
      formData.append("file", file);
      formData.append("useGrayscale", useGrayscale); // Send grayscale option to the backend
  
      setTestImageStatus("Testing image...");
      const response = await axios.post(`${process.env.REACT_APP_API_BASE_URL}/test-image`, formData);
  
      if (response.data.predicted_class) {
        setTestImageStatus(
          `Image classified successfully! Predicted Class: ${response.data.predicted_class}`
        );
      } else if (response.data.error) {
        setTestImageStatus(`Error: ${response.data.error}`);
      }
    } catch (error) {
      console.error("Error testing image:", error);
      setTestImageStatus("Error testing the image. Please try again.");
    }
  };
  

  return (
    <div className="App">
      <header className="App-header">
        <h1 style={{ color: "#4caf50", fontSize: "2.5rem", margin: "20px 0" }}>
          Dayyan's LeNet5 Based CNN Model Generator
        </h1>
        <div className="section">
          <h2 style={{ color: "#3f51b5", fontSize: "1.8rem" }}>Mode Selection</h2>
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <label
              style={{
                fontSize: "1rem",
                color: useGrayscale ? "#555" : "#4caf50",
              }}
            >
              RGB
            </label>
            <input
              type="checkbox"
              checked={useGrayscale}
              onChange={() => setUseGrayscale(!useGrayscale)}
              style={{ transform: "scale(1.5)" }}
            />
            <label
              style={{
                fontSize: "1rem",
                color: useGrayscale ? "#4caf50" : "#555",
              }}
            >
              Grayscale
            </label>
          </div>
        </div>
        <div className="section">
          <h2 style={{ color: "#3f51b5", fontSize: "1.8rem" }}>
            Epochs (Test/Train Iterations)
          </h2>
          <input
            type="number"
            min="1"
            value={numEpochs}
            onChange={(e) => setNumEpochs(e.target.value)}
            style={{
              padding: "10px",
              fontSize: "1rem",
              width: "80px",
              marginBottom: "20px",
              textAlign: "center",
            }}
          />
        </div>
        <div className="section">
          <h2 style={{ color: "#3f51b5", fontSize: "1.8rem" }}>
            Number of Classes in Dataset
          </h2>
          <input
            type="number"
            min="1"
            value={numClasses}
            onChange={(e) => setNumClasses(e.target.value)}
            style={{
              padding: "10px",
              fontSize: "1rem",
              width: "80px",
              marginBottom: "20px",
              textAlign: "center",
            }}
          />
        </div>
        <div className="section">
          <h2 style={{ color: "#3f51b5", fontSize: "1.8rem" }}>
            Upload Dataset Here
          </h2>
          <UploadDataset onUpload={handleDatasetUpload} />
        </div>
        <div className="section">
          <h2 style={{ color: "#3f51b5", fontSize: "1.8rem" }}>
            Training Progress
          </h2>
          <div
            style={{
              width: "100%",
              backgroundColor: "#f4f4f4",
              borderRadius: "10px",
              marginTop: "10px",
              border: "1px solid #ddd",
              padding: "5px",
            }}
          >
            <div
              style={{
                width: `${trainingProgress}%`,
                backgroundColor: "#4caf50",
                height: "20px",
                borderRadius: "10px",
                transition: "width 0.5s",
              }}
            ></div>
          </div>
          <p style={{ marginTop: "10px", fontSize: "1rem", color: "#555" }}>
            {trainingStatus}
          </p>
        </div>
        {trainAccuracy !== null && testAccuracy !== null && (
          <div className="section">
            <h2 style={{ color: "#3f51b5", fontSize: "1.8rem" }}>
              Final Model Accuracy
            </h2>
            <p style={{ fontSize: "1rem", color: "#555" }}>
              <strong>Train Accuracy:</strong> {trainAccuracy.toFixed(2)}%
            </p>
            <p style={{ fontSize: "1rem", color: "#555" }}>
              <strong>Test Accuracy:</strong> {testAccuracy.toFixed(2)}%
            </p>
          </div>
        )}
        <div className="section">
          <h2 style={{ color: "#3f51b5", fontSize: "1.8rem" }}>
            Test a Single Image
          </h2>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => handleImageTest(e.target.files[0])}
          />
          <p style={{ marginTop: "10px", fontSize: "1rem", color: "#555" }}>
            {testImageStatus}
          </p>
        </div>
        <div className="section">
          <h2 style={{ color: "#3f51b5", fontSize: "1.8rem" }}>
            Download Trained Model Here!
          </h2>
          <DownloadModel />
        </div>
      </header>
    </div>
  );
}

export default App;
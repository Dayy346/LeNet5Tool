import React, { useState } from "react";
import "./App.css";
import UploadDataset from "./UploadDataset";
import DownloadModel from "./DownloadModel";
import axios from "axios";


function App() {
  const [trainingStatus, setTrainingStatus] = useState("");
  const [trainingProgress, setTrainingProgress] = useState(0); // Progress bar percentage
  const [numEpochs, setNumEpochs] = useState(1); // Default epochs
  const [numClasses, setNumClasses] = useState(5); // Default to 5 classes


  const handleDatasetUpload = async (file) => {
    try {
      setTrainingStatus("Uploading dataset...");
      setTrainingProgress(0);


      const formData = new FormData();
      formData.append("file", file);


      const uploadResponse = await axios.post(
        "http://127.0.0.1:5000/upload-dataset",
        formData
      );


      if (uploadResponse.data.message) {
        setTrainingStatus(uploadResponse.data.message);
        getFirstEpochTime();
      }
    } catch (error) {
      console.error("Error uploading dataset:", error);
      setTrainingStatus("Failed to upload dataset. Please try again.");
    }
  };


  const getFirstEpochTime = async () => {
    try {
      setTrainingStatus("Calculating first epoch duration...");
      const epochTimeResponse = await axios.post(
        "http://127.0.0.1:5000/epoch-time",
        { num_classes: numClasses } // Ensure num_classes is sent
      );
 
      if (epochTimeResponse.data.epoch_time) {
        const epochTime = epochTimeResponse.data.epoch_time;
        setTrainingStatus(`First epoch took ${epochTime.toFixed(2)} seconds.`);
        startTraining(epochTime);
      }
    } catch (error) {
      console.error("Error calculating first epoch time:", error);
      setTrainingStatus("Failed to calculate first epoch time. Please try again.");
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
 
      const trainResponse = await axios.post("http://127.0.0.1:5000/train", {
        num_epochs: numEpochs,
        num_classes: numClasses, // Send number of classes to backend
      });
 
      if (trainResponse.data.message) {
        setTrainingStatus("Training completed!");
        setTrainingProgress(100); // Training completed
        clearInterval(interval); // Ensure interval is cleared
      }
    } catch (error) {
      console.error("Error during training:", error);
      setTrainingStatus("Training failed. Please try again.");
      setTrainingProgress(0);
    }
  };
 
  return (
    <div className="App">
      <header className="App-header">
        <h1 style={{ color: "#4caf50", fontSize: "2.5rem", margin: "20px 0" }}>
          LeNet5 Based CNN Model Generator
        </h1>
        <div className="section">
          <h2 style={{ color: "#3f51b5", fontSize: "1.8rem" }}>Epochs (aka test/train iterations)</h2>
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
          <h2 style={{ color: "#3f51b5", fontSize: "1.8rem" }}>Number of Classes Present in Dataset</h2>
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
          <h2 style={{ color: "#3f51b5", fontSize: "1.8rem" }}>Upload Dataset Here</h2>
          <UploadDataset onUpload={handleDatasetUpload} />
        </div>
        <div className="section">
          <h2 style={{ color: "#3f51b5", fontSize: "1.8rem" }}>Training Progress</h2>
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
        <div className="section">
          <h2 style={{ color: "#3f51b5", fontSize: "1.8rem" }}>Download Trained Model Here!</h2>
          <DownloadModel />
        </div>
      </header>
    </div>
  );
}


export default App;



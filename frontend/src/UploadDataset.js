import React, { useState } from "react";

function UploadDataset({ onUpload }) {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]); // Update the selected file
  };

  const handleUpload = () => {
    // Call onUpload with the selectedFile or null if no file is selected
    onUpload(selectedFile); 
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload} style={{ marginTop: "10px" }}>
        Upload and Train
      </button>
      {!selectedFile && (
        <p style={{ color: "gray", fontSize: "0.9rem", marginTop: "5px" }}>
          No file selected. Defaulting to MNIST dataset if you proceed.
        </p>
      )}
    </div>
  );
}

export default UploadDataset;

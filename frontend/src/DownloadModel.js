import React from "react";


function DownloadModel() {
  const handleDownload = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/download-model");


      if (!response.ok) {
        throw new Error("Failed to download the model.");
      }


      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);


      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "CustomLeNetModel.pth");
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error("Download error:", error.message);
    }
  };


  return (
    <div>
      <button onClick={handleDownload}>Download Trained Model</button>
    </div>
  );
}


export default DownloadModel;



